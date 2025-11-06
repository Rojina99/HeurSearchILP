# final_parallel_no_predicate_invention.py

import argparse
import os
import pdb
import pickle
import random
import time
import numpy as np
from typing import List, Dict, Callable, Tuple, Any, Optional
import re
import sys
import shutil

# Multiprocessing
from torch.multiprocessing import get_context, Queue
from multiprocessing.process import BaseProcess

from environments.env_utils import get_environment
from deepxube.environments.environment_abstract import Environment, State, Goal, EnvGrndAtoms
from search.astar import AStar, get_path, Node
from deepxube.search.search_utils import is_valid_soln
from heur.heur_utils import get_popper_examples, aggregate_all_solutions
from logic.popper_utils import ClauseSet
from heur.test_prolog_heur import HeurILP
from nnet.nnet_utils import load_and_process_clause_to_get_unique_clause
from deepxube.utils import misc_utils
from deepxube.utils.data_utils import Logger

# --- Configuration ---
DEFAULT_STATES_PER_DEPTH_TOTAL_FOR_ILP = 1000
ASTAR_HEURISTIC_CALL_BATCH_SIZE = 500
ILP_CONVERGENCE_ITERATIONS = 3
DEFAULT_BK_FILE_NAME = "bk.pl"
DEFAULT_BIAS_FILE_NAME = "bias.pl"
NUM_PARALLEL_ASTAR_WORKERS = 2
DEFAULT_ASTAR_WEIGHT = 1.0
# NEW: Configuration for A* expansion retry
DEFAULT_MAX_ASTAR_EXPANSIONS_RETRY_MULTIPLIER = 5  # Multiply by this factor on retry
DEFAULT_MAX_ASTAR_EXPANSION_INCREMENT_ON_RETRY = 5000 # Or add a fixed large number

# --- PureILPHeuristic Class --- (Assuming this is using the version with binary search in HeurILP)
class PureILPHeuristic:
    def __init__(self, env: EnvGrndAtoms, popper_run_base_dir: str, initial_rules_file: str = None):
        self.env = env
        self.popper_run_base_dir = popper_run_base_dir
        self.heur_ilp = HeurILP(env, popper_run_base_dir=self.popper_run_base_dir)
        self.rules_file = initial_rules_file
        self.load_rules(self.rules_file)

    def load_rules(self, rules_file: str):
        self.heur_ilp.clear()
        self.rules_file = rules_file
        if rules_file and os.path.exists(rules_file) and os.path.getsize(rules_file) > 0:
            try:
                # Assuming master_ilp_rules_file stores Dict[int, List[str]] directly
                # as per save_rules_to_file
                with open(rules_file, 'rb') as f:
                    loaded_rules_raw: Dict[int, List[str]] = pickle.load(f)
                
                # load_and_process_clause_to_get_unique_clause expects a filename
                # and processes Dict[depth, List[clauses]]
                # It ensures clauses are unique per depth and across depths (min depth wins)
                # The output is Dict[float, List[str]] (float keys for HeurILP)
                processed_rules = load_and_process_clause_to_get_unique_clause(rules_file)

                if processed_rules:
                    for depth_float, clauses in processed_rules.items():
                        if clauses:
                            for clause in clauses:
                                self.heur_ilp.add(depth_float, clause) # HeurILP expects float depth
                    print(f"    PureILPHeuristic: Loaded and processed rules from {rules_file}")
                else:
                    print(f"    PureILPHeuristic: No rules found in {rules_file} after processing by load_and_process_clause_to_get_unique_clause.")
            except Exception as e:
                print(f"    PureILPHeuristic: Error loading or processing rules from {rules_file}: {e}")
        else:
            print(f"    PureILPHeuristic: No pre-existing rules file found or file is empty ({rules_file}).")

    def get_heur(self, states: List[State], goals: List[Goal] = None) -> np.ndarray:
        ilp_values_optional: List[Optional[float]] = self.heur_ilp.get_heur(states)
        processed_heuristics = [val if val is not None else 0.0 for val in ilp_values_optional]
        return np.array(processed_heuristics, dtype=np.float64)


def prepare_popper_input_files(
    target_depth: int, base_program_dir: str,
    popper_run_dir_for_target_depth: str, all_learned_rules: Dict[int, List[str]]
):
    if not os.path.exists(popper_run_dir_for_target_depth):
        os.makedirs(popper_run_dir_for_target_depth)
    base_bk_path = os.path.join(base_program_dir, DEFAULT_BK_FILE_NAME)
    base_bias_path = os.path.join(base_program_dir, DEFAULT_BIAS_FILE_NAME)
    target_bk_path = os.path.join(popper_run_dir_for_target_depth, DEFAULT_BK_FILE_NAME)
    target_bias_path = os.path.join(popper_run_dir_for_target_depth, DEFAULT_BIAS_FILE_NAME)

    bk_content = ""
    if os.path.exists(base_bk_path):
        with open(base_bk_path, 'r') as f: bk_content += f.read()
    else:
        print(f"Warning: Base BK file not found at {base_bk_path}")

    with open(target_bk_path, 'w') as f: f.write(bk_content)

    bias_content = ""
    if os.path.exists(base_bias_path):
        with open(base_bias_path, 'r') as f: bias_content += f.read()
    else:
        print(f"Warning: Base bias file not found at {base_bias_path}")

    with open(target_bias_path, 'w') as f:
        f.write(bias_content)
    print(f"  Prepared Popper input files (BK & Bias for depth {target_depth}) in {popper_run_dir_for_target_depth}")


# --- A* Example Generation (Parallelized) ---
def astar_example_generation_worker( # Accepts max_astar_steps_per_instance
    worker_id: int, env_name: str, popper_run_base_dir_for_heur: str,
    master_rules_file_for_heur: str, start_states_chunk: List[State],
    goals_chunk: List[Goal], target_path_cost: int, astar_search_batch_size: int,
    max_astar_steps_per_instance: int, # This is the crucial parameter for retry
    result_queue: Queue, astar_weight: float
):
    process_env, _ = get_environment(env_name)
    if not isinstance(process_env, EnvGrndAtoms):
        print(f"[Worker {worker_id}] Error: Environment is not EnvGrndAtoms type.")
        result_queue.put((worker_id, [], [], max_astar_steps_per_instance)) # Return original steps
        return
        
    worker_heuristic_obj = PureILPHeuristic(process_env, popper_run_base_dir_for_heur, master_rules_file_for_heur)
    local_positive_examples, local_negative_examples = [], []

    if not start_states_chunk:
        result_queue.put((worker_id, local_positive_examples, local_negative_examples, max_astar_steps_per_instance))
        return

    astar_search = AStar(process_env)
    weights_for_astar = [astar_weight] * len(start_states_chunk)
    astar_search.add_instances(start_states_chunk, goals_chunk, weights_for_astar, worker_heuristic_obj)

    # print(f"    [Worker {worker_id}] Starting A* with max_steps_per_instance: {max_astar_steps_per_instance}") # Debug
    while not all(instance.finished for instance in astar_search.instances):
        active_instances = [inst for inst in astar_search.instances if not inst.finished]
        if not active_instances: break
        for inst_idx, inst in enumerate(astar_search.instances):
            if not inst.finished and inst.step_num >= max_astar_steps_per_instance: # Use >=
                inst.finished = True
                # print(f"    [Worker {worker_id}] Instance {inst_idx} reached max A* steps ({max_astar_steps_per_instance}).")
        if all(instance.finished for instance in astar_search.instances): break
        astar_search.step(worker_heuristic_obj, astar_search_batch_size, verbose=False)

    for i, instance in enumerate(astar_search.instances):
        original_start_state = start_states_chunk[i]
        if instance.goal_node:
            path_cost = instance.goal_node.path_cost
            if path_cost >= target_path_cost: local_positive_examples.append(original_start_state)
            elif path_cost < target_path_cost: local_negative_examples.append(original_start_state)
        # elif instance.finished: # TODO need to change logic later, not for 8 puzzle
        #     min_f_open = float('inf')
        #     if instance.open_set: min_f_open = instance.open_set[0][0]
        #     if min_f_open < target_path_cost or not instance.open_set:
        #         local_negative_examples.append(original_start_state)
    
    # Return the max_astar_steps_per_instance actually used by this worker for this run
    result_queue.put((worker_id, local_positive_examples, local_negative_examples, max_astar_steps_per_instance))


def run_astar_for_example_generation_parallel(
    env: EnvGrndAtoms, env_name_str: str, heuristic_fn_obj_main_thread: PureILPHeuristic,
    num_states_for_astar_run: int, max_random_walk_steps_from_goal: int,
    target_path_cost: int, astar_search_batch_size: int,
    # This becomes the initial max_astar_steps_per_instance
    initial_max_astar_expansions_per_instance: int, 
    num_workers: int, astar_weight: float,
    # NEW: Parameters for retry logic
    retry_expansion_multiplier: float,
    retry_expansion_increment: int,
    allow_retry: bool = True # Flag to control if retry is allowed for this call
) -> Tuple[List[State], List[State], int]: # Return updated max_expansions

    start_time = time.time()
    current_max_astar_expansions = initial_max_astar_expansions_per_instance
    attempt = 0
    max_attempts = 2 if allow_retry else 1 # Initial attempt + 1 retry if allowed

    overall_positive_examples, overall_negative_examples = [], []
    print("    ############# Allow Retry", allow_retry, "#################")
    while attempt < max_attempts:
        attempt += 1
        if attempt > 1: # This is a retry
            print(f"    [A* Gen Parallel Retry Attempt {attempt-1}] No positive examples found. Increasing A* expansions.")
            # Increase by multiplier OR a fixed large increment, whichever is larger
            multiplied_expansions = int(current_max_astar_expansions * retry_expansion_multiplier)
            incremented_expansions = current_max_astar_expansions + retry_expansion_increment
            current_max_astar_expansions = max(multiplied_expansions, incremented_expansions)
            print(f"    [A* Gen Parallel Retry Attempt {attempt-1}] New max_astar_expansions_per_instance: {current_max_astar_expansions}")
        
        # Clear previous results for retry
        overall_positive_examples.clear()
        overall_negative_examples.clear()

        print(f"  [A* Gen Parallel] Target Cost: {target_path_cost}. States: ~{num_states_for_astar_run}. Workers: {num_workers}. A* Weight: {astar_weight}. Max Expansions: {current_max_astar_expansions}.")

        if num_states_for_astar_run == 0:
            return [], [], current_max_astar_expansions # Return current (possibly initial)

        actual_max_scramble = max(0, max_random_walk_steps_from_goal)
        if num_states_for_astar_run > 0:
            scramble_depths_list = [random.randint(0, actual_max_scramble) for _ in range(num_states_for_astar_run)]
        else:
            scramble_depths_list = []

        print(f"    [A* Gen Parallel] Generating {len(scramble_depths_list)} total start states for A*.")
        if scramble_depths_list:
            print(f"    [A* Gen Parallel] Scramble depths: min={min(scramble_depths_list) if scramble_depths_list else 'N/A'}, "
                  f"max={max(scramble_depths_list) if scramble_depths_list else 'N/A'}, "
                  f"avg={(sum(scramble_depths_list)/len(scramble_depths_list)) if scramble_depths_list else 'N/A':.2f}")

        start_states_all, goals_all = env.get_start_goal_pairs(scramble_depths_list)
        if not start_states_all:
            return [], [], current_max_astar_expansions

        state_counts_per_worker = misc_utils.split_evenly(len(start_states_all), num_workers)
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        processes: List[BaseProcess] = []
        
        current_start_idx = 0
        active_workers = 0
        for i in range(num_workers):
            num_states_for_this_worker = state_counts_per_worker[i]
            if num_states_for_this_worker == 0: continue
            active_workers +=1
            
            chunk_end_idx = current_start_idx + num_states_for_this_worker
            states_subset = start_states_all[current_start_idx:chunk_end_idx]
            goals_subset = goals_all[current_start_idx:chunk_end_idx]
            
            master_rules_file = heuristic_fn_obj_main_thread.rules_file
            popper_run_base_dir = heuristic_fn_obj_main_thread.popper_run_base_dir

            p = ctx.Process(target=astar_example_generation_worker, args=(
                i, env_name_str, popper_run_base_dir, master_rules_file,
                states_subset, goals_subset, target_path_cost,
                astar_search_batch_size, 
                current_max_astar_expansions, # Pass current (possibly increased) max expansions
                result_queue, astar_weight
            ))
            processes.append(p)
            p.daemon = True
            p.start()
            current_start_idx = chunk_end_idx

        max_expansions_used_by_any_worker_in_this_run = initial_max_astar_expansions_per_instance

        for _ in range(active_workers): # Iterate based on active workers
            timeout_time = 14400 # 4 hour timeout
            try:
                # Worker returns (worker_id, pos_ex, neg_ex, expansions_limit_it_used)
                worker_id, pos_ex, neg_ex, worker_expansions_limit = result_queue.get(timeout=timeout_time) # 4 hour timeout
                overall_positive_examples.extend(pos_ex)
                overall_negative_examples.extend(neg_ex)
                # Track the actual limit used if it was dynamically set per worker (not the case here, but good for future)
                # For now, all workers in a run use current_max_astar_expansions
                # max_expansions_used_by_any_worker_in_this_run = max(max_expansions_used_by_any_worker_in_this_run, worker_expansions_limit)

                print(f"    [A* Gen Parallel] Worker {worker_id} finished (used limit {worker_expansions_limit}), found Pos={len(pos_ex)}, Neg={len(neg_ex)}")
            except Exception as e:
                print(f"    [A* Gen Parallel] Worker result retrieval error or timeout: {e} with timeout time {timeout_time}")
        
        for p in processes:
            p.join(timeout=60) 
            if p.is_alive():
                print(f"    [A* Gen Parallel] Worker process {p.pid} did not terminate, killing."); p.terminate(); p.join()

        print(f" Before Deduplication: {len(overall_positive_examples), len(overall_negative_examples)}")

        overall_positive_examples = list(dict.fromkeys(overall_positive_examples))
        overall_negative_examples = list(dict.fromkeys(overall_negative_examples))

        print(f"    [A* Gen Parallel] Classification Complete (Attempt {attempt}): Total Unique Positives={len(overall_positive_examples)}, Total Unique Negatives={len(overall_negative_examples)}")

        print(f"    After Deduplication: {len(overall_positive_examples), len(overall_negative_examples)}")

        if overall_positive_examples: # Found positive examples, no need to retry
            break 
        if not allow_retry and not overall_positive_examples: # No retry allowed and no positives
            print(f"    [A* Gen Parallel] No positive examples found and no retry allowed.")
            break
        if attempt >= max_attempts and not overall_positive_examples: # Max attempts reached
            print(f"    [A* Gen Parallel] Max retry attempts reached, no positive examples found.")
            break

    print("    [A* Gen Parallel]  Time to run:  %.2f" % (time.time() - start_time))

    # The value of current_max_astar_expansions reflects the highest value used if retries occurred.
    return overall_positive_examples, overall_negative_examples, current_max_astar_expansions


# --- Utility Functions for Saving/Loading --- (Ensure these handle int keys correctly)
def save_rules_to_file(filepath: str, all_learned_rules: Dict[int, List[str]]):
    processed_rules_to_save = {int(k): list(set(v)) for k, v in all_learned_rules.items()}
    with open(filepath, "wb") as f: pickle.dump(processed_rules_to_save, f)
    print(f"  Rules saved to {filepath}")

def load_rules_from_file(filepath: str) -> Dict[int, List[str]]:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
            return {int(k): v for k,v in loaded_data.items()}
    return {}

def save_main_checkpoint(filepath: str, current_depth_to_learn: int, all_accumulated_rules: Dict[int, List[str]],
                         current_max_astar_expansions: int): # Save the adaptive parameter
    checkpoint_data = {
        'next_depth_to_learn': current_depth_to_learn + 1,
        'all_accumulated_rules': {int(k): v for k, v in all_accumulated_rules.items()},
        'max_astar_expansions_per_instance': current_max_astar_expansions # Save current value
    }
    with open(filepath, "wb") as f: pickle.dump(checkpoint_data, f)
    print(f"  Main checkpoint saved to {filepath} (next depth: {current_depth_to_learn + 1}, max_exp: {current_max_astar_expansions})")

def load_main_checkpoint(filepath: str, default_max_astar_expansions: int) -> Tuple[int, Dict[int, List[str]], int]:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
            rules = checkpoint_data.get('all_accumulated_rules', {})
            # Load saved max_astar_expansions, or use the default from args if not in checkpoint
            max_exp = checkpoint_data.get('max_astar_expansions_per_instance', default_max_astar_expansions)
            return checkpoint_data.get('next_depth_to_learn', 1), \
                   {int(k): v for k,v in rules.items()}, \
                   max_exp
    return 1, {}, default_max_astar_expansions # Return default from args if no checkpoint

def save_learned_rules_to_txt(filepath: str, all_learned_rules: Dict[int, List[str]]):
    try:
        sorted_depths = sorted([int(k) for k in all_learned_rules.keys()])
        with open(filepath, "w") as f:
            f.write("=" * 40 + "\nFinal Learned ILP Rules by Depth\n" + "=" * 40 + "\n\n")
            if not sorted_depths: f.write("No rules were learned.\n")
            else:
                for depth in sorted_depths:
                    rules = all_learned_rules.get(depth, []) 
                    if not rules and float(depth) in all_learned_rules:
                        rules = all_learned_rules[float(depth)]

                    f.write("-" * 15 + f" Depth {depth} " + "-" * 15 + "\n")
                    if rules:
                        for rule_str in rules: f.write(f"{rule_str}\n")
                    else: f.write("(No rules learned for this depth)\n")
                    f.write("\n")
            f.write("=" * 40 + "\n")
        print(f"  Formatted learned rules saved to {filepath}")
    except Exception as e: print(f"Error saving formatted rules to {filepath}: {e}")


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Iteratively learn ILP rules for N-Puzzle using A* and Popper.")
    parser.add_argument('--env', type=str, default="puzzle8", help="Environment name (e.g., puzzle8)")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory to save learned rules and checkpoints.")
    parser.add_argument('--learned_program_dir', type=str, required=True, help="Directory containing template bk.pl and bias.pl for Popper.")
    parser.add_argument('--popper_runner_path', type=str, default='logic/popper_runner.py', help="Path to popper_runner.py utility.")
    parser.add_argument('--max_depth_to_learn', type=int, default=31, help="Maximum lower bound depth to learn rules for.")
    parser.add_argument('--states_per_astar_total', type=int, default=DEFAULT_STATES_PER_DEPTH_TOTAL_FOR_ILP, help="Total states for A* example generation for one depth's ILP refinement.")
    parser.add_argument('--max_scramble_steps', type=int, default=60, help="Max random walk steps from goal for generating A* start states.")
    parser.add_argument('--astar_heuristic_batch_size', type=int, default=ASTAR_HEURISTIC_CALL_BATCH_SIZE, help="Batch size for A* when calling heuristic function.")
    parser.add_argument('--ilp_convergence_iterations', type=int, default=ILP_CONVERGENCE_ITERATIONS, help="Max iterations for ILP rule refinement for a single depth.")
    parser.add_argument('--max_astar_expansions_per_instance', type=int, default=20000, help="Initial maximum A* node expansions for a single problem instance.") # Renamed to 'initial'
    parser.add_argument('--num_astar_workers', type=int, default=NUM_PARALLEL_ASTAR_WORKERS, help="Number of parallel workers for A* example generation.")
    parser.add_argument('--astar_weight', type=float, default=DEFAULT_ASTAR_WEIGHT, help="Weight for Weighted A* search in example generation.")
    parser.add_argument('--astar_retry_multiplier', type=float, default=DEFAULT_MAX_ASTAR_EXPANSIONS_RETRY_MULTIPLIER, help="Multiplier for max_astar_expansions on retry if no positive examples.")
    parser.add_argument('--astar_retry_increment', type=int, default=DEFAULT_MAX_ASTAR_EXPANSION_INCREMENT_ON_RETRY, help="Fixed increment for max_astar_expansions on retry.")

    time_all = time.time()

    args = parser.parse_args()

    # Initialize current_max_astar_expansions from args, this will be updated if retries happen
    # and also loaded from checkpoint.
    current_max_astar_expansions = args.max_astar_expansions_per_instance

    args.model_dir = os.path.join(args.model_dir, f'env_{args.env}_pdir_{args.learned_program_dir.strip("/")}_depth_{args.max_depth_to_learn}_'
                                            f'ip_states_each_depth_{args.states_per_astar_total}_ss_{args.max_scramble_steps}_'
                                            f'a_star_bs_{args.astar_heuristic_batch_size}_ci_{args.ilp_convergence_iterations}_'
                                            f'a_star_exp_start_{args.max_astar_expansions_per_instance}_'
                                            f'workers_{args.num_astar_workers}_rm_{args.astar_retry_multiplier}_ri_{args.astar_retry_increment}_no_predicate_invention')

    # print("Model directory", args.model_dir)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    popper_run_base_dir = os.path.join(args.model_dir, "popper_runs")
    if not os.path.exists(popper_run_base_dir):
        os.makedirs(popper_run_base_dir)

    master_ilp_rules_file = os.path.join(args.model_dir, "master_learned_ilp_rules.pkl")
    main_checkpoint_file = os.path.join(args.model_dir, "main_loop_checkpoint.pkl")
    logger_file = os.path.join(args.model_dir, "output.txt")

    sys.stdout = Logger(logger_file, "a")

    print("\nModel directory   ", args.model_dir, "\n")

    env, _ = get_environment(args.env)
    if not isinstance(env, EnvGrndAtoms):
        raise TypeError("Environment must be an instance of EnvGrndAtoms for ILP rule learning.")

    heuristic_fn_wrapper = PureILPHeuristic(env, popper_run_base_dir=popper_run_base_dir, initial_rules_file=master_ilp_rules_file)

    # Load checkpoint, including the potentially adapted max_astar_expansions
    start_depth_main_loop, all_accumulated_rules_loaded, loaded_max_astar_expansions = load_main_checkpoint(main_checkpoint_file, args.max_astar_expansions_per_instance)
    all_accumulated_rules: Dict[int, List[str]] = {int(k): v for k, v in all_accumulated_rules_loaded.items()}

    # Use loaded_max_astar_expansions if checkpoint existed, otherwise stick to args.
    if os.path.exists(main_checkpoint_file): # Check if checkpoint was actually loaded
        current_max_astar_expansions = loaded_max_astar_expansions
        print(f"Resuming. Loaded max_astar_expansions_per_instance: {current_max_astar_expansions}")
    else:
        print(f"Starting new run. Initial max_astar_expansions_per_instance: {current_max_astar_expansions}")


    if all_accumulated_rules:
        save_rules_to_file(master_ilp_rules_file, all_accumulated_rules)
        heuristic_fn_wrapper.load_rules(master_ilp_rules_file)
        print(f"Resuming. Loaded {len(all_accumulated_rules)} depths of rules from checkpoint. Next depth to learn: {start_depth_main_loop}")
    else:
        print("Starting new run or no rules in checkpoint.")

    for current_depth_to_learn in range(start_depth_main_loop, args.max_depth_to_learn + 1):
        start_time_depth = time.time()
        print(f"\n===== Learning rules for TARGET DEPTH/LOWER BOUND = {current_depth_to_learn} =====")
        current_popper_run_dir = os.path.join(popper_run_base_dir, str(current_depth_to_learn))

        prepare_popper_input_files(
            current_depth_to_learn, args.learned_program_dir,
            current_popper_run_dir, all_accumulated_rules 
        )

        accumulated_pos_examples_this_depth: List[State] = []
        accumulated_neg_examples_this_depth: List[State] = []
        previous_clauses_for_this_depth = ClauseSet([])
        final_learned_clauses_for_this_depth: List[str] = []
        
        states_per_refinement_iter = args.states_per_astar_total // args.ilp_convergence_iterations
        if states_per_refinement_iter == 0 and args.states_per_astar_total > 0 : states_per_refinement_iter = args.states_per_astar_total
        if states_per_refinement_iter < 10 and args.states_per_astar_total > 0 : states_per_refinement_iter = max(10, args.states_per_astar_total)

        # Adaptive A* expansions logic for this depth
        # The run_astar_for_example_generation_parallel will handle the retry with its own copy of current_max_astar_expansions
        # We need to update our main loop's current_max_astar_expansions if it was increased by a successful retry.

        for ilp_refinement_iter in range(args.ilp_convergence_iterations):
            start_time_iterations = time.time()
            print(f"  [ILP Refine iter {ilp_refinement_iter + 1}/{args.ilp_convergence_iterations} for depth {current_depth_to_learn}]")
            
            # Allow retry only for the first ILP refinement iteration, or always allow?
            # For now, allow retry in each ILP iteration if needed.
            # allow_current_retry = True
            allow_current_retry = False

            start_time_a_star = time.time()
            new_pos_examples, new_neg_examples, updated_max_expansions_after_run = run_astar_for_example_generation_parallel(
                env, args.env, heuristic_fn_wrapper,
                states_per_refinement_iter, args.max_scramble_steps,
                current_depth_to_learn, args.astar_heuristic_batch_size,
                current_max_astar_expansions, # Pass the current adaptive value
                args.num_astar_workers,
                args.astar_weight,
                args.astar_retry_multiplier,
                args.astar_retry_increment,
                allow_retry=allow_current_retry
            )

            print("  Time to obtain positive and negative samples:  %.2f" % (time.time() - start_time_a_star))

            # Update the global max_astar_expansions if it was increased by the run
            if updated_max_expansions_after_run > current_max_astar_expansions:
                print(f"    [Main Loop] max_astar_expansions_per_instance updated from {current_max_astar_expansions} to {updated_max_expansions_after_run} due to retry.")
                current_max_astar_expansions = updated_max_expansions_after_run

            accumulated_pos_examples_this_depth.extend(new_pos_examples)
            accumulated_neg_examples_this_depth.extend(new_neg_examples)
            accumulated_pos_examples_this_depth = list(dict.fromkeys(accumulated_pos_examples_this_depth))
            accumulated_neg_examples_this_depth = list(dict.fromkeys(accumulated_neg_examples_this_depth))
            
            print(f"    Total unique examples for depth {current_depth_to_learn} after A*: Pos={len(accumulated_pos_examples_this_depth)}, Neg={len(accumulated_neg_examples_this_depth)}")

            if not accumulated_pos_examples_this_depth and ilp_refinement_iter == 0 : # No positive examples even after potential retry on first iteration
                 print(f"    [Warning] No positive examples generated for depth {current_depth_to_learn}. Popper may not find rules.")

            start_time_popper = time.time()

            current_rules_for_this_depth_iter = []
            if accumulated_pos_examples_this_depth or accumulated_neg_examples_this_depth: # Run Popper if any examples
                popper_output_dict = get_popper_examples(
                    accumulated_pos_examples_this_depth, accumulated_neg_examples_this_depth,
                    current_depth_to_learn, args.learned_program_dir,
                    current_popper_run_dir,
                    args.popper_runner_path,
                    logger_file
                )
                current_rules_for_this_depth_iter = popper_output_dict.get(current_depth_to_learn, [])
            else: print("    No positive or negative examples to run Popper.")

            print("\n Time to obtain popper rules:  %.2f" % (time.time() - start_time_popper))
            
            if current_rules_for_this_depth_iter:
                 print(f"    Rules learned by Popper in this iteration: {len(current_rules_for_this_depth_iter)}")
            else: print(f"    Popper found no rules in this iteration.")

            final_learned_clauses_for_this_depth = current_rules_for_this_depth_iter
            current_clauseset_for_this_depth = ClauseSet(final_learned_clauses_for_this_depth)

            if previous_clauses_for_this_depth is not None and current_clauseset_for_this_depth == previous_clauses_for_this_depth:
                print(f"    ILP rules for depth {current_depth_to_learn} CONVERGED.")
                break
            previous_clauses_for_this_depth = current_clauseset_for_this_depth
            if ilp_refinement_iter == args.ilp_convergence_iterations - 1:
                print(f"    Max ILP refinement iterations reached for depth {current_depth_to_learn}.")

            print("Time to run iteration:  %.2f" % (time.time() - start_time_iterations))

        rules_changed_this_depth = False
        if final_learned_clauses_for_this_depth:
            if current_depth_to_learn not in all_accumulated_rules or \
               ClauseSet(final_learned_clauses_for_this_depth) != ClauseSet(all_accumulated_rules.get(current_depth_to_learn,[])):
                rules_changed_this_depth = True
                all_accumulated_rules[current_depth_to_learn] = final_learned_clauses_for_this_depth
                print(f"  Final rules for depth {current_depth_to_learn} (count: {len(final_learned_clauses_for_this_depth)}).")
            else: print(f"  No change to rules for depth {current_depth_to_learn} compared to master list.")
        elif current_depth_to_learn in all_accumulated_rules:
            print(f"  No rules found for depth {current_depth_to_learn}, removing existing from master list.")
            del all_accumulated_rules[current_depth_to_learn]
            rules_changed_this_depth = True
        else: print(f"  No rules finalized for depth {current_depth_to_learn}.")

        if rules_changed_this_depth or (not os.path.exists(master_ilp_rules_file) and final_learned_clauses_for_this_depth):
            save_rules_to_file(master_ilp_rules_file, all_accumulated_rules)
            heuristic_fn_wrapper.load_rules(master_ilp_rules_file)
        
        aggregate_all_solutions(popper_run_base_dir, os.path.join(popper_run_base_dir, 'all_popper_solutions.txt'))
        # Save the potentially updated current_max_astar_expansions to checkpoint
        save_main_checkpoint(main_checkpoint_file, current_depth_to_learn, all_accumulated_rules, current_max_astar_expansions)
        
        if current_depth_to_learn >= args.max_depth_to_learn:
             print(f"Reached max_depth_to_learn {args.max_depth_to_learn}. Stopping.")
             break

        print("Time per depth:  %.2f" % (time.time() - start_time_depth))

    print("\n===== Direct ILP rule learning process finished. =====")
    formatted_rules_filepath = os.path.join(args.model_dir, "final_learned_rules.txt")
    save_learned_rules_to_txt(formatted_rules_filepath, all_accumulated_rules)
    print("Script complete.")

    print("Time to run script:  %.2f" % (time.time() - time_all))


if __name__ == "__main__":
    main()