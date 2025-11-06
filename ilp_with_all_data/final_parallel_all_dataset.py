# final_parallel_all_dataset.py

# python final_parallel_all_dataset.py --env puzzle8 --model_dir puzzle_8_output_depth_31_all_dataset --learned_program_dir puzzle8_rules_source_new --popper_runner_path logic/popper_runner_timeout.py --max_depth_to_learn 31 --ilp_convergence_iterations 1 --data_file data/puzzle8.pkl --time_out 1800

import argparse
import os
import pdb
import pickle
import random
import time
import numpy as np
from typing import List, Dict, Set, Callable, Tuple, Any, Optional
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
from heur.heur_utils import get_popper_examples_timeout, aggregate_all_solutions
from logic.popper_utils import ClauseSet
from heur.test_prolog_heur import HeurILP
from nnet.nnet_utils import load_and_process_clause_to_get_unique_clause
from deepxube.utils import misc_utils
from deepxube.utils.data_utils import Logger

# --- Configuration ---
DEFAULT_STATES_PER_DEPTH_TOTAL_FOR_ILP = 1000
ILP_CONVERGENCE_ITERATIONS = 1
DEFAULT_BK_FILE_NAME = "bk.pl"
DEFAULT_BIAS_FILE_NAME = "bias.pl"

# --- BK and Bias File Preparation ---
def format_learned_rule_for_bk(rule_str: str, new_head_predicate: str) -> str:
    formatted_rule = ""
    if ":-" in rule_str:
        head, body = rule_str.split(":-", 1)
        new_head = re.sub(r"^\s*(f|clause|goal)\s*\(\s*([A-Za-z_0-9]+)\s*\)", rf"{new_head_predicate}(\2)", head.strip())
        body_cleaned = body.strip()
        if not body_cleaned: formatted_rule = f"{new_head}."
        else:
            formatted_rule = f"{new_head} :- {body_cleaned}"
            if not formatted_rule.endswith('.'): formatted_rule += "."
    else:
        new_head = re.sub(r"^\s*(f|clause|goal)\s*\(\s*([A-Za-z_0_9]+)\s*\)", rf"{new_head_predicate}(\2)", rule_str.strip())
        formatted_rule = f"{new_head}."
    if formatted_rule.endswith(".."): formatted_rule = formatted_rule[:-1]
    return formatted_rule

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
    added_predicates_for_bias = []
    added_predicates_for_direction_and_type_for_bias = []
    
    bk_content = ""
    if os.path.exists(base_bk_path):
        with open(base_bk_path, 'r') as f: bk_content += f.read()
    else: print(f"Warning: Base BK file not found at {base_bk_path}")

    bk_content += "\n\n% --- Appended Learned Rules from Lower Depths (Unique Predicates) --- % \n"
    sorted_prev_depths = sorted([d for d in all_learned_rules.keys() if int(d) < target_depth])

    for depth_val_int in sorted_prev_depths:
        rules = all_learned_rules.get(depth_val_int, [])
        if rules:
            bk_content += f"\n% Rules from depth {depth_val_int}\n"
            for rule_idx, rule_str in enumerate(rules):
                unique_pred_name = f"clause_depth_{depth_val_int}_rule_{rule_idx + 1}"
                if depth_val_int == target_depth - 1:
                    added_predicates_for_bias.append(unique_pred_name)
                else:
                    added_predicates_for_direction_and_type_for_bias.append(unique_pred_name)

                formatted_rule = format_learned_rule_for_bk(rule_str, unique_pred_name)
                bk_content += f"{formatted_rule}\n"
    with open(target_bk_path, 'w') as f: f.write(bk_content)
    
    bias_content = ""
    if os.path.exists(base_bias_path):
        with open(base_bias_path, 'r') as f: bias_content += f.read()
    else: print(f"Warning: Base bias file not found at {base_bias_path}")
    
    bias_content += "\n\n% --- Appended Bias Directives for Learned Predicates (Unique Rules) --- % \n"
    unique_added_predicates = sorted(list(set(added_predicates_for_bias)))
    # unique_added_predicates_direction_and_type = sorted(list(set(added_predicates_for_direction_and_type_for_bias)))

    # for pred_name in unique_added_predicates_direction_and_type:
    #     bias_content += f"type({pred_name}, (list,)).\n"
    #     bias_content += f"direction({pred_name}, (in,)).\n\n"

    # bias_content += "\n\n% --- Appended Bias Directives and Body for Learned Predicates (Unique Rules) --- % \n"

    for pred_name in unique_added_predicates:
        bias_content += f"body_pred({pred_name}, 1).\n"
        bias_content += f"type({pred_name}, (list,)).\n"
        bias_content += f"direction({pred_name}, (in,)).\n\n"
    
    with open(target_bias_path, 'w') as f:
        f.write(bias_content)
    print(f"  Prepared Popper input files (BK & Bias for depth {target_depth}) in {popper_run_dir_for_target_depth}")


def all_dataset_for_example_generation(env: Environment,  target_path_cost: int, data_file: str):
    """
    @param env: Environment
    """

    ctg_to_states: Dict[int, Set[State]] = pickle.load(open(data_file, "rb"))

    positive_examples, negative_examples = [], []

    # pdb.set_trace()

    for cost_to_go, num_states_i in ctg_to_states.items():
            if cost_to_go >= target_path_cost:
                positive_examples.extend(list(ctg_to_states[cost_to_go]))
            elif cost_to_go < target_path_cost:
                negative_examples.extend(list(ctg_to_states[cost_to_go]))

    # pdb.set_trace()

    return positive_examples, negative_examples

# TODO How is ILP done if all exampels are positive, Is loss minimization required, save tarrget file ?

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

def save_main_checkpoint(filepath: str, current_depth_to_learn: int, all_accumulated_rules: Dict[int, List[str]]): # Save the adaptive parameter
    checkpoint_data = {
        'next_depth_to_learn': current_depth_to_learn + 1,
        'all_accumulated_rules': {int(k): v for k, v in all_accumulated_rules.items()}
    }
    with open(filepath, "wb") as f: pickle.dump(checkpoint_data, f)
    print(f"  Main checkpoint saved to {filepath} (next depth: {current_depth_to_learn + 1})")

def load_main_checkpoint(filepath: str) -> Tuple[int, Dict[int, List[str]], int]:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
            rules = checkpoint_data.get('all_accumulated_rules', {})
            # Load saved max_astar_expansions, or use the default from args if not in checkpoint
            return checkpoint_data.get('next_depth_to_learn', 1), {int(k): v for k,v in rules.items()}

    return 1, {} # Return default from args if no checkpoint

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
    parser.add_argument('--popper_runner_path', type=str, default='logic/popper_runner_timeout.py', help="Path to popper_runner.py utility.")
    parser.add_argument('--max_depth_to_learn', type=int, default=31, help="Maximum lower bound depth to learn rules for.")
    parser.add_argument('--ilp_convergence_iterations', type=int, default=ILP_CONVERGENCE_ITERATIONS, help="Max iterations for ILP rule refinement for a single depth.")
    parser.add_argument('--data_file', type=str, default='data/puzzle8.pkl', help="Path to 8 puzzle data")
    parser.add_argument('--time_out', type=int, default=1800, help="popper timeout")


    time_all = time.time()

    args = parser.parse_args()

    args.model_dir = os.path.join(args.model_dir, f'env_{args.env}_pdir_{args.learned_program_dir.strip("/")}_depth_{args.max_depth_to_learn}_'
                                            f'ci_{args.ilp_convergence_iterations}_time_out_{args.time_out}_all')

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

    print("\nModel directory   ", args.model_dir, "\n", "Timeout: ", args.time_out, "\n")

    env, _ = get_environment(args.env)
    if not isinstance(env, EnvGrndAtoms):
        raise TypeError("Environment must be an instance of EnvGrndAtoms for ILP rule learning.")

    # heuristic_fn_wrapper = PureILPHeuristic(env, popper_run_base_dir=popper_run_base_dir, initial_rules_file=master_ilp_rules_file)

    # Load checkpoint, including the potentially adapted max_astar_expansions
    start_depth_main_loop, all_accumulated_rules_loaded = load_main_checkpoint(main_checkpoint_file)
    all_accumulated_rules: Dict[int, List[str]] = {int(k): v for k, v in all_accumulated_rules_loaded.items()}


    if all_accumulated_rules:
        save_rules_to_file(master_ilp_rules_file, all_accumulated_rules)
        # heuristic_fn_wrapper.load_rules(master_ilp_rules_file)
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


        for ilp_refinement_iter in range(args.ilp_convergence_iterations):
            start_time_iterations = time.time()
            print(f"  [ILP Refine iter {ilp_refinement_iter + 1}/{args.ilp_convergence_iterations} for depth {current_depth_to_learn}]")

            start_time_a_star = time.time()

            new_pos_examples, new_neg_examples = all_dataset_for_example_generation(
                env, current_depth_to_learn, args.data_file)

            print("  Time to obtain positive and negative samples:  %.2f" % (time.time() - start_time_a_star))

            accumulated_pos_examples_this_depth.extend(new_pos_examples)
            accumulated_neg_examples_this_depth.extend(new_neg_examples)
            accumulated_pos_examples_this_depth = list(dict.fromkeys(accumulated_pos_examples_this_depth))
            accumulated_neg_examples_this_depth = list(dict.fromkeys(accumulated_neg_examples_this_depth))
            
            print(f"    Total unique examples for depth {current_depth_to_learn} after all dataset: Pos={len(accumulated_pos_examples_this_depth)}, Neg={len(accumulated_neg_examples_this_depth)}")

            if not accumulated_pos_examples_this_depth and ilp_refinement_iter == 0 : # No positive examples even after potential retry on first iteration
                 print(f"    [Warning] No positive examples generated for depth {current_depth_to_learn}. Popper may not find rules.")

            start_time_popper = time.time()

            current_rules_for_this_depth_iter = []
            if accumulated_pos_examples_this_depth or accumulated_neg_examples_this_depth: # Run Popper if any examples
                popper_output_dict = get_popper_examples_timeout(
                    accumulated_pos_examples_this_depth, accumulated_neg_examples_this_depth,
                    current_depth_to_learn, args.learned_program_dir,
                    current_popper_run_dir,
                    args.popper_runner_path,
                    logger_file,
                    args.time_out
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
            # heuristic_fn_wrapper.load_rules(master_ilp_rules_file)
        
        aggregate_all_solutions(popper_run_base_dir, os.path.join(popper_run_base_dir, 'all_popper_solutions.txt'))

        save_main_checkpoint(main_checkpoint_file, current_depth_to_learn, all_accumulated_rules)
        
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