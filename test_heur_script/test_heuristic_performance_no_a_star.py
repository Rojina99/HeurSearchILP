# python test_heuristic_performance_no_a_star.py --env puzzle8 --model_dir puzzle8_output_depth_31_m/env_puzzle8_pdir_puzzle8_rules_source_depth_31_ip_states_each_depth_50_ss_60_a_star_bs_1_ci_1_a_star_exp_start_1000_workers_4_rm_5.0_ri_5000/ --num_problems 480 --max_depths 31   --output_file results_8puzzle_heuristic_test_no_astar.csv --test_dir saved_test_data --data_file data/puzzle8.pkl --trained_heuristic True
"""File to get initial heuristic value as shown in R^2 and MSE results in paper"""

import argparse
import os
import pdb
import pickle
import random
import time
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import csv
import json

from environments.env_utils import get_environment
from deepxube.environments.environment_abstract import Environment, State, Action, Goal, EnvGrndAtoms
from deepxube.utils.data_utils import Logger
import sys
from search.astar import AStar, get_path, Node # Assuming astar.py is in the search directory
from heur.test_prolog_heur import HeurILP # Assuming this has the binary search for depth
from nnet.nnet_utils import load_and_process_clause_to_get_unique_clause # For processing rules
from deepxube.utils import misc_utils

# --- PureILPHeuristic Class (Copied from final_parallel.py for self-containment, ensure it matches your version) ---
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
                processed_rules = load_and_process_clause_to_get_unique_clause(rules_file)
                if processed_rules:
                    for depth_float, clauses in processed_rules.items():
                        if clauses:
                            for clause in clauses:
                                self.heur_ilp.add(depth_float, clause)
                    print(f"    PureILPHeuristic: Loaded and processed rules from {rules_file}")
                else:
                    print(f"    PureILPHeuristic: No rules found in {rules_file} after processing.")
            except Exception as e:
                print(f"    PureILPHeuristic: Error loading or processing rules from {rules_file}: {e}")
        else:
            print(f"    PureILPHeuristic: Rules file not found or empty ({rules_file}). Heuristic may be trivial.")

    def get_heur(self, states: List[State], goals: List[Goal] = None) -> np.ndarray:
        ilp_values_optional: List[Optional[float]] = self.heur_ilp.get_heur(states)
        processed_heuristics = [val if val is not None else 0.0 for val in ilp_values_optional]
        return np.array(processed_heuristics, dtype=np.float64)


class LoadTestData:
    def __init__(self, env: Environment, max_depths: int, num_problems: int, data_file: str):
        print("Generating data")

        # pdb.set_trace()
        start_state_time = time.time()

        self.states_start: List[State] = []
        self.states_goals: List[State] = []
        self.depths_list: List[int] = []

        # min_depth = 1 if max_depths > 0 else 0

        min_depth = 0

        cost_to_go_list = list(range(min_depth, max_depths+1))

        num_states_per_step: List[int] = misc_utils.split_evenly(num_problems, len(cost_to_go_list))

        ctg_to_states: Dict[int, Set[State]] = pickle.load(open(data_file, "rb"))

        # states_all: List[State] = []
        # for states_i in ctg_to_states.values():
        #     states_all.extend(list(states_i))

        states_all: List[State] = []
        for states_i in ctg_to_states:
            states_all.extend([len(list(ctg_to_states[states_i]))])

        # goal_state = list(ctg_to_states[0])

        for cost_to_go, num_states_i in zip(cost_to_go_list, num_states_per_step):
            if num_states_i > 0:
                steps_i: List[int] = [cost_to_go] * num_states_i
                # start_states, goal_states_list = env.get_start_goal_pairs(scramble_depths)
                # start_states_list, goal_states_list = env.get_start_goal_pairs(steps_i)
                available_states = list(ctg_to_states[cost_to_go])

                if len(available_states) <= num_states_i:
                    start_states_list = available_states
                else:
                    start_states_list = random.sample(available_states, num_states_i)

                number_of_states_per_depth = len(start_states_list)
                goal_states_list = env.get_goal_states(number_of_states_per_depth)
                self.states_start.extend(start_states_list)
                self.states_goals.extend(goal_states_list)
                self.depths_list.extend(steps_i[:len(start_states_list)])

        print(f"Time to generate test samples {time.time()-start_state_time}")

def load_data(test_dir: str, env, max_depths, num_problems, data_file):
    test_file: str = "%s/test_dataset_file.pkl" % test_dir

    if os.path.isfile(test_file):
        print("Loading data from file")
        generated_test_data: LoadTestData = pickle.load(open("%s/test_dataset_file.pkl" % test_dir, "rb"))
        # print(f"Loaded with itr: {status.itr}, update_num: {status.update_num}, "
        #       f"per_solved_best: {status.per_solved_best}")
    else:
        generated_test_data: LoadTestData = LoadTestData(env, max_depths, num_problems, data_file)
        pickle.dump(generated_test_data, open(test_file, "wb"), protocol=-1)

    return generated_test_data.states_start, generated_test_data.states_goals, generated_test_data.depths_list

def format_path_for_csv(path_states: List[State], path_actions: List[Action]) -> str:
    """Formats the solution path into a human-readable string for CSV."""
    if not path_states:
        return "N/A"
    
    path_str_parts = []
    # Assuming states can be represented as strings (e.g., tile configuration)
    # and actions too. You might need to adapt this based on your State/Action types.
    try:
        path_str_parts.append(f"S0: {str(path_states[0].tiles)}") # Example for NPState
        for i, action in enumerate(path_actions):
            # Assuming Action has a simple representation.
            # If Action is an object, use str(action) or action.to_string()
            action_str = str(action.action) if hasattr(action, 'action') else str(action)
            path_str_parts.append(f" -> A{i}: {action_str} -> S{i+1}: {str(path_states[i+1].tiles)}")
    except AttributeError: # Fallback if .tiles or .action not present
         path_str_parts = [str(s) for s in path_states]


    return " | ".join(path_str_parts)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def solve_problem_with_timeout(
    start_state: State,
    goal_state: Goal,
    heuristic_fn: PureILPHeuristic,
) -> Tuple[Optional[float], int, float, bool, int, Optional[str], Optional[float]]:
    """
    Solves a single problem instance using A* with a time limit.

    Returns:
        Tuple: (path_cost, num_expansions, time_taken, solved_within_time, 
                final_open_set_size, solution_path_str, initial_heuristic_value)
               path_cost and solution_path_str are None if not solved.
    """
    # Get initial heuristic value for the start state
    initial_h_value_array = heuristic_fn.get_heur([start_state], [goal_state])
    initial_h_value = initial_h_value_array[0] if initial_h_value_array.size > 0 else 0.0
    print(f"\n    Initial heuristic value for start state: {initial_h_value}")

    # astar = AStar(env)
    # The AStar class internally calls heuristic_fn.get_heur for the root node as well
    # when add_instances is called, to set its initial cost.
    # astar.add_instances([start_state], [goal_state], [astar_weight], heuristic_fn)

    start_time = time.time()
    num_expansions_total = 0 # This will be instance.step_num
    # instance = astar.instances[0]

    # while not instance.finished:
    #     current_time = time.time()
    #     if current_time - start_time > time_limit_seconds:
    #         print(f"    Problem timed out after {time_limit_seconds:.2f} seconds.")
    #         return None, instance.step_num, time_limit_seconds, False, instance.num_nodes_generated, len(instance.open_set), None, initial_h_value

        # if max_expansions is not None and instance.step_num >= max_expansions:
        #     print(f"    Problem reached max_iterations limit of {max_expansions}.")
        #     instance.finished = True # Ensure it stops if not already marked
        #     break

        # astar.step(heuristic_fn, batch_size=astar_heuristic_batch_size, verbose=False)
        # num_expansions_total = instance.step_num # step_num is updated inside AStar.step

        # If AStar.step itself can mark instance.finished due to internal limits or solution
        # if instance.finished:
        #     break

    end_time = time.time()
    time_taken = end_time - start_time
    # final_open_set_size = len(instance.open_set)
    # num_expansions_final = instance.step_num # Get final expansion count

    # if instance.goal_node:
    #     path_states, path_actions, path_cost = get_path(instance.goal_node)
    #     solution_path_str = format_path_for_csv(path_states, path_actions)
    #     return path_cost, num_expansions_final, time_taken, True, instance.num_nodes_generated, final_open_set_size, solution_path_str, initial_h_value
    # else:
    #     return None, num_expansions_final, time_taken, False, instance.num_nodes_generated, final_open_set_size, None, initial_h_value

    return time_taken, initial_h_value


def main():
    parser = argparse.ArgumentParser(description="Test ILP Heuristic Performance.")
    parser.add_argument('--env', type=str, default="puzzle8", help="Environment name (e.g., puzzle8)")
    # parser.add_argument('--rules_file', type=str, required=True, help="Path to the master_learned_ilp_rules.pkl file.")
    # parser.add_argument('--popper_run_base_dir', type=str, required=True, help="Base directory where Popper runs were stored (e.g., model_dir/popper_runs).")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory with learned rules i.e. master_learned_ilp_rules.pkl and Base directory where popper runs were stored (e.g., model_dir/popper_runs).")
    parser.add_argument('--num_problems', type=int, default=480, help="Number of random problems to test.")
    parser.add_argument('--max_depths', type=int, default=31, help="Maximum steps for test problems.")
    parser.add_argument('--output_file', type=str, default="heuristic_test_results.csv", help="CSV file to save results.")
    parser.add_argument('--test_dir', type=str, default="saved_test_data", help="Directory to save test dataset to do A* search.")
    parser.add_argument('--data_file', type=str, default='data/puzzle8.pkl', help="Path to 8 puzzle data")
    parser.add_argument('--trained_heuristic', type=str2bool, default=True, help="Mode to test for trained or no heuristic")

    start_program_time = time.time()

    args = parser.parse_args()
    args.rules_file = os.path.join(args.model_dir, 'master_learned_ilp_rules.pkl') # Path to the `master_learned_ilp_rules.pkl` generated by `final_parallel.py`
    args.popper_run_base_dir = os.path.join(args.model_dir, 'popper_runs') # This is crucial because the heuristic needs to load the correct `bk.pl` context (which includes rules from lower depths) when checking rules for a specific depth via its binary search.
    output_file_path = os.path.join(args.model_dir, args.output_file)
    # stats_file_path = os.path.join(args.model_dir, os.path.splitext(args.output_file) + '.json')

    log_file_path = os.path.join(args.model_dir, os.path.splitext(args.output_file)[0] + '_test_log'+'.txt')

    if args.trained_heuristic:
        if not os.path.exists(args.rules_file) or not os.path.exists(args.popper_run_base_dir):
            raise Exception("Training file path are incorrect")
    else:
        if os.path.exists(args.rules_file) or os.path.exists(args.popper_run_base_dir):
            raise Exception("Please pass arbitrary file path not trained model")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    sys.stdout = Logger(log_file_path, "a")

    env, _ = get_environment(args.env)
    if not isinstance(env, EnvGrndAtoms):
        raise TypeError("Environment must be an instance of EnvGrndAtoms.")

    print(f"Loading heuristic with rules from: {args.rules_file}")
    print(f"Heuristic will look for depth-specific BK files in subdirectories of: {args.popper_run_base_dir}")

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    
    heuristic = PureILPHeuristic(env, args.popper_run_base_dir, args.rules_file)

    print(f"\nGenerating {args.num_problems} test problems (max scramble: {args.max_depths})...")

    start_states, goal_states_list, depths_list = load_data(args.test_dir, env, args.max_depths, args.num_problems, args.data_file)
    args.num_problems = len(start_states)
    print(f"\nUpdated {args.num_problems} test problems (max scramble: {args.max_depths})...")

    unique_lengths, counts = np.unique(depths_list, return_counts=True)

    print(f"Number of start states {len(start_states)} \nNumber of goal states {len(goal_states_list)} \nNumber of cost to go {len(depths_list)} \nDifferent Depth{unique_lengths} \n Corresponding Number of Values {counts}")

    # Assuming a single, fixed goal for these types of problems usually.
    # If get_start_goal_pairs returns specific goals per start_state, use goal_states_list[i].
    fixed_goal = goal_states_list[0] # Static goal with static dataset

    # Otherwise, get a generic goal.

    results = []
    total_time_solved = 0.0
    total_initial_h_solved = 0.0 # For averaging heuristic value of solved problems
    total_ground_truth_cost_solved = 0.0

    completed_ids = []
    start_index = 0

    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    completed_ids.append(int(row["problem_id"]))

                    # Rebuild stats from row if solved
                    total_time_solved += float(row["time_taken_s"])
                    total_ground_truth_cost_solved += float(row["ground_truth_cost"])
                    ih = row["initial_heuristic"]
                    if ih is not None and ih != "":
                        total_initial_h_solved += float(ih)
        except Exception:
            raise

        print(f"Loaded {len(completed_ids)} problems from CSV.")

        start_index = len(completed_ids)

        problem_ids = [int(row["problem_id"]) for row in results]

        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("Duplicate problem_id entries found in CSV!")

        print(f"\n--- Load from CSV Summary:---")
        print(f"Problems Attempted: {start_index + 1 / args.num_problems}")

    # Ensure all keys that might be added are in the initial keys list for robust header writing
    all_keys = ["problem_id", "ground_truth_cost", "initial_heuristic", "time_taken_s"]

    print(f"\nRunning {args.num_problems-start_index} problems out of initial {args.num_problems} problems...")

    for i in range(start_index, args.num_problems):
        if i + 1 in completed_ids:
            continue

        print(f"Solving problem {i+1}/{args.num_problems} (scramble depth: {depths_list[i]})")
        start_state = start_states[i]
        # Use the fixed_goal or goal_states_list[i] as appropriate
        current_goal = goal_states_list[i] if goal_states_list and i < len(goal_states_list) else fixed_goal

        time_taken, initial_h = solve_problem_with_timeout(start_state, current_goal, heuristic)

        results.append({"problem_id": i + 1, "ground_truth_cost": depths_list[i], "initial_heuristic": initial_h, "time_taken_s": time_taken})

        if results:
            with open(output_file_path, 'w', newline='') as output_csv_file:
                dict_writer = csv.DictWriter(output_csv_file, fieldnames=all_keys)
                dict_writer.writeheader()
                dict_writer.writerows(results)
            print(f"\nResults saved to {output_file_path} till state index {start_index + 1}")

        total_time_solved += time_taken
        if initial_h is not None: # initial_h should always be float now
             total_initial_h_solved += initial_h
        total_ground_truth_cost_solved += float(depths_list[i])
        print(f"Initial h={initial_h}, Ground Truth Cost={depths_list[i]}, Time={time_taken:.2f}s")

        print(f"\n--- Test Summary:---")
        print(f"Problems Attempted: {i+1}/{args.num_problems}")

    print(f"Program End Time", time.time() - start_program_time)


    # Save results to CSV
    # if results:
    #     # Ensure all keys that might be added are in the initial keys list for robust header writing
    #     all_keys = ["problem_id", "ground_truth_cost", "initial_heuristic", "solved", "cost",
    #                 "iterations", "time_taken_s", "number_of_nodes_generated", "open_set_size_at_end",
    #                 "astar_weight_used", "time_limit_s", "solution_path"]
    #     # Dynamically get all keys from the first result if needed, but predefined is safer
    #     # if results: keys = results[0].keys()
    #
    #     with open(output_file_path, 'w', newline='') as output_csv_file:
    #         dict_writer = csv.DictWriter(output_csv_file, fieldnames=all_keys)
    #         dict_writer.writeheader()
    #         dict_writer.writerows(results)
    #     print(f"\nResults saved to {output_file_path}")

if __name__ == "__main__":
    main()