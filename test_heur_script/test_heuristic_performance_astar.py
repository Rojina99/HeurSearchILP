import argparse
import os
import pickle
import time
import numpy as np
import csv
from typing import List, Dict, Tuple, Optional

from environments.env_utils import get_environment
# Add this alongside your other imports
from environments.n_puzzle import NPState
from deepxube.environments.environment_abstract import Environment, State, Action, Goal, EnvGrndAtoms
from search.astar import AStar, get_path, Node 
from heur.test_prolog_heur import HeurILP
from nnet.nnet_utils import load_and_process_clause_to_get_unique_clause
from deepxube.utils import misc_utils

# --- PureILPHeuristic Class (Self-contained) ---
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

def format_path_for_csv(path_states: List[State], path_actions: List[Action]) -> str:
    """Formats the solution path into a human-readable string for CSV."""
    if not path_states:
        return "N/A"
    
    path_str_parts = []
    try:
        # Special handling for states with a 'tiles' attribute (like N-Puzzle)
        if hasattr(path_states[0], 'tiles'):
            path_str_parts.append(f"S0: {str(path_states[0].tiles)}")
            for i, action in enumerate(path_actions):
                action_str = str(action.action) if hasattr(action, 'action') else str(action)
                path_str_parts.append(f" -> A{i}: {action_str} -> S{i+1}: {str(path_states[i+1].tiles)}")
        else: # Generic fallback
            path_str_parts = [str(s) for s in path_states]

    except Exception as e:
         print(f"Warning: Could not format path in detail, falling back to simple string representation. Error: {e}")
         path_str_parts = [str(s) for s in path_states]

    return " | ".join(path_str_parts)

def solve_problem_with_timeout(
    env: Environment,
    start_state: State,
    goal_state: Goal,
    heuristic_fn: PureILPHeuristic,
    astar_weight: float,
    time_limit_seconds: float,
    max_expansions: Optional[int] = None
) -> Dict:
    """
    Solves a single problem instance using A* with a time limit.

    Returns:
        A dictionary containing all relevant solve statistics.
    """
    # Get initial heuristic value for the start state
    initial_h_value_array = heuristic_fn.get_heur([start_state], [goal_state])
    initial_h_value = initial_h_value_array[0] if initial_h_value_array.size > 0 else 0.0
    print(f"    Initial heuristic value for start state: {initial_h_value}")

    astar = AStar(env)
    astar.add_instances([start_state], [goal_state], [astar_weight], heuristic_fn)

    start_time = time.time()
    instance = astar.instances[0]

    # --- A* Search Loop ---
    # The batch size is now fixed to 1 as requested.
    A_STAR_HEURISTIC_BATCH_SIZE = 1

    while not instance.finished:
        current_time = time.time()
        if current_time - start_time > time_limit_seconds:
            print(f"    Problem timed out after {time_limit_seconds:.2f} seconds.")
            break

        if max_expansions is not None and instance.step_num >= max_expansions:
            print(f"    Problem reached max_expansions limit of {max_expansions}.")
            break

        astar.step(heuristic_fn, batch_size=A_STAR_HEURISTIC_BATCH_SIZE, verbose=False)

    end_time = time.time()
    time_taken = end_time - start_time
    
    # --- Prepare Results ---
    result_details = {
        "initial_heuristic": initial_h_value,
        "expansions": instance.step_num,
        "time_taken_s": time_taken,
        "open_set_size_at_end": len(instance.open_set),
        "solved": False,
        "cost": -1,
        "solution_path": "N/A",
        "goal_g_value": -1,
        "goal_h_value": -1,
        "goal_f_value": -1,
    }

    if instance.goal_node:
        path_states, path_actions, path_cost = get_path(instance.goal_node)
        solution_path_str = format_path_for_csv(path_states, path_actions)
        
        goal_node = instance.goal_node
        # FIX: Use the path_cost variable for the g-value.
        goal_g = path_cost
        
        # FIX: The heuristic (h-value) for a goal node is always 0.
        goal_h = 0.0
        
        result_details.update({
            "solved": True,
            "cost": path_cost,
            "solution_path": solution_path_str,
            "goal_g_value": goal_g,
            "goal_h_value": goal_h,
            "goal_f_value": goal_g + (astar_weight * goal_h)
        })

    return result_details


def main():
    parser = argparse.ArgumentParser(description="Test ILP Heuristic Performance with A* on a static dataset.")
    parser.add_argument('--env', type=str, default="puzzle8", help="Environment name (e.g., puzzle8).")
    parser.add_argument('--rules_file', type=str, required=True, help="Path to the master_learned_ilp_rules.pkl file.")
    parser.add_argument('--popper_run_base_dir', type=str, required=True, help="Base directory where Popper runs were stored (e.g., model_dir/popper_runs).")
    parser.add_argument('--test_data_file', type=str, required=True, help="Path to the pickle file containing a list of test states.")
    parser.add_argument('--astar_weight', type=float, default=0.8, help="Weight W for A* (f = g + W*h).")
    parser.add_argument('--time_limit_seconds', type=float, default=600.0, help="Time limit per A* problem in seconds.")
    parser.add_argument('--max_expansions_per_problem', type=int, default=100000, help="Optional overall A* expansion limit per problem.")
    parser.add_argument('--output_file', type=str, default="heuristic_test_results_static.csv", help="CSV file to save results.")

    args = parser.parse_args()

    env, _ = get_environment(args.env)
    if not isinstance(env, EnvGrndAtoms):
        raise TypeError("Environment must be an instance of EnvGrndAtoms.")

    print(f"Loading heuristic with rules from: {args.rules_file}")
    heuristic = PureILPHeuristic(env, args.popper_run_base_dir, args.rules_file)

    print(f"\nLoading static test data from: {args.test_data_file}")
    try:
        with open(args.test_data_file, 'rb') as f:
            raw_states_from_pickle = pickle.load(f)
        if not isinstance(raw_states_from_pickle, list):
            raise TypeError("Test data file must contain a list of states.")
        print(f"Loaded {len(raw_states_from_pickle)} test problems.")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {args.test_data_file}")
        return
    except (pickle.UnpicklingError, TypeError) as e:
        print(f"Error loading or parsing pickle file: {e}")
        return
    
    # --- FIX: Convert raw lists to proper environment State objects ---
    print("Converting raw state lists to environment State objects...")
    start_states = []
    for raw_list in raw_states_from_pickle:
        # The environment's State class constructor likely expects a NumPy array of a specific type.
        # We create the array and then pass it to the environment's State constructor.
        tiles_np = np.array(raw_list, dtype=env.dtype)
        start_states.append(NPState(tiles=tiles_np))
    print("Conversion complete.")
    # --- END FIX ---

    # A single, fixed goal is assumed for these problem types.
    fixed_goal = env.get_goal_states(1)[0]
    num_problems = len(start_states)

    results = []
    solved_count = 0
    total_time_solved = 0
    total_expansions_solved = 0
    total_initial_h_solved = 0

    print(f"\nRunning A* on {num_problems} problems with a time limit of {args.time_limit_seconds}s each.")
    print(f"A* Weight: {args.astar_weight}, A* Heuristic Batch Size: 1 (Fixed)")
    print("-" * 50)


    for i, start_state in enumerate(start_states):
        print(f"Solving problem {i+1}/{num_problems}...")
        
        solve_stats = solve_problem_with_timeout(
            env, start_state, fixed_goal, heuristic,
            args.astar_weight, args.time_limit_seconds, args.max_expansions_per_problem
        )
        
        # Combine args and results for a complete log record
        full_record = {
            "problem_id": i + 1,
            "start_state_str": str(start_state.tiles) if hasattr(start_state, 'tiles') else str(start_state),
            **solve_stats, # Unpack all stats from the solver function
            "astar_weight_used": args.astar_weight,
            "time_limit_s": args.time_limit_seconds,
            "max_expansions_limit": args.max_expansions_per_problem if args.max_expansions_per_problem else "None"
        }
        results.append(full_record)
        
        if full_record["solved"]:
            solved_count += 1
            total_time_solved += full_record["time_taken_s"]
            total_expansions_solved += full_record["expansions"]
            total_initial_h_solved += full_record["initial_heuristic"]
            print(f"  Problem {i+1}: SOLVED! Cost={full_record['cost']}, Expansions={full_record['expansions']}, Time={full_record['time_taken_s']:.2f}s")
        else:
            print(f"  Problem {i+1}: NOT SOLVED (Timeout or Max Expansions). Expansions={full_record['expansions']}, Time={full_record['time_taken_s']:.2f}s")
        
    print("\n" + "=" * 20 + " Test Summary " + "=" * 20)
    print(f"Problems Attempted: {num_problems}")
    print(f"Problems Solved: {solved_count} ({ (solved_count/num_problems)*100 if num_problems > 0 else 0 :.2f}%)")
    if solved_count > 0:
        print(f"Average Time (solved): {total_time_solved / solved_count:.2f}s")
        print(f"Average Expansions (solved): {total_expansions_solved / solved_count:.2f}")
        print(f"Average Initial Heuristic (solved): {total_initial_h_solved / solved_count:.2f}")
    print("=" * 54)


    # Save results to CSV
    if results:
        # Define all possible headers for the CSV file
        fieldnames = [
            "problem_id", "solved", "cost", "expansions", "time_taken_s", 
            "initial_heuristic", "goal_g_value", "goal_h_value", "goal_f_value",
            "open_set_size_at_end", "astar_weight_used", "time_limit_s", 
            "max_expansions_limit", "start_state_str", "solution_path"
        ]
        
        print(f"\nSaving detailed results to {args.output_file}...")
        with open(args.output_file, 'w', newline='', encoding='utf-8') as output_csv_file:
            dict_writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print("Save complete.")

if __name__ == "__main__":
    main()