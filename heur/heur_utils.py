# In heur/heur_utils.py

import pdb
from popper.util import Settings, order_prog, format_rule
from popper.loop import learn_solution
import os
from typing import Dict, List
from deepxube.environments.environment_abstract import State
import numpy as np
import pickle
import subprocess
import time

def print_prog_score(settings: Settings, prog, score: List[int]):
    tp, fn, tn, fp, size = score
    precision = 'n/a'
    if (tp + fp) > 0:
        precision = f'{tp / (tp + fp):0.2f}'
    recall = 'n/a'
    if (tp + fn) > 0:
        recall = f'{tp / (tp + fn):0.2f}'
    # print('*' * 10 + ' SOLUTION ' + '*' * 10)
    # if settings.noisy: # TODO If noisy return this value not implemented currently
    #     print(f'Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size} MDL:{size + fn + fp}')
    # else:
    #     print(f'Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size}')
    # print(self.format_prog(order_prog(prog)))
    clauses = []
    for rule in order_prog(prog):
        # print(format_rule(settings.order_rule(rule)))
        clauses.append(format_rule(settings.order_rule(rule)).strip('.'))

    # print('*' * 30)

    return clauses, precision, recall, tp, fn, tn, fp, size

def aggregate_all_solutions(results_dir: str, output_file: str):
    with open(output_file, "w") as fout:
        for folder in sorted(os.listdir(results_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
            depth_path = os.path.join(results_dir, folder)
            solution_file = os.path.join(depth_path, "solution.txt")
            if os.path.isdir(depth_path) and os.path.exists(solution_file):
                with open(solution_file, "r") as fin:
                    content = fin.read().strip()
                    if content:
                        fout.write(content + "\n\n")

def write_no_solution(file_path: str, depth: int):
    with open(file_path, "w") as f:
        f.write(f"********** DEPTH {depth} **********\n")
        f.write("NO SOLUTION\n")
        f.write("******************************\n")

def write_solution_file(file_path: str, depth: int, clauses: List[str], stats: Dict):
    with open(file_path, "w") as f:
        f.write(f"********** DEPTH {depth} **********\n")
        f.write(
            f"Precision:{stats['Precision']} Recall:{stats['Recall']} TP:{stats['TP']} FN:{stats['FN']} TN:{stats['TN']} FP:{stats['FP']} Size:{stats['Size']}\n"
        )
        for clause in clauses:
            f.write(f"{clause}\n")
        f.write("******************************\n")

def read_previous_solution(file_path: str) -> (List[str], Dict):
    clauses = []
    stats = None
    if not os.path.exists(file_path):
        return clauses, stats
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Precision"):
                try:
                    parts = line.strip().split()
                    stats = {
                        "Precision": float(parts[0].split(":")[-1]),
                        "Recall": float(parts[1].split(":")[-1]),
                        "TP": int(parts[2].split(":")[-1]),
                        "FN": int(parts[3].split(":")[-1]),
                        "TN": int(parts[4].split(":")[-1]),
                        "FP": int(parts[5].split(":")[-1]),
                        "Size": int(parts[6].split(":")[-1]),
                    }
                except Exception:
                    stats = None
            elif line.strip() and not line.startswith("*") and not line.startswith("NO SOLUTION"):
                clauses.append(line.strip())
    return clauses, stats

# def get_popper_examples(positive_states: List[State], negative_states: List[State], depth: int, task_folder: str, popper_bk_bias_dir: str):
#     bk_file: str = os.path.join(task_folder, "bk.pl")
#     bias_file: str = os.path.join(task_folder, "bias.pl")
#
#     depth_folder: str = os.path.join(popper_bk_bias_dir, str(depth))
#     if not os.path.exists(depth_folder):
#         os.makedirs(depth_folder)
#     with open(os.path.join(depth_folder, "exs.pl"), "w") as f:
#         for state in positive_states:
#             f.write("pos(f([" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "])).\n")
#         for state in negative_states:
#             f.write("neg(f([" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "])).\n")
#
#     bk_dest: str = os.path.join(depth_folder, "bk.pl")
#     bias_dest: str = os.path.join(depth_folder, "bias.pl")
#     with open(bk_file, "r") as src, open(bk_dest, "w") as dest:
#         dest.write(src.read())
#     with open(bias_file, "r") as src, open(bias_dest, "w") as dest:
#         dest.write(src.read())
#
#     prev_solution_file = os.path.join(depth_folder, "solution.txt")
#     prev_clauses, prev_stats = read_previous_solution(prev_solution_file)
#     # prev_clauses = []
#     # has_prev_solution = os.path.exists(prev_solution_file)
#     # if has_prev_solution:
#     #     with open(prev_solution_file, "r") as f:
#     #         lines = f.readlines()
#     #         for line in lines:
#     #             if line.strip() and not line.startswith("*") and not line.startswith(
#     #                     "Precision") and not line.startswith("NO SOLUTION"):
#     #                 prev_clauses.append(line.strip())
#
#     pdb.set_trace()
#
#     settings: Settings = Settings(kbpath=depth_folder, max_vars=6, max_body=8, timeout=1220) # TODO need to load bk dynamically
#     # settings.last_valid_solution = None  # <-- Track last good solution globally
#
#     try:
#         prog, score, stats = learn_solution(settings) # TODO need to change 1200 time limit
#         # if prog is None and settings.last_valid_solution:
#         #     prog, score = settings.last_valid_solution
#     except Exception as e:
#         print(f"Popper failed with error: {e}")
#         prog = None
#         # if prev_clauses:
#         #     return {depth: prev_clauses}
#         # if not has_prev_solution:
#         #     with open(prev_solution_file, "w") as f:
#         #         f.write(f"********** DEPTH {depth} **********\n")
#         #         f.write("NO SOLUTION\n")
#         #         f.write("******************************\n")
#         # return {depth: []}
#
#     pdb.set_trace()
#
#     if prog is None and settings.best_program_last:
#         # if prev_clauses:
#         #     return {depth: prev_clauses}
#         # if not has_prev_solution:
#         #     with open(prev_solution_file, "w") as f:
#         #         f.write(f"********** DEPTH {depth} **********\n")
#         #         f.write("NO SOLUTION\n")
#         #         f.write("******************************\n")
#         # return {depth: []}
#         prog, score = settings.best_program_last, settings.best_program_score_last
#         pdb.set_trace()
#
#     if prog is None:
#         if prev_clauses:
#             return {depth: prev_clauses}
#         write_no_solution(prev_solution_file, depth)
#         return {depth: []}
#
#     pdb.set_trace()
#
#     # Evaluate program
#     try:
#         clauses, precision, recall, tp, fn, tn, fp, size = print_prog_score(settings, prog, score)
#     except Exception as e:
#         print(f"Failed to parse program at depth {depth}: {e}")
#         if prev_clauses:
#             return {depth: prev_clauses}
#         write_no_solution(prev_solution_file, depth)
#         return {depth: []}
#
#     # Discard programs with false positives
#     if fp > 0:
#         print(f"Discarded due to FP={fp}")
#         if settings.best_program_last:
#             try:
#                 clauses, precision, recall, tp, fn, tn, fp, size = print_prog_score(settings, settings.best_program_last, settings.best_program_score_last)
#             except Exception as e:
#                 print(f"Failed to parse program at depth {depth}: {e}")
#                 if prev_clauses:
#                     return {depth: prev_clauses}
#                 write_no_solution(prev_solution_file, depth)
#                 return {depth: []}
#         # return {depth: prev_clauses if prev_clauses else []}
#
#     if fp>0:
#         print(f"Discarded due to FP={fp} on both depth")
#         if prev_clauses:
#             return {depth: prev_clauses}
#         write_no_solution(prev_solution_file, depth)
#         return {depth: []}
#
#     # Save new solution if it's better or first
#     stats = {
#         "Precision": precision,
#         "Recall": recall,
#         "TP": tp,
#         "FN": fn,
#         "TN": tn,
#         "FP": fp,
#         "Size": size,
#     }
#
#     pdb.set_trace()
#
#     if not prev_stats or tp > prev_stats.get("TP", 0):
#         write_solution_file(prev_solution_file, depth, clauses, stats)
#
#     return {depth: clauses}
#
#     # try:
#     #     clauses, precision, recall, tp, fn, tn, fp, size = print_prog_score(settings, prog, score)
#     #     if fp > 0:
#     #         print(f"Discarded program at depth {depth} due to FP={fp}")
#     #         if prev_clauses:
#     #             return {depth: prev_clauses}
#     #         if not has_prev_solution:
#     #             with open(prev_solution_file, "w") as f:
#     #                 f.write(f"********** DEPTH {depth} **********\n")
#     #                 f.write("NO SOLUTION\n")
#     #                 f.write("******************************\n")
#     #         return {depth: []}
#     # except Exception as e:
#     #     print(f"Error parsing program at depth {depth}: {e}")
#     #     if prev_clauses:
#     #         return {depth: prev_clauses}
#     #     if not has_prev_solution:
#     #         with open(prev_solution_file, "w") as f:
#     #             f.write(f"********** DEPTH {depth} **********\n")
#     #             f.write("NO SOLUTION\n")
#     #             f.write("******************************\n")
#     #     return {depth: []}
#
#     pdb.set_trace()
#
#     # clauses_dict: Dict = {depth: []}
#     #
#     # with open(prev_solution_file, "w") as f:
#     #     f.write(f"********** DEPTH {depth} **********\n")
#     #     f.write(f"Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size}\n")
#     #     for clause in clauses:
#     #         f.write(f"{clause}\n")
#     #         clauses_dict[depth].append(clause)
#     #     f.write("******************************\n")
#
#     # with open(os.path.join(depth_folder, "solution.txt"), "w") as f:
#     #     f.write(f"********** DEPTH {depth} **********\n")
#     #     if prog is not None:
#     #         # settings.print_prog_score(prog, score)
#     #         clauses, precision, recall, tp, fn, tn, fp, size = print_prog_score(settings, prog, score)
#     #         f.write(f"Precision:{precision} Recall:{recall} TP:{tp} FN:{fn} TN:{tn} FP:{fp} Size:{size}\n")
#     #         for clause in clauses:
#     #             f.write(f"{clause}\n")
#     #             clauses_dict[depth].append(clause)
#     #         # pdb.set_trace()
#     #     else:
#     #         f.write("NO SOLUTION\n")
#     #     f.write("******************************\n")
#
#     return clauses_dict

def get_popper_examples_timeout(positive_states: List[State], negative_states: List[State],
                        depth: int,
                        task_folder: str,
                        # Base directory for original bk.pl, bias.pl (used only if prepare_popper_input_files failed)
                        popper_depth_specific_run_dir: str,  # THIS IS THE KEY: e.g., model_dir/popper_runs/1
                        popper_runner_path: str, logger_file: str, time_out: int):
    """
    Prepares example file and runs Popper using the files in popper_depth_specific_run_dir.
    BK and Bias files should have already been prepared in popper_depth_specific_run_dir
    by the prepare_popper_input_files function.
    """

    # depth_folder is the specific directory for this Popper run.
    depth_folder: str = popper_depth_specific_run_dir

    # Ensure this directory exists (prepare_popper_input_files should have created it)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
        print(f"Warning in get_popper_examples: Popper run directory {depth_folder} did not exist and was created.")
        print(
            f"         This is unexpected if prepare_popper_input_files was supposed to create it with BK/Bias files.")
        # As a fallback, if it didn't exist, we might copy base BK/Bias, but this complicates the logic.
        # For now, we assume prepare_popper_input_files handles BK/Bias creation there.

    # Write exs.pl to depth_folder/exs.pl
    exs_file_path = os.path.join(depth_folder, "exs.pl")
    with open(exs_file_path, "w") as f:
        for state in positive_states:
            # Assuming state.tiles is available and is a flat list/array of numbers for NPuzzleState
            # For other state types, this state_to_prolog_list conversion needs to be general
            # or specific to the NPState object if type checking is done.
            if hasattr(state, 'tiles'):
                prolog_list_str = "[" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "]"
                f.write(f"pos(f({prolog_list_str})).\n")
            else:
                print(
                    f"Warning: State object of type {type(state)} does not have 'tiles' attribute. Cannot create example.")
        for state in negative_states:
            if hasattr(state, 'tiles'):
                prolog_list_str = "[" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "]"
                f.write(f"neg(f({prolog_list_str})).\n")
            else:
                print(
                    f"Warning: State object of type {type(state)} does not have 'tiles' attribute. Cannot create example.")

    print(f"    [Popper Call] Examples written to {exs_file_path}")
    print(f"    [Popper Call] Using BK/Bias from: {depth_folder}")

    # Popper runner is called with `depth_folder`. Popper will look for exs.pl, bk.pl, bias.pl there.
    print(f"    [Popper Call] Running Popper for depth {depth} using config in folder: {depth_folder}")
    try:
        # Ensure command arguments are all strings
        cmd = ["python", str(popper_runner_path), str(depth), str(depth_folder), str(time_out)]
        # print(f"    [Popper Call] Executing: {' '.join(cmd)}") # For debugging the command

        with open(logger_file, "a") as log_file:  # Open in append mode
            print(f"    [Popper Call] Logging output for depth {depth} to {logger_file}")
            subprocess.run(
                cmd,
                check=True,
                stdout=log_file,
                stderr=log_file,
                text=True
            )

        # subprocess.run(cmd, check=True, capture_output=True, text=True) # Capture output for better error display

    except subprocess.CalledProcessError as e:
        # print(f"!!! Popper execution failed for depth {depth} in {depth_folder} !!!")
        # print(f"    Command: {e.cmd}")
        # print(f"    Return code: {e.returncode}")
        # print(f"    Stdout: {e.stdout}")
        # print(f"    Stderr: {e.stderr}")

        with open(logger_file, "a") as log_file:
            log_file.write(f"\n!!! Popper failed for depth {depth} in {depth_folder}!!!\n")
            log_file.write(f"    Command: {' '.join(e.cmd)}\n")
            log_file.write(f"    Return code: {e.returncode}\n")
            log_file.write(f"    Stdout:\n{e.stdout}\n")
            log_file.write(f"    Stderr:\n{e.stderr}\n")

        return {depth: []}

    # Load results from clauses_temp.pkl
    clauses_file = os.path.join(depth_folder, "clauses_temp.pkl")
    clauses_dict = {depth: []}

    wait_time = 0
    max_wait = 2.0  # Increased max_wait slightly
    while not os.path.exists(clauses_file) and wait_time < max_wait:
        time.sleep(0.1)
        wait_time += 0.1

    if os.path.exists(clauses_file):
        try:
            with open(clauses_file, "rb") as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict) and depth in loaded_data:
                    clauses_dict = loaded_data
                else:
                    print(
                        f"Warning: clauses_temp.pkl for depth {depth} has unexpected format or missing key. Content: {loaded_data}")
                    clauses_dict = {depth: []}
        except EOFError:
            print(f"Warning: EOFError reading clauses_temp.pkl for depth {depth}. File might be empty or corrupted.")
            clauses_dict = {depth: []}
        except Exception as e:
            print(f"Warning: Error reading clauses_temp.pkl for depth {depth}: {e}")
            clauses_dict = {depth: []}

        try:
            os.remove(clauses_file)
        except OSError as e:
            print(f"Warning: Could not remove temporary clauses file {clauses_file}: {e}")
    else:
        print(f"Warning: clauses_temp.pkl not found for depth {depth} in {depth_folder} after Popper run.")

    return clauses_dict

def get_popper_examples(positive_states: List[State], negative_states: List[State],
                        depth: int,
                        task_folder: str, # Base directory for original bk.pl, bias.pl (used only if prepare_popper_input_files failed)
                        popper_depth_specific_run_dir: str, # THIS IS THE KEY: e.g., model_dir/popper_runs/1
                        popper_runner_path: str, logger_file: str):
    """
    Prepares example file and runs Popper using the files in popper_depth_specific_run_dir.
    BK and Bias files should have already been prepared in popper_depth_specific_run_dir
    by the prepare_popper_input_files function.
    """

    # depth_folder is the specific directory for this Popper run.
    depth_folder: str = popper_depth_specific_run_dir

    # Ensure this directory exists (prepare_popper_input_files should have created it)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
        print(f"Warning in get_popper_examples: Popper run directory {depth_folder} did not exist and was created.")
        print(f"         This is unexpected if prepare_popper_input_files was supposed to create it with BK/Bias files.")
        # As a fallback, if it didn't exist, we might copy base BK/Bias, but this complicates the logic.
        # For now, we assume prepare_popper_input_files handles BK/Bias creation there.

    # Write exs.pl to depth_folder/exs.pl
    exs_file_path = os.path.join(depth_folder, "exs.pl")
    with open(exs_file_path, "w") as f:
        for state in positive_states:
            # Assuming state.tiles is available and is a flat list/array of numbers for NPuzzleState
            # For other state types, this state_to_prolog_list conversion needs to be general
            # or specific to the NPState object if type checking is done.
            if hasattr(state, 'tiles'):
                prolog_list_str = "[" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "]"
                f.write(f"pos(f({prolog_list_str})).\n")
            else:
                print(f"Warning: State object of type {type(state)} does not have 'tiles' attribute. Cannot create example.")
        for state in negative_states:
            if hasattr(state, 'tiles'):
                prolog_list_str = "[" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "]"
                f.write(f"neg(f({prolog_list_str})).\n")
            else:
                print(f"Warning: State object of type {type(state)} does not have 'tiles' attribute. Cannot create example.")

    print(f"    [Popper Call] Examples written to {exs_file_path}")
    print(f"    [Popper Call] Using BK/Bias from: {depth_folder}")


    # Popper runner is called with `depth_folder`. Popper will look for exs.pl, bk.pl, bias.pl there.
    print(f"    [Popper Call] Running Popper for depth {depth} using config in folder: {depth_folder}")
    try:
        # Ensure command arguments are all strings
        cmd = ["python", str(popper_runner_path), str(depth), str(depth_folder)]
        # print(f"    [Popper Call] Executing: {' '.join(cmd)}") # For debugging the command

        with open(logger_file, "a") as log_file:  # Open in append mode
            print(f"    [Popper Call] Logging output for depth {depth} to {logger_file}")
            subprocess.run(
                cmd,
                check=True,
                stdout=log_file,
                stderr=log_file,
                text=True
            )

        # subprocess.run(cmd, check=True, capture_output=True, text=True) # Capture output for better error display

    except subprocess.CalledProcessError as e:
        # print(f"!!! Popper execution failed for depth {depth} in {depth_folder} !!!")
        # print(f"    Command: {e.cmd}")
        # print(f"    Return code: {e.returncode}")
        # print(f"    Stdout: {e.stdout}")
        # print(f"    Stderr: {e.stderr}")

        with open(logger_file, "a") as log_file:
            log_file.write(f"\n!!! Popper failed for depth {depth} in {depth_folder}!!!\n")
            log_file.write(f"    Command: {' '.join(e.cmd)}\n")
            log_file.write(f"    Return code: {e.returncode}\n")
            log_file.write(f"    Stdout:\n{e.stdout}\n")
            log_file.write(f"    Stderr:\n{e.stderr}\n")

        return {depth: []}

    # Load results from clauses_temp.pkl
    clauses_file = os.path.join(depth_folder, "clauses_temp.pkl")
    clauses_dict = {depth: []} 

    wait_time = 0
    max_wait = 2.0 # Increased max_wait slightly
    while not os.path.exists(clauses_file) and wait_time < max_wait:
        time.sleep(0.1)
        wait_time += 0.1
    
    if os.path.exists(clauses_file):
        try:
            with open(clauses_file, "rb") as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict) and depth in loaded_data:
                    clauses_dict = loaded_data
                else:
                    print(f"Warning: clauses_temp.pkl for depth {depth} has unexpected format or missing key. Content: {loaded_data}")
                    clauses_dict = {depth: []} 
        except EOFError:
            print(f"Warning: EOFError reading clauses_temp.pkl for depth {depth}. File might be empty or corrupted.")
            clauses_dict = {depth: []} 
        except Exception as e:
            print(f"Warning: Error reading clauses_temp.pkl for depth {depth}: {e}")
            clauses_dict = {depth: []} 
        
        try:
            os.remove(clauses_file)
        except OSError as e:
            print(f"Warning: Could not remove temporary clauses file {clauses_file}: {e}")
    else:
        print(f"Warning: clauses_temp.pkl not found for depth {depth} in {depth_folder} after Popper run.")

    return clauses_dict