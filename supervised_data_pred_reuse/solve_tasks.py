import os
import re
import argparse
import pickle
import subprocess
from multiprocessing import Pool
from typing import Dict, Set
from deepxube.environments.environment_abstract import State
import sys
from deepxube.utils.data_utils import Logger
import time
import shutil


def gen_ctg_task_files(ctg_states: Dict[int, Set[State]], init_depth: int, final_depth: int, dir: str):
    for d in range(init_depth, final_depth):
        depth_folder = os.path.join(dir, str(d))
        os.makedirs(depth_folder, exist_ok=True)

        with open(os.path.join(depth_folder, "exs.pl"), "w") as f:
            for ctg, states in ctg_states.items():
                for state in states:
                    kind = "neg" if ctg < d else "pos"
                    line = f"{kind}(f([" + ", ".join(f"t{val}" if val > 0 else "b" for val in state.tiles) + "])).\n"
                    f.write(line)

        for filename in ["bk.pl", "bias.pl"]:
            src = os.path.join(dir, filename)
            dst = os.path.join(depth_folder, filename)
            with open(src, "r") as src_file, open(dst, "w") as dst_file:
                dst_file.write(src_file.read())

def log_to_main_file(log_file_path, message):
    with open(log_file_path, "a") as f:
        f.write(message + "\n")

def add_invented_predicates(base_dir, depth, final_depth):
    if depth == final_depth:
        return False
    results_path = os.path.join(base_dir, "results.txt")
    with open(results_path, "r") as f:
        content = f.read()
        if 'NO SOLUTION' in content:
            return False

        count = 0

        for line in content.splitlines():
            if line.startswith(f'd{depth}_r'):
                head_pred = (line.split(':-')[0].strip()).split('(')[0]
                for d in range(depth + 1, final_depth + 1):
                    new_depth_dir = os.path.join(base_dir, str(d))
                    bk_file_path = os.path.join(new_depth_dir, "bk.pl")
                    with open(bk_file_path, "a") as bk_file:
                        if count == 0:
                            bk_file.write("\n\n")
                        bk_file.write("\n" + line)
                next_depth_dir = os.path.join(base_dir, str(depth + 1))
                with open(os.path.join(next_depth_dir, "bias.pl"), "a") as bias_file:
                    if count == 0:
                        bias_file.write("\n\n")
                    bias_file.write(f"\nbody_pred({head_pred}, 1).\ndirection({head_pred}, (in,)).\ntype({head_pred}, (list,)).")
                count = count + 1
    return True
                
    

def run_solver_process(args):
    start_time = time.time()
    dir, depth, timeout, final_depth = args['dir'], args['depth'], args['timeout'], args['final_depth']
    result_path = os.path.join(dir, "results.txt")
    depth_dir = os.path.join(dir, str(depth))
    log_file_path = os.path.join(depth_dir, "log.txt")

    logger_file = args['logger_file']

    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            if f"********** DEPTH {depth} **********" in f.read():
                print(f"[Checkpoint] Skipping depth {depth}")
                log_to_main_file(logger_file, f"[Checkpoint] Skipping depth {depth}")
                return

    solve_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solve_one_depth.py")
    print(solve_script)

    print(f"Solving task at depth {depth}")
    log_to_main_file(logger_file, f"Solving task at depth {depth}")

    try:
        start_time_proc = time.time()
        with open(log_file_path, "w") as log_file:
            # print(f"    [Popper Call] Logging output for depth {depth} to {log_file_path}")
            subprocess.run(
                ["python", solve_script, "--dir", dir, "--depth", str(depth), "--main_results_file", result_path, "--timeout", str(timeout)],
                stdout=log_file,
                stderr=log_file,
                check=True
            )
        print(f"Solved task for depth {depth}\nAppending invented predicates to next depths")
        log_to_main_file(logger_file, f"Solved task for depth {depth}\nAppending invented predicates to next depths")

        res = add_invented_predicates(dir, depth, final_depth)
        if res:
            print(f"Added invented predicates for depth {depth}")
            log_to_main_file(logger_file, f"Added invented predicates for depth {depth}")
        else:
            print(f"No invented predicates added for depth {depth}, next depth (if required) will be executed without predicate invention")
            log_to_main_file(logger_file, f"No invented predicates added for depth {depth}, next depth (if required) will be executed without predicate invention")
        
        print(f"Time to run process for depth {depth} ", time.time()-start_time_proc)
        log_to_main_file(logger_file, f"Time to run process for depth {depth} {time.time()-start_time_proc}")
    except subprocess.CalledProcessError as e:
        with open(log_file_path, "a") as log_file:
            log_file.write(f"\n!!! Popper failed for depth {depth}!!!\n")
            log_file.write(f"    Time taken: {time.time()-start_time_proc}\n")
            log_file.write(f"    Command: {' '.join(e.cmd)}\n")
            log_file.write(f"    Return code: {e.returncode}\n")
            log_file.write(f"    Stdout:\n{e.stdout}\n")
            log_file.write(f"    Stderr:\n{e.stderr}\n")

            log_to_main_file(logger_file, f"\n!!! Popper failed for depth {depth}!!!\n")

    # print(f"[Fork] Finished subprocess for depth {depth}")
    print(f"Total time to run all process for depth {depth} ", time.time()-start_time)
    log_to_main_file(logger_file, f"Total time to run all process for depth {depth} {time.time()-start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--init_depth', type=int, required=True)
    parser.add_argument('--final_depth', type=int, required=True)
    parser.add_argument('--ctg_states_pickle', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          
    HEUR_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))     
    SRC_DIR    = os.path.join(HEUR_ROOT, "puzzle8_rules_source_new") # source for bias.pl / bk.pl

    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"Source dir not found: {SRC_DIR}")
    for _f in ("bias.pl", "bk.pl"):
        if not os.path.exists(os.path.join(SRC_DIR, _f)):
            raise FileNotFoundError(f"Missing {_f} in {SRC_DIR}")

    DEST_DIR = args.dir if os.path.isabs(args.dir) else os.path.join(HEUR_ROOT, args.dir)
    os.makedirs(DEST_DIR, exist_ok=True)

    for _f in ("bias.pl", "bk.pl"):
        shutil.copy(os.path.join(SRC_DIR, _f), os.path.join(DEST_DIR, _f))
        print(f"Copied {_f} → {DEST_DIR}")

    args.dir = DEST_DIR

    time_prog = time.time()

    logger_file = os.path.join(args.dir, "logger_file.txt")

    sys.stdout = Logger(logger_file, "a")

    print(f"Starting program with time limit {args.timeout}")

    with open(args.ctg_states_pickle, "rb") as f:
        ctg_states = pickle.load(f)

    gen_ctg_task_files(ctg_states, args.init_depth, args.final_depth+1, args.dir)

    for d in range(args.init_depth, args.final_depth + 1):
        task_args = {'dir': args.dir, 'depth': d, 'timeout': args.timeout, 'final_depth': args.final_depth, 'logger_file': logger_file}
        run_solver_process(task_args)

    print("Time to run program ", time.time()-time_prog, "\n")
