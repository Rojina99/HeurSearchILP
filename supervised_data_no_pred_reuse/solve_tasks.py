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
from pathlib import Path
import shutil


def gen_ctg_task_files(ctg_states: Dict[int, Set[State]], init_depth, final_depth: int, dir):
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

def run_solver_process(args):
    start_time = time.time()
    dir, depth, timeout = args['dir'], args['depth'], args['timeout']
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

    print(f"[Fork] Starting subprocess for depth {depth}")
    log_to_main_file(logger_file, f"[Fork] Starting subprocess for depth {depth}")

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
        print(f"[Fork] Finished subprocess for depth {depth}")
        log_to_main_file(logger_file, f"[Fork] Finished subprocess for depth {depth}")
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
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--ctg_states_pickle', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        
    HEUR_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   
    SRC_DIR    = os.path.join(HEUR_ROOT, "puzzle8_rules_source_new")

    # Validate source dir and files
    for _p in [SRC_DIR, os.path.join(SRC_DIR, "bias.pl"), os.path.join(SRC_DIR, "bk.pl")]:
        if not os.path.exists(_p):
            raise FileNotFoundError(f"Required source missing: {_p}")

    DEST_DIR = args.dir if os.path.isabs(args.dir) else os.path.join(HEUR_ROOT, args.dir)
    os.makedirs(DEST_DIR, exist_ok=True)

    for fname in ("bias.pl", "bk.pl"):
        shutil.copy(os.path.join(SRC_DIR, fname), os.path.join(DEST_DIR, fname))
        print(f"Copied {fname} → {DEST_DIR}")

    args.dir = DEST_DIR

    time_prog = time.time()

    logger_file = os.path.join(args.dir, "logger_file.txt")

    sys.stdout = Logger(logger_file, "a")

    print(f"Starting program with time limit {args.timeout}")

    with open(args.ctg_states_pickle, "rb") as f:
        ctg_states = pickle.load(f)

    gen_ctg_task_files(ctg_states, args.init_depth, args.final_depth+1, args.dir)

    task_args = [{'dir': args.dir, 'depth': d, 'timeout': args.timeout, 'logger_file': logger_file} for d in range(args.init_depth, args.final_depth + 1)]
    with Pool(processes=args.procs) as pool:
        pool.map(run_solver_process, task_args)

    print("Time to run program ", time.time()-time_prog)
