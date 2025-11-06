import argparse
import pdb
import re
import os
import pickle
import tempfile
import shutil

from nnet.nnet_utils import load_and_process_clause_to_get_unique_clause

def rename_results_file(base_dir):
    """
    Safely rename results.txt to final_learned_rules.txt.
    Ensures atomicity and no partial rename under any failure.
    """
    src = os.path.join(base_dir, "results.txt")
    dst = os.path.join(base_dir, "final_learned_rules.txt")

    # destination already exists, nothing to do
    if os.path.exists(dst):
        print(f"Using existing final_learned_rules.txt")
        return dst

    # source missing
    if not os.path.exists(src):
        raise FileNotFoundError(f"Neither results.txt nor final_learned_rules.txt found in {base_dir}")

    tmp_dst = dst + ".tmp"

    try:
        # copy src to temp file, same filesystem, atomic rename ready
        shutil.copy2(src, tmp_dst)

        # verify copy same size
        src_size = os.path.getsize(src)
        tmp_size = os.path.getsize(tmp_dst)
        if src_size != tmp_size:
            raise IOError(f"Copy verification failed: size mismatch ({src_size} vs {tmp_size})")

        # rename temp to final
        os.replace(tmp_dst, dst)
        print(f"Renamed results.txt to final_learned_rules.txt (atomic)")

        # remove old file only after confirmed success
        os.remove(src)
        return dst

    except Exception as e:
        print(f"Rename failed: {e}")
        # Cleanup any partial files
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)
        raise e

# def rename_results_file(base_dir):
#     """Rename results.txt to final_learned_rules.txt if it exists."""
#     src = os.path.join(base_dir, "results.txt")
#     dst = os.path.join(base_dir, "final_learned_rules.txt")
#     if os.path.exists(src):
#         os.rename(src, dst)
#         print(f"Renamed results.txt to final_learned_rules.txt")
#     elif not os.path.exists(dst):
#         raise FileNotFoundError(f"Neither results.txt nor final_learned_rules.txt found in {base_dir}")
#     return dst

def move_popper_runs_atomic(base_dir, rules_filename):
    """Move numeric folders (1–31) atomically into popper_runs/, copy final_learned_rules.txt safetly."""
    final_dest = os.path.join(base_dir, "popper_runs")
    temp_dest = os.path.join(base_dir, "popper_runs_tmp")

    if os.path.exists(final_dest):
        print(f"Folder {final_dest} already exists, skipping creation.")
        return final_dest

    # Create temp destination folder
    os.makedirs(temp_dest, exist_ok=True)

    moved_folders = []
    try:
        # Move folders 1–31 into temporary destination
        for i in range(1, 32):
            folder_name = str(i)
            src = os.path.join(base_dir, folder_name)
            dst = os.path.join(temp_dest, folder_name)
            if os.path.exists(src) and os.path.isdir(src):
                print(f"Moving {src} to {dst}")
                shutil.move(src, dst)
                moved_folders.append((src, dst))

        # Copy results.txt to all_popper_solutions.txt
        src_results = os.path.join(base_dir, rules_filename)
        dst_results = os.path.join(temp_dest, "all_popper_solutions.txt")
        if os.path.exists(src_results):
            shutil.copy(src_results, dst_results)
            print(f"Copied results.txt to {dst_results}")
        else:
            print("No results.txt found in base directory.")

        # Rename temp folder to final folder
        os.rename(temp_dest, final_dest)
        print(f"Move complete. All folders safely relocated to {final_dest}")
        return final_dest

    except Exception as e:
        print(f"Error during move: {e}. Rolling back...")
        # Roll back any partially moved folders
        for src, dst in moved_folders:
            if os.path.exists(dst) and not os.path.exists(src):
                try:
                    shutil.move(dst, src)
                except Exception as rollback_err:
                    print(f"Rollback failed for {dst}: {rollback_err}")
        # Clean up temp folder
        if os.path.exists(temp_dest):
            shutil.rmtree(temp_dest, ignore_errors=True)
        raise e

def parse_and_process_results(result_dir):
    """Parse final_learned_rules.txt, extract clauses, and process ILP rules."""
    results_path = os.path.join(result_dir, "final_learned_rules.txt")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"final_learned_rules.txt not found in {result_dir}")

    with open(results_path, "r") as file:
        lines = file.readlines()

    clause_goals = {}
    current_depth = None

    for line in lines:
        line = line.strip()

        # Detect depth line
        depth_match = re.match(r"\*+ DEPTH (\d+) \*+", line.strip())
        if depth_match:
            current_depth = int(depth_match.group(1))
            clause_goals[current_depth] = []


        # if re.match(r"^f\d+_\d+\(V0\):-", line.strip()):
        if re.match(r"^f\(V0\):-", line.strip()):
            # rule_body = re.sub(r"^f\d+_\d+\(V0\):-", "goal :-", line.strip()).rstrip('.').strip()
            rule_body = re.sub(r"^f\d+_\d+\(V0\):-", "f(V0):-", line.strip()).rstrip('.').strip()
            clause_goals[current_depth].append(rule_body)

    
    clause_goals_sorted = dict(sorted(clause_goals.items()))

    parsed_path = os.path.join(result_dir, "parsed_clause_goals.pkl")
    with open(parsed_path, "wb") as f:
        pickle.dump(clause_goals_sorted, f)
        # pickle.dump(clause_goals, f)

    # with open(os.path.join(RESULT_DIR, "parsed_clause_goals.pkl"), "rb") as f:
    #     clause_goals_read = pickle.load(f)

    processed_clause = load_and_process_clause_to_get_unique_clause(parsed_path)

    master_path = os.path.join(result_dir, "master_learned_ilp_rules.pkl")
    with open(master_path, "wb") as f:
        pickle.dump(processed_clause, f)

    # with open(os.path.join(RESULT_DIR, "parsed_clause_goals.pkl"), "rb") as f:
        # clause_goals_read = pickle.load(f)

    # with open(os.path.join(RESULT_DIR, "master_learned_ilp_rules.pkl"), "rb") as f:
    #     clause_goals_read = pickle.load(f)

    print(f"Parsing and processing completed.\nResults stored in: {result_dir}")

# RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result_data')
# RESULT_DIR = os.path.join(os.path.dirname(__file__), 'pred_inv_new_bk')
# RESULT_DIR = "/Users/rojinapanta/results/all_dataset_no_p_i/tasks"
# RESULT_DIR = "/Users/rojinapanta/results/all_dataset_no_p_i/tasks_8"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to result tasks directory")
    args = parser.parse_args()

    # --- Resolve relative vs absolute path based on this file location ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # e.g., .../HeurSearchILP/supervised_data_no_pred_reuse
    HEUR_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))    # e.g., .../HeurSearchILP

    # If --dir is relative, place it under HEUR_ROOT; else keep as-is
    RESULT_DIR = args.dir if os.path.isabs(args.dir) else os.path.join(HEUR_ROOT, args.dir)

    if not os.path.isdir(RESULT_DIR):
        raise ValueError(f"Invalid directory: {RESULT_DIR}")

    print(f"Working in: {RESULT_DIR}")

    # Rename results.txt first
    rules_filename = os.path.basename(rename_results_file(RESULT_DIR))

    # Move folders atomically and copy renamed file 
    popper_runs_dir = move_popper_runs_atomic(RESULT_DIR, rules_filename)

    # Parse and process ILP results
    parse_and_process_results(RESULT_DIR)

    print(f"\n All numeric folders safely moved to: {popper_runs_dir}")
    print(f" Processed clause files saved in: {RESULT_DIR}")

# Write to output file in pretty format
# with open(os.path.join(RESULT_DIR, "parsed_clause_goals.txt"), "w") as out_file:
#     out_file.write("# clause_goals = {\n")
#     for depth, rules in sorted(clause_goals.items()):
#         out_file.write(f"    {depth}: [")
#         for i, rule in enumerate(rules):
#             sep = "," if i < len(rules) - 1 else ""
#             out_file.write(f"\"{rule}\"{sep} ")
#         out_file.write("],\n")
#     out_file.write("# }")