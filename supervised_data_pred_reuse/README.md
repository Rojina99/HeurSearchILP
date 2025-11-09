## Run code from root directory (i.e. HeurSearchILP)

# Heuristic-Driven ILP Rule Learning and Evaluation for supervised data with predicate reuse

1. `supervised_data_pred_reuse/solve_tasks.py`: For learning ILP rules.
2. `supervised_data_pred_reuse/generate_master_ilp_file.py`: To preprocess data to be used in test script
3. `test_heur_script/test_heuristic_performance_no_a_star.py`:  For evaluating the performance of the learned ILP heuristic.
4.  `test_heur_script/test_heuristic_performance_astar.py`: For evaluating the performance of the learned ILP heuristic by solving them.

`test_heur_script` is common across all training and can be used for all expreiments including predicate reuse, no predicate reuse, supervised and astar dataset. But defined with each readme for clarity.

## Part 1: Learning ILP Rules (`supervised_data_pred_reuse/solve_tasks.py`)

This script iteratively learns Prolog rules for increasing depths (costs-to-go). For each depth, it classifies positive and negative examples using supervised data generated using `ilp_gen_supervised_data/gen_labeled_data.py` and then calls Popper to learn new rules.

### How to Run `supervised_data_pred_reuse/solve_tasks.py`

1.  **Activate Conda Environment:**
    ```bash
    conda activate heursearchilp
    ```

2.  **Ensure SWI-Prolog Environment Variables are Set:**  run `source setup.sh` to setup `PATH` and ensure [Installation Instructions](../INSTALL.md) is read throughly to setup `LD_LIBRARY_PATH`, and `SWI_HOME_DIR` for custom SWI-Prolog installation

3.  **Run the Script:**

    ```bash
    python supervised_data_pred_reuse/solve_tasks.py  \
        --dir {model_dir}/ \
        --init_depth {initial_depth} \
        --final_depth {final_depth} \
        --timeout {timeout_in_seconds}  \
        --ctg_states_pickle {path_to_supervised_dataset}
    ```

    # Actual parameter used for experiment with time timit half an hour and workers 16. Ensure `bk.pl` and `bias.pl` is present in `puzzle8_rules_source_new`. Else create a folder and add `bk.pl` and `bias.pl`. Otherwise will encounter error.

    ```bash
    python supervised_data_pred_reuse/solve_tasks.py \
    --dir tasks_1_5_p_r \
    --init_depth 1 \
    --final_depth 31 \
    --timeout 1800 \
    --ctg_states_pickle data/puzzle8.pkl
    ```

### Parameters for `supervised_data_pred_reuse/solve_tasks.py`:

* `--dir`: Directory to save learned rules.
* `--init_depth`: Starting depth to lern from.
* `--final_depth`: Maximum cost-to-go value for which to learn rules (e.g., 10, 15, or up to 20-25 for 8-puzzle if time permits).
* `--timeout`: Maximum timeout for popper.
* `--ctg_states_pickle`: Path to supervised dataset. The dataset generated with code `ilp_gen_supervised_data/gen_labeled_data.py`.

**Output:**
* `1/`,`2/`, ...`final_depth/`: Subdirecties in `--model_dir` with Popper's working files for each depth.
* `results.txt`: A human-readable text file of all learned rules.

## Part 2: Preprocess generated dataset so it is compatible with `test_heur_script/*.py`
This script takes the learned rules form `supervised_data_pred_reuse/solve_tasks.py` and preprocess it so that it can be used with test scripts.

### How to Run `supervised_data_pred_reuse/generate_master_ilp_file.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `supervised_data_pred_reuse/solve_tasks.py` subdirectories that can be moved to `popper_runs` directory once `popper_runs` directory is created. `popper_runs` directory needs to be created as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking. It will also create `parsed_clause_goals.pkl` and `master_learned_ilp_rules.pkl` file. `parsed_clause_goals.pkl` only necessary to generate `master_learned_ilp_rules.pkl` file.

3.  **Run the Script:**

    ```bash
        python supervised_data_pred_reuse/generate_master_ilp_file.py --dir <learned_supervised_model_dir_path>
    ```

    # Actual parameter used for experiment on the basis of learned folder in part 1

    ```bash
        python supervised_data_pred_reuse/generate_master_ilp_file.py --dir tasks_1_5_p_r
    ```

## Part 3: Evaluating Learned Heuristic (`test_heur_script/test_heuristic_performance_no_a_star.py`)

This script takes the learned rules from `supervised_data_pred_reuse/solve_tasks.py` and evaluates their performance to get heurisitc estimate of each states.

### How to Run `test_heur_script/test_heuristic_performance_no_a_star.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `supervised_data_pred_reuse/solve_tasks.py` has run** and produced the necessary `master_learned_ilp_rules.pkl` file and the `popper_runs` directory structure (as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking).
3.  **Run the Script:**

    ```
    bash
        python test_heur_script/test_heuristic_performance_no_a_star.py \
        --env puzzle8 \
        --model_dir {full_model_dir}/ \
        --num_problems 1140 \
        --max_depths 31 \
        --output_file heuristic_test_results_45_no_astar_{states}.csv \
        --test_dir saved_test_data_45 \
        --data_file data/puzzle8.pkl \
        --trained_heuristic True
    ```

    # Actual parameter used for experiment with all dataset, here we are using atmost 45 states for each depth for test

    ```
    bash
        python test_heur_script/test_heuristic_performance_no_a_star.py \
        --env puzzle8 \
        --model_dir tasks_1_5_p_r/ \
        --num_problems 1140 \
        --max_depths 31 \
        --output_file heuristic_test_results_45_no_astar_all_dataset.csv \
        --test_dir saved_test_data_45 \
        --data_file data/puzzle8.pkl \
        --trained_heuristic True
    ```
    
### Parameters for `test_heur_script/test_heuristic_performance_no_a_star.py`:

* `--env`: Environment name (must match the one rules were learned for).
* `--model_dir`: directory to learned models with popper runs and learned popper rules, Popper runs (e.g., `<model_dir>/popper_runs/`). This is crucial because the heuristic needs to load the correct `bk.pl` context (which includes rules from lower depths) when checking rules for a specific depth via its binary search.
* `--num_problems`: Number of test instances to generate and solve (e.g., 100 or 1000 for thorough testing).
* `--max_depths`: Maximum depth for generating these test instances. Should cover a range of difficulties.
* `--output_file`: Name of the CSV file to store detailed results. Will be stored in path `<model_dir>/output_file`
* `--test_dir`: Directory to save test dataset so A* is tested on same dataset for different results. We use saved_test_data_45 for test in paper
* `--data_file`: Path to 8 puzzle data generated using gen_labeled_data.py
* `--trained_heuristic`: Flag to ensure if trained heuristic or no heuristic mode is used, if want to use trained heuristic need to pass folder with trained results and said the flag to true else set the flag to False with arbitrary folder name not trained folder path

**Output:**
* Terminal output summarizing ground_truth_cost, initial heuristic values and time taken in seconds.
* A CSV file (e.g., `heuristic_test_results_45_no_astar_50.csv`) containing per-problem ground truth, initial heuristic value, time.


## Part 4: Evaluating Learned Heuristic by solving states with astar (`test_heur_script/test_heuristic_performance_astar.py`)

This script takes the learned rules from `supervised_data_pred_reuse/solve_tasks.py` and evaluates their performance as a heuristic in solving a new set of randomly generated problems.

### How to Run `test_heur_script/test_heuristic_performance_astar.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `supervised_data_pred_reuse/solve_tasks.py` has run** and produced the necessary `master_learned_ilp_rules.pkl` file and the `popper_runs` directory structure (as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking).
3.  **Run the Script:**

    ```bash
    python test_heur_script/test_heuristic_performance_astar.py \
        --rules_file {full_model_dir} \
        --popper_run_base_dir {full_model_dir}/popper_runs \
        --test_data_file test_data/saved_test_data_astar/eight_puzzle_states.pkl \
        --time_limit_seconds 600 \
        --max_expansions_per_problem 1000 \
        --output_file {full_model_dir}/puzzle8_output_astar_test.csv
    ```

    # Actual parameter used for experiment for 50 test set

    ```bash
    python test_heur_script/test_heuristic_performance_astar.py \
        --rules_file tasks_1_5_p_r/master_learned_ilp_rules.pkl \
        --popper_run_base_dir tasks_1_5_p_r/popper_runs \
        --test_data_file test_data/saved_test_data_astar/eight_puzzle_states.pkl \
        --time_limit_seconds 1000 \
        --max_expansions_per_problem 10000 \
        --output_file tasks_1_5_p_r/puzzle8_output_depth_31_all_dataset_1000.csv
    ```
    
### Key Parameters for `test_heur_script/test_heuristic_performance_astar.py`:

* `--rules_file`: directory to learned models with popper runs and learned popper rules, Popper runs (e.g., `output_8puzzle_rules/popper_runs/`). This is crucial because the heuristic needs to load the correct `bk.pl` context (which includes rules from lower depths) when checking rules for a specific depth via its binary search.
* `--popper_run_base_dir`: path to learned popper rule (i.e. `<model_dir>/popper_runs`)
* `--test_data_file`: Path to generated test data. We have genrated and used 50 data stored in path `test_data/saved_test_data_astar/eight_puzzle_states.pkl`. The corresponding ground truth is in file `state_costs.csv`.
* `--time_limit_seconds`: Time limit per problem (e.g., 600 seconds = 10 minutes). We used 1000.
* `--max_expansions_per_problem`: Optional overall expansion limit for A* per test problem (e.g., 100,000 or 1,000,000). We used 10000.
* `--output_file`: Name of the CSV file to store detailed results. Will be stored in path `model_dir/output_file`

**Output:**
* Terminal output summarizing solved counts, average time, expansions, and initial heuristic values.
* A CSV file (e.g., `puzzle8_output_depth_31_all_dataset_1000.csv`) containing per-problem statistics, including whether it was solved, cost, expansions, time, initial heuristic value, and the solution path.


