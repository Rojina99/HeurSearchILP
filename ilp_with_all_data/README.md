
## Run code from root directory (i.e. HeurSearchILP)

# Heuristic-Driven ILP Rule Learning and Evaluation for `ilp_with_all_data/final_parallel_all_dataset.py`

This is the code for learning heuristic with supervised data in same way as learning with astar data `ilp_with_all_data/final_parallel.py`.

This was not actual script used for experiments os not added in main README.md. But can be used as alternative way to learn heuristc with supervised data and predicate reuse.

1. `ilp_with_all_data/final_parallel_all_dataset.py`: For learning ILP rules for predicate reuse.
2. `test_heur_script/test_heuristic_performance_no_a_star.py`:  For evaluating the performance of the learned ILP heuristic.
3.  `test_heur_script/test_heuristic_performance_astar.py`: For evaluating the performance of the learned ILP heuristic by solving them.

## Part 1: Learning ILP Rules (`ilp_with_all_data/final_parallel_all_dataset.py`)

This script iteratively learns Prolog rules for increasing depths (costs-to-go). For each depth, it classifies positive and negative examples from supervised dataset generated with `ilp_gen_supervised_data/gen_labeled_data.py` guided by rules learned for previous depths and then calls Popper to learn new rules.

### How to Run `ilp_with_all_data/final_parallel_all_dataset.py`

1.  **Activate Conda Environment:**
    ```bash
    conda activate heursearchilp
    ```

2.  **Ensure SWI-Prolog Environment Variables are Set:**  run `source setup.sh` to setup `PATH` and ensure [Installation Instructions](../INSTALL.md) is read throughly to setup `LD_LIBRARY_PATH`, and `SWI_HOME_DIR` for custom SWI-Prolog installation

3.  **Run the Script:**

    ```bash
        python ilp_with_all_data/final_parallel_all_dataset.py \
        --env puzzle8 \
        --model_dir {model_dir_path} \
        --learned_program_dir puzzle8_rules_source_new \
        --popper_runner_path logic/popper_runner_timeout.py \
        --max_depth_to_learn 31 \
        --ilp_convergence_iterations 1 \
        --data_file data/puzzle8.pkl\
        --time_out {timeout}
    ```

     # Actual parameter used for experiment

    ```bash
        python ilp_with_all_data/final_parallel_all_dataset.py  \
        --env puzzle8 \
        --model_dir puzzle_8_output_depth_31_new_all_dataset \
        --learned_program_dir puzzle8_rules_source_new \
        --popper_runner_path logic/popper_runner_timeout.py \
        --max_depth_to_learn 31 \
        --ilp_convergence_iterations 1 \
        --data_file data/puzzle8.pkl 
        --time_out 3600
     ```

### Parameters for `ilp_with_all_data/final_parallel_all_dataset.py`:

* `--env`: Environment name (e.g., `puzzle8`).
* `--model_dir`: Directory to save learned rules (`master_learned_ilp_rules.pkl`), checkpoints, and Popper run subdirectories.
* `--learned_program_dir`: Directory containing the initial `bk.pl` and `bias.pl` for Popper.
* `--popper_runner_path`: Path to `logic/popper_runner_timeout.py`.
* `--max_depth_to_learn`: Maximum cost-to-go value for which to learn rules (e.g., 10, 15, or up to 20-25 for 8-puzzle if time permits).
* `--max_scramble_steps`: Maximum depth of random walks from the goal state to generate A* start states. For learning deeper rules, this might need to be higher (e.g., 30-60).
* `--ilp_convergence_iterations`: Number of times to refine rules with new A* examples for a single depth (e.g., 1-3). Currenlty only 1 is used.
*  `--data_file`: path to supervised dataset generated using `ilp_gen_supervised_data/gen_labeled_data.py`
*  `--time_out`: maximum time for popper.

**Output:**
* `master_learned_ilp_rules.pkl`: A pickle file in `--model_dir` containing all learned rules, mapping depth (int) to a list of rule strings.
* `main_loop_checkpoint.pkl`: Checkpoint file.
* `popper_runs/`: Subdirectory in `--model_dir` with Popper's working files for each depth.
* `final_learned_rules.txt`: A human-readable text file of all learned rules.

## Part 2: Evaluating Learned Heuristic (`test_heur_script/test_heuristic_performance_no_a_star.py`)

This script takes the learned rules from `ilp_with_all_data/final_parallel_all_dataset.py` and evaluates their performance to get heurisitc estimate of each states.

### How to Run `test_heur_script/test_heuristic_performance_no_a_star.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `ilp_with_all_data/final_parallel_all_dataset.py` has run** and produced the necessary `master_learned_ilp_rules.pkl` file and the `popper_runs` directory structure (as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking).
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

    # Actual parameter used for experiment with --states_per_astar_total 50, here we are using atmost 45 states for each depth for test

    ```
    bash
        python test_heur_script/test_heuristic_performance_no_a_star.py \
        --env puzzle8 \
        --model_dir puzzle_8_output_depth_31_new_all_dataset/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ci_1_time_out_1800_all \
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


## Part 3: Evaluating Learned Heuristic by solving states with astar (`test_heur_script/test_heuristic_performance_astar.py`)

This script takes the learned rules from `ilp_with_all_data/final_parallel_all_dataset.py` and evaluates their performance as a heuristic in solving a new set of randomly generated problems.

### How to Run `test_heur_script/test_heuristic_performance_astar.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `ilp_with_all_data/final_parallel_all_dataset.py` has run** and produced the necessary `master_learned_ilp_rules.pkl` file and the `popper_runs` directory structure (as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking).
3.  **Run the Script:**

    ```bash
    python test_heur_script/test_heuristic_performance_astar.py \
        --rules_file {full_model_dir} \
        --popper_run_base_dir {full_model_dir}/popper_runs \
        --test_data_file test_data/saved_test_data_astar/eight_puzzle_states.pkl \
        --time_limit_seconds 600 \
        --max_expansions_per_problem 1000 \
        --output_file {full_model_dir}/puzzle8_output_astar_test.csv \
    ```

    # Actual parameter used for experiment for 50 test set

    ```bash
    python test_heur_script/test_heuristic_performance_astar.py \
        --rules_file puzzle_8_output_depth_31_new_all_dataset/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ci_1_time_out_1800_all/master_learned_ilp_rules.pkl \ 
        --popper_run_base_dir puzzle_8_output_depth_31_new_all_dataset/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ci_1_time_out_1800_all/popper_runs \
        --test_data_file test_data/saved_test_data_astar/eight_puzzle_states.pkl \
        --time_limit_seconds 1000 \
        --max_expansions_per_problem 10000 \
        --output_file puzzle_8_output_depth_31_new_all_dataset/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ci_1_time_out_1800_all/puzzle8_output_depth_31_all_dataset_1000.csv
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
``


