## Run code from root directory (i.e. HeurSearchILP)

# Heuristic-Driven ILP Rule Learning and Evaluation for `ilp_with_astar_data/final_parallel.py`

1. `ilp_with_astar_data/final_parallel.py`: For learning ILP rules for predicate reuse.
2. `test_heur_script/test_heuristic_performance_no_a_star.py`:  For evaluating the performance of the learned ILP heuristic.
3.  `test_heur_script/test_heuristic_performance_astar.py`: For evaluating the performance of the learned ILP heuristic by solving them.

`test_heur_script` is common across all training and can be used for all expreiments including predicate reuse, no predicate reuse, supervised and astar dataset. But defined with each readme for clarity.

## Part 1: Learning ILP Rules (`ilp_with_astar_data/final_parallel.py`)

This script iteratively learns Prolog rules for increasing depths (costs-to-go). For each depth, it generates positive and negative examples using A* search (guided by rules learned for previous depths) and then calls Popper to learn new rules.

### How to Run `ilp_with_astar_data/final_parallel.py`

1.  **Activate Conda Environment:**
    ```bash
    conda activate heursearchilp
    ```

2.  **Ensure SWI-Prolog Environment Variables are Set:**  run `source setup.sh` to setup `PATH` and ensure [Installation Instructions](../INSTALL.md) is read throughly to setup `LD_LIBRARY_PATH`, and `SWI_HOME_DIR` for custom SWI-Prolog installation

3.  **Run the Script:**

    ```bash
    python ilp_with_astar_data/final_parallel.py \
        --env puzzle8 \
        --model_dir {model_dir}/ \
        --learned_program_dir {learned_dir}/ \
        --popper_runner_path logic/popper_runner.py \
        --max_depth_to_learn 31 \
        --states_per_astar_total {states} \
        --max_scramble_steps {scramble} \
        --ilp_convergence_iterations 1 \
        --astar_heuristic_batch_size 1 \
        --max_astar_expansions_per_instance 1000 \
        --num_astar_workers 16 \
        --astar_weight 1.0 \
        --astar_retry_multiplier 5 \
        --astar_retry_increment 5000
    ```

    # Actual parameter used for experiment with --states_per_astar_total 50, 100, 500, 1000, 2000 and --num_astar_workers 16

    ```bash
    python ilp_with_astar_data/final_parallel.py \
        --env puzzle8 \
        --model_dir puzzle8_output_depth_31_new_4_v1/ \
        --learned_program_dir puzzle8_rules_source_new/ \
        --popper_runner_path logic/popper_runner.py \
        --max_depth_to_learn 31 \
        --states_per_astar_total 50 \
        --max_scramble_steps 500 \
        --ilp_convergence_iterations 1 \
        --astar_heuristic_batch_size 1 \
        --max_astar_expansions_per_instance 1000 \
        --num_astar_workers 16 \
        --astar_weight 1.0 \
        --astar_retry_multiplier 5 \
        --astar_retry_increment 5000
    ```

### Parameters for `ilp_with_astar_data/final_parallel.py`:

* `--env`: Environment name (e.g., `puzzle8`).
* `--model_dir`: Directory to save learned rules (`master_learned_ilp_rules.pkl`), checkpoints, and Popper run subdirectories.
* `--learned_program_dir`: Directory containing the initial `bk.pl` and `bias.pl` for Popper.
* `--popper_runner_path`: Path to `logic/popper_runner.py`.
* `--max_depth_to_learn`: Maximum cost-to-go value for which to learn rules (e.g., 10, 15, or up to 20-25 for 8-puzzle if time permits).
* `--max_scramble_steps`: Maximum depth of random walks from the goal state to generate A* start states. For learning deeper rules, this might need to be higher (e.g., 30-60).
* `--states_per_astar_total`: Total A* instances to run per ILP refinement iteration per depth. More states can lead to better examples for Popper but increases A* time (e.g., 50-2000).
* `--ilp_convergence_iterations`: Number of times to refine rules with new A* examples for a single depth (e.g., 1-3). Currenlty only 1 is used.
* `--astar_heuristic_batch_size`: Batch size for heuristic calls within A* (e.g., 100-500). Only 1 is used.
* `--max_astar_expansions_per_instance`: Initial limit on A* node expansions for a single problem. The script has adaptive retry logic. Start with a moderate value (e.g., 500-2000). The adaptive logic will increase it if no positive examples are found.
* `--num_astar_workers`: Number of parallel A* worker processes (e.g., match number of CPU cores available, minus a couple).
* `--astar_weight`: Weight for Weighted A* during example generation (e.g., `1.0` for standard A*, `2.0` or `3.0` for faster, greedier search). Only 1 is used.
* `--astar_retry_multiplier`, `--astar_retry_increment`: Parameters for the adaptive A* expansion limit. But `allow_current_retry` is set to false inside code so no effect.

**Output:**
* `master_learned_ilp_rules.pkl`: A pickle file in `--model_dir` containing all learned rules, mapping depth (int) to a list of rule strings.
* `main_loop_checkpoint.pkl`: Checkpoint file.
* `popper_runs/`: Subdirectory in `--model_dir` with Popper's working files for each depth.
* `final_learned_rules.txt`: A human-readable text file of all learned rules.

## Part 2: Evaluating Learned Heuristic (`test_heur_script/test_heuristic_performance_no_a_star.py`)

This script takes the learned rules from `ilp_with_astar_data/final_parallel.py` and evaluates their performance to get heurisitc estimate of each states.

### How to Run `test_heur_script/test_heuristic_performance_no_a_star.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `ilp_with_astar_data/final_parallel.py` has run** and produced the necessary `master_learned_ilp_rules.pkl` file and the `popper_runs` directory structure (as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking).
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
        --model_dir puzzle8_output_depth_31_new_4_v1/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ip_states_each_depth_50_ss_500_a_star_bs_1_ci_1_a_star_exp_start_1000_workers_16_rm_5.0_ri_5000 \
        --num_problems 1140 \
        --max_depths 31 \
        --output_file heuristic_test_results_45_no_astar_50.csv \
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

This script takes the learned rules from `ilp_with_astar_data/final_parallel.py` and evaluates their performance as a heuristic in solving a new set of randomly generated problems.

### How to Run `test_heur_script/test_heuristic_performance_astar.py`

1.  **Activate Conda Environment** (if not already active).
2.  **Ensure `ilp_with_astar_data/final_parallel.py` has run** and produced the necessary `master_learned_ilp_rules.pkl` file and the `popper_runs` directory structure (as `PureILPHeuristic` needs to access the depth-specific `bk.pl` files from these directories for context during rule checking).
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
        --rules_file puzzle8_output_depth_31_new_4_v1/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ip_states_each_depth_50_ss_500_a_star_bs_1_ci_1_a_star_exp_start_1000_workers_16_rm_5.0_ri_5000/master_learned_ilp_rules.pkl \ 
        --popper_run_base_dir puzzle8_output_depth_31_new_4_v1/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ip_states_each_depth_50_ss_500_a_star_bs_1_ci_1_a_star_exp_start_1000_workers_16_rm_5.0_ri_5000/popper_runs \
        --test_data_file test_data/saved_test_data_astar/eight_puzzle_states.pkl \
        --time_limit_seconds 1000 \
        --max_expansions_per_problem 10000 \
        --output_file puzzle8_output_depth_31_new_4_v1/env_puzzle8_pdir_puzzle8_rules_source_new_depth_31_ip_states_each_depth_50_ss_500_a_star_bs_1_ci_1_a_star_exp_start_1000_workers_16_rm_5.0_ri_5000/puzzle8_output_depth_31_all_dataset_1000.csv
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

**Same format of script will work for `ilp_with_astar_data/final_parallel_no_predicate_invention.py` except use`ilp_with_astar_data/final_parallel_no_predicate_invention.py` instead of  `ilp_with_astar_data/final_parallel.py` while running scripts and use different `--model_dir` so learned model will not be overwritten in case of no predicate reuse.**
