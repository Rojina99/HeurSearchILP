The repository contains the implementation for the paper **_Inductive Logic Programming for Heuristic Search_**.

To set up the repository, refer to the accompanying **`requirements.txt`** and **`environment.yml`** files for environment dependencies.  

A verified installation guide for **Red Hat Enterprise Linux 9.5 (Plow)**, 64-bit (**x86_64 architecture**, kernel **5.14.0**) is provided here:

- [**Installation Instructions**](INSTALL.md)

## Prerequisites

1. **Project Structure:**
    The scripts assume a certain directory structure for imports (e.g., `environments.env_utils`, `search.astar`, `heur.test_prolog_heur`). Ensure your files are organized accordingly or adjust import paths.
    * `ilp_with_astar_data/` contains required script for learning with and without predicate reuse with astar data   
        *`final_parallel.py`
        *`final_parallel_no_predicate_invention.py`
    * `supervised_data_no_pred_reuse/` contains required script for learning without predicate reuse with supervised data
        * `solve_one_depth.py`
        * `solve_tasks.py`
        * `generate_master_ilp_file.py`
    * `supervised_data_pred_reuse/` contains required script for learning with predicate reuse with supervised data  
        * `solve_one_depth.py`
        * `solve_tasks.py`
        * `generate_master_ilp_file.py`
    * `test_heur_script` contains required script for testing
        * `test_heuristic_performance_astar.py`
        * `test_heuristic_performance_no_a_star.py`
    * `setup.sh` script to set environment paths and variables
    * `logic/popper_runner.py`, `logic/popper_utils.py`, `logic/popper_runner_timeout.py`
    * `heur/test_prolog_heur.py` (contains `PureILPHeuristic`, `HeurILP` with binary search)
    * `heur/heur_utils.py`
    * `search/astar.py`
    * `environments/n_puzzle.py`, `environments/env_utils.py`
    * `nnet/nnet_utils.py` (for `load_and_process_clause_to_get_unique_clause`)
    * `puzzle8_rules_source_new/` (or similar, specified by `--learned_program_dir`)
        * `bk.pl` (base background knowledge for Popper)
        * `bias.pl` (Popper's declarative bias)
        I have also included archive to store older and new version of background file and bias file. Not required for running and getting reusults for now.
    * `test_data` folder that contains test dataset used in paper, you can generate and use your own test dataset as well. It is in pickle format.
        * `saved_test_data_45`
        * `saved_test_data_astar`

Once the environment setup is completed run :

```
source setup.sh
```

from root directory (i.e. HeurSearchILP) to configure the required environment paths and variables.

Then generate **supervised dataset** for **8-puzzle** with

```
mkdir data
python ilp_gen_supervised_data/gen_labeled_data.py --env puzzle8 --depth 31 --save test_data/puzzle8.pkl
```

The generated dataset has 181440 states in total

There are four main experiments in the paper **Astar data without predicate reuse**, **Astar data with predicate reuse**, **Supervised data without predicate reuse**, **Supervised data with predicate reuse**. Script to generate each of the experiments will be shown.

## 1. Running code for Predicate Reuse and No Predicate Reuse with Astar data

- [**Running Instructions**](ilp_with_astar_data/README.md)

## 2. Running code for Predicate Reuse with Supervised data (i.e. data/puzzle.pkl) generated earlier

- [**Running Instructions**](supervised_data_pred_reuse/README.md)

## 3. Running code for No Predicate Reuse with Supervised data (i.e. data/puzzle.pkl) generated earlier

- [**Running Instructions**](supervised_data_no_pred_reuse/README.md)

## Contributors

This repository and the accompanying experiments were developed by:

- **Rojina Panta**  
  Department of Computer Science and Engineering  
  University of South Carolina  

- **Vedant Khandelwal**  
  Department of Computer Science and Engineering  
  University of South Carolina  

- **Celeste Veronese**  
  Department of Computer Science  
  University of Verona  

- **Daniele Meli**  
  Department of Computer Science  
  University of Verona  

- **Forest Agostinelli**  
  Department of Computer Science and Engineering  
  University of South Carolina

If you encounter any issues while running the code or notice any errors, **please feel free to contact us** at **rpanta@email.sc.edu**.




 