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
        We have also included archive to store older and new version of background file and bias file. Not required for running and getting reusults for now.
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
python ilp_gen_supervised_data/gen_labeled_data.py --env puzzle8 --depth 31 --save data/puzzle8.pkl
```

The generated dataset has 181440 states in total

There are four main experiments in the paper **Astar data without predicate reuse**, **Astar data with predicate reuse**, **Supervised data without predicate reuse**, **Supervised data with predicate reuse**. Script to generate each of the experiments will be shown.

## 1. Running code for Predicate Reuse and No Predicate Reuse with Astar data

- [**Running Instructions**](ilp_with_astar_data/README.md)

## 2. Running code for Predicate Reuse with Supervised data (i.e. data/puzzle.pkl) generated earlier

- [**Running Instructions**](supervised_data_pred_reuse/README.md)

## 3. Running code for No Predicate Reuse with Supervised data (i.e. data/puzzle.pkl) generated earlier

- [**Running Instructions**](supervised_data_no_pred_reuse/README.md)

## Download Learned Models and Test Results

You can download the trained heuristic models and evaluation outputs here:

**[learned_models_and_test_result.zip](https://github.com/Rojina99/HeurSearchILP/releases/tag/learned_models)**

This pre-release bundles trained models and evaluation outputs.

This ZIP file contains models learned using different version of HeurSearchILP

# The learned models are in thee folder 
- `a_star_data_pr`: Model learned with a_star data with predicate reuse
- `a_star_data_no_pr` : Model learned with a_star data withou predicate reuse
- `supervised_data` : Model learned with supervised data without predicate reuse and with predicate reuse

# Result of A* Search with learned models is in folder `test_result_for_learned_models_with_a_star`. We use model learned with sample size 1000, 2000 for the one learned with a_star data.

For `test_result_for_learned_models_with_a_star` nodes is expanded nodes instead of nodes generated and nodes/second is calculated for nodes expanded per time in second for each fifty samples.

## Contributors

This repository and the accompanying experiments were developed by:

- **Rojina Panta**  
  Department of Computer Science and Engineering  
  University of South Carolina
  GitHub: [Rojina99](https://github.com/Rojina99) 

- **Vedant Khandelwal**  
  Department of Computer Science and Engineering  
  University of South Carolina
  GitHub: [khvedant02](https://github.com/khvedant02)  

- **Celeste Veronese**  
  Department of Computer Science  
  University of Verona
  GitHub: [vrncst](https://github.com/vrncst)

- **Daniele Meli**  
  Department of Computer Science  
  University of Verona  

- **Forest Agostinelli**  
  Department of Computer Science and Engineering  
  University of South Carolina
  GitHub: [forestagostinelli](https://github.com/forestagostinelli)

## License

This project is licensed under the terms of the **MIT License**.  
See the full text in the [LICENSE](LICENSE) file.

If you encounter any issues while running the code or notice any errors, **please feel free to contact us** at **rpanta@email.sc.edu**





 