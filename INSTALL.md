# Environment Setup Guide for HeurSearchILP

This guide explains how to install and configure the full environment required to run **HeurSearchILP**, including **Popper**, **SWI-Prolog with Janus**, and **DeepXube** integration.  

The installation steps are verified for **Red Hat Enterprise Linux 9.5 (Plow)**, 64-bit (x86_64), kernel **5.14.0**.

## System Requirements

Operating System: Red Hat Enterprise Linux 9.5 (Plow)  
Architecture: x86_64 (64-bit)  
Kernel: 5.14.0 or later  
Compiler: GCC 11.5.0  
CMake: ≥ 3.31.1  
Python: 3.10 (Conda environment)

## Python Environment
    * Python 3.8+ (Python 3.10 used during development).
    * A Conda environment (e.g., named `heursearchilp`) is recommended for managing dependencies.
    * Required Python packages (install via `pip` or `conda`):
        * `numpy`
        * `torch` (for multiprocessing, actual PyTorch models not used in this ILP-only flow)
        * `pyswip` (for Prolog interaction from `test_prolog_heur.py`)
        * `janus-swi` (Python interface for SWI-Prolog, used by Popper)
        * `popper-ilp` (the Popper ILP system)

This guide explains how to create and activate the Conda environment required to run **HeurSearchILP** (Popper + DeepXube integration).

## Firstly need to setup swipl,  Install SWI-Prolog 9.3.22 (with Janus Python bindings)

```
wget https://www.swi-prolog.org/download/devel/src/swipl-9.3.22.tar.gz
tar -xzf swipl-9.3.22.tar.gz
cd swipl-9.3.22/
module load gcc11/11.5.0
mkdir build
cd build
module load cmake/3.31.1
cmake ..   -DCMAKE_C_COMPILER=gcc   -DCMAKE_CXX_COMPILER=g++   -DCMAKE_INSTALL_PREFIX=$HOME/.local/swi-prolog-9.3.3-janus   -DINSTALL_DOCUMENTATION=OFF   -DSWIPL_PACKAGES_QT=OFF   -DSWIPL_PACKAGES_X=OFF   -DSWIPL_PACKAGES_GUITOOLS=OFF   -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3   -DPython3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.10   -DPython3_LIBRARY=$CONDA_PREFIX/lib/libpython3.10.so
make -j8
make install
```

## Update PATH and Environment Variables

```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Check path and whether swipl is installed or not
```$HOME/.local/swi-prolog-9.3.3-janus/bin/swipl
swipl
```

## Update path
```
export PATH=$HOME/.local/swi-prolog-9.3.3-janus/bin:$PATH
```

## Check swipl is installed or not
```
swipl
```

## To make these changes permanent
```nano ~/.bashrc
export PATH=$HOME/.local/swi-prolog-9.3.3-janus/bin:$PATH
source ~/.bashrc
```

## Check swipl is installed or not again and swipl version
``` 
swipl
swipl --version
```

## Then we can create and activate conda environment

```
conda create --prefix=<path_to_env>/envs/heursearchilp python=3.10.16
conda activate <path_to_env>/envs/heursearchilp
```

## Install Dependencies

```
pip install deepxube
pip install git+https://github.com/logic-and-learning-lab/Popper@main
conda install -c conda-forge \
  matplotlib=3.10.0 \
  numpy=1.26.4 \
  pillow=11.1.0 \
  pytorch=2.6.0
pip install --user scipy > 1.8
pip install pyswip
conda install nptyping
```

