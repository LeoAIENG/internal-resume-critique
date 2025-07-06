#!/bin/bash
# Function/Variable
function conda_info() {
	field=$1
    conda info -a |\
    	grep -i "$field" |\
		cut -d ":" -f 2|\
		xargs echo |\
		sed 's/ *$//g'
}
function jupyter_info() {
	field=$1
    jupyter --version |\
    	grep -i "$field" |\
		cut -d ":" -f 2|\
		xargs echo |\
		sed 's/ *$//g'
}
function numba_info() {
	field=$1
    numba -s |\
    	grep -i "$field" |\
		cut -d ":" -f 2|\
		xargs echo |\
		sed 's/ *$//g'
}

## System Info
printf "## GLOBAL INFO\n"
# System Python Version
printf "Conda Python Version: "
conda_info "python version"
# Conda Base
printf "Conda Base Path: "
conda_info "CONDA_ROOT"
# Conda Base
printf "Conda Base Version: "
conda_info "conda version"

## Environment Info
printf "\n## ENVIRONMENT INFO\n"
# Active Conda Environment
printf "Active Environment: "
conda_info "active environment"
# Python Env Version
printf "Environment Python Version: "
python --version
# Python Env Path
printf "Environment Python Path: "
which python
# Ipykernel Version
printf "Environment IPython Version: "
jupyter_info "IPython"
# Ipykernel Version
printf "Environment IPykernel Version: "
jupyter_info "ipykernel"

## Hardware Info
printf "\n## GPU INFO:\n"
# CUDA Info
printf "CUDA Device Initialized "
numba_info "CUDA Device Initialized"
# GPU Info
printf "\nGPU Info: "
nvidia-smi