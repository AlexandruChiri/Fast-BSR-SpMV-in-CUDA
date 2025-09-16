#!/bin/bash

CC=gcc
NVCC=module load libraries/cuda-11.4 && nvcc


OPT_FLAGS=-O3 -g
SHARED_LIB_FLAGS_CPU=-shared -fPIC
SHARED_LIB_FLAGS_GPU=-shared -Xcompiler -fPIC
PARALLEL_FLAGS=-fopenmp
GPU_ARCH=-arch=sm_60
DISPLAY_REG=--ptxas-options=-v

CUDA_FLAGS=${GPU_ARCH} ${DISPLAY_REG} ${SHARED_LIB_FLAGS_GPU}



cpu_so_files=normal.dll
cuda_so_files=cu_batched_row-major.dll cu_batched_experiment.dll cu_batched_single-acc.dll cu_batched_single-acc.dll cu_batched_tiny.dll
all_so_files=${cpu_so_files} ${cuda_so_files}

cpu_all: ${cpu_so_files}

cuda_all: ${cuda_so_files}


cu_batched_row-major.dll: cu_batched_row-major.cu
	${NVCC} ${CUDA_FLAGS} --maxrregcount=64 -o cu_batched_row-major.dll cu_batched_row-major.cu
	echo

cu_batched_experiment.dll: cu_batched_experiment.cu
	${NVCC} ${CUDA_FLAGS} --maxrregcount=64 -o cu_batched_experiment.dll cu_batched_experiment.cu
	echo

cu_batched_single-acc.dll: cu_batched_single-acc.cu
	${NVCC} ${CUDA_FLAGS} --maxrregcount=64 -o cu_batched_single-acc.dll cu_batched_single-acc.cu
	echo

cu_batched_tiny.dll: cu_batched_tiny.cu
	${NVCC} ${CUDA_FLAGS} --maxrregcount=77 -o cu_batched_tiny.dll cu_batched_tiny.cu
	echo



normal.dll: normal.c
	${CC} normal.c ${SHARED_LIB_FLAGS_CPU} ${OPT_FLAGS} ${PARALLEL_FLAGS} -o normal.dll


clean:
	rm -f ${all_so_files}

