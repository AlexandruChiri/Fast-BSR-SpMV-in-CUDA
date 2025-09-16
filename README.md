# Documentation
	The documentation explaining the theory behind the custom BSR SpMV implementations is in the PDF file from the `docs` directory.



# Sources
## C source
	The file `normal.c` contains all the BSR SpMV custom C+OpenMP implementations.



## CUDA sources
	The file `cu_batched_row-major.cu` contains the naive BSR SpMV custom CUDA implementation. It accumulates partial results directly into shared memory.

	The file `cu_batched_experiment.cu` contains the BSR SpMV custom CUDA implementation which uses a thread-private accumulation vector and maintains the strided access pattern on array `Ax` that the row-major approach uses. The vector is allocated by NVCC version 11.4 in VRAM, so the latency for accumulation is high.

	The file `cu_batched_single-acc.cu` contains the BSR SpMV custom CUDA implementation which uses a single thread-private scalar accumulator and groups the threads of every thread block in teams of workers. The thread not grouped in teams do not do any work.

	The file `cu_batched_tiny.cu` contains the BSR SpMV custom CUDA implementation that uses a vector accumulator. Unlike the `experiment` approach, it uses loops with known margins in order to access it so it yields two advantages at the cost of partially trading coalescing (impact mitigated by the L1 cache): it uses register accumulation and reduces integer computational overhead per nonzero.



# Data generation script
	markov_generator.py
	Run with "python markov_generator.py <N> <R> <C> <density> <output_directory>"
	Creates a new directory (or uses an existing one) and creates a set of 6 input test files in it:
		Ai.bin - the row indices of the matrix
		Aj.bin - the column indices of the matrix
		Ax.bin - the nonzero data of the matrix
		Xx.bin - the multiplicand vector
		Yx.bin - the accumulator vector
		etalon_matvec_{M}_{N}_{density}_1000.bin - reference file for the result of 1000 accumulations
	The generated matrix is block-banded stochastic with the dimension `N*R`x`N*C` and the block shape of `RxC` with a global density of `density`, these variables being the input variables passed to the stript in the command line interface.



# Test scripts
## CPU test script
	test_all.py
	Run with "python test_all.py \"[<num_threads_list>]\" <input_directory>"
	Takes a list of nonzero thread count numbers (e.g. "[1, 2, 4, 8]") and runs the SciPy BSR SpMV implementation (singlethreaded) and all the BSR SpMV custom CPU implementations for the given thread counts 1000 times on the test data from the given directory.

## NumPy test script
	test_numpy.py
	Run with "python test_numpy.py \"[<num_threads_list>]\" <input_directory>"
	Same as `test_all.py`, but runs the numpy implementation instead.

## GPU test script
	test_cuda.py
	Run with "python test_cuda.py <library_file> <input_directory>"
	Runs the given custom BSR SpMV CUDA implementation (the `bsr_matvec` C wrapper function) 1000 times on the test data from the given directory.
	Will expect to be given the number of registers the kernel uses per thread. Giving an inacurate number could result in degraded performance or even cause the test to fail.

## cuSPARSE test script
	test_cuda.py
	Run with "python test_cusparse.py <input_directory>"
	Same as `test_cuda.py`, but runs the BSR SpMV cuSPARSE implementation (cusparseDbsrmv) instead.
