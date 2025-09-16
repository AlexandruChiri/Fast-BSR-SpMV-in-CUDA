#include <omp.h>
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>


#include <stdio.h>
#include <stdatomic.h>

#define __DEBUG__

#ifdef __DEBUG__
	#define PRINT_VAR(x, id) printf(#x " = " id "\n", x);
#else
	#define PRINT_VAR(x, id)
#endif

int tid;
#pragma omp threadprivate(tid)


#define I int32_t
#define T double
#define npy_intp int64_t


// Do dense matrix-vector multiplication and accumulate the result to a given vector
inline __attribute__((always_inline)) void gemv(
	const I R,
	const I C,
	const T A[],
	const T x[],
	T y[]
) {
	for (I i = 0; i < R; i++) {
		T dot = 0;
		const T *const A_row = A + C * i;
		for (I j = 0; j < C; ++j) {
			dot += A_row[j] * x[j];
		}
		y[i] += dot;
	}
}


extern void bsr_matvec_sequential(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta)
{
	assert(R > 0 && C > 0);

	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;

	for (I i = 0; i < n_row; ++i)
		Yx[i] *= beta;

	T *const sum_private = malloc(R * sizeof(T));

	for (I i = 0; i < n_brow; ++i) {
		for (I bi = 0; bi < R; bi++) {
			sum_private[bi] = 0;
		}
		for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
			const I j = Aj[jj];
			const T * A = Ax + RC * jj;
			const T * x = Xx + (npy_intp)C * j;
			gemv(R, C, A, x, sum_private);
		}
		T * y = Yx + (npy_intp)R * i;
		for (I bi = 0; bi < R; bi++) {
			y[bi] += alpha * sum_private[bi];
		}
	}

	free(sum_private);
}


extern void bsr_matvec_granular_row(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads)
{
	assert(R > 0 && C > 0);

	if (num_threads != 0)
		omp_set_num_threads(num_threads);

	// if (R == 1 && C == 1) {
	//	 //use CSR for 1x1 blocksize
	//	 csr_matvec(n_brow, n_bcol, Ap, Aj, Ax, Xx, Yx);
	//	 return;
	// }

	// T *const sum = calloc(n_brow * R, sizeof(T));
	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;

	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		T *const sum_private = malloc(R * sizeof(T));

		#pragma omp for schedule(dynamic, 1)
		for (I i = 0; i < n_brow; ++i) {
			for (I bi = 0; bi < R; bi++) {
				sum_private[bi] = 0;
			}
			for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
				const I j = Aj[jj];
				const T * A = Ax + RC * jj;
				const T * x = Xx + (npy_intp)C * j;
				gemv(R, C, A, x, sum_private);
			}
			T * y = Yx + (npy_intp)R * i;
			for (I bi = 0; bi < R; bi++) {
				y[bi] += alpha * sum_private[bi];
			}
		}

		free(sum_private);
	}
	// free(sum);
}


extern void bsr_matvec_fair_auto_dummy(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads)
{
	assert(R > 0 && C > 0);

	if (num_threads != 0)
		omp_set_num_threads(num_threads);

	// T *const sum = calloc(n_brow * R, sizeof(T));
	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;


	#pragma omp parallel for schedule(guided, 1)
	for (I i = 0; i < n_row; ++i)
		Yx[i] *= beta;

	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel
	{
		sum_private = malloc(R * sizeof(T));
	}

	for (I i = 0; i < n_brow; ++i) {
		#pragma omp parallel
		{
			for (I bi = 0; bi < R; bi++) {
				sum_private[bi] = 0;
			}
			#pragma omp for schedule(guided, 1)
			for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
				const I j = Aj[jj];
				const T * A = Ax + RC * jj;
				const T * x = Xx + (npy_intp)C * j;
				gemv(R, C, A, x, sum_private);
			}
			T * y = Yx + (npy_intp)R * i;
			for (I bi = 0; bi < R; bi++) {
				sum_private[bi] *= alpha;
				#pragma omp atomic
				y[bi] += sum_private[bi];
			}
		}
	}

	#pragma omp parallel
	{
		free(sum_private);
	}
	// free(sum);
}


extern void bsr_matvec_fair_auto(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const int32_t chunks_per_thread)
{
	assert(R > 0 && C > 0);

	if (num_threads != 0)
		omp_set_num_threads(num_threads);

	// T *const sum = calloc(n_brow * R, sizeof(T));
	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;


	I blocks_per_chunk = Ap[n_brow] / (num_threads * chunks_per_thread);
	if (blocks_per_chunk < 1)
		blocks_per_chunk = 1;


	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel
	{
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		char did_work = 0;
		sum_private = malloc(R * sizeof(T));

		for (I i = 0; i < n_brow; ++i) {
			for (I bi = 0; bi < R; bi++) {
				sum_private[bi] = 0;
			}
			#pragma omp for schedule(dynamic, blocks_per_chunk) nowait
			for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
				did_work = 1;
				const I j = Aj[jj];
				const T * A = Ax + RC * jj;
				const T * x = Xx + (npy_intp)C * j;
				gemv(R, C, A, x, sum_private);
			}
			if (did_work) {
				T * y = Yx + (npy_intp)R * i;
				for (I bi = 0; bi < R; bi++) {
					sum_private[bi] *= alpha;
					#pragma omp atomic
					y[bi] += sum_private[bi];
				}
			}
		}
		free(sum_private);
	}
}


extern void bsr_matvec_task(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const int32_t chunks_per_thread)
{
	assert(R > 0 && C > 0);

	if (num_threads <= 0)
		*(int32_t*)(&num_threads) = omp_get_num_threads();
	else if (num_threads != 0)
		omp_set_num_threads(num_threads);

	// T *const sum = calloc(n_brow * R, sizeof(T));
	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;


	I blocks_per_chunk = Ap[n_brow] / (num_threads * chunks_per_thread);
	if (blocks_per_chunk < 1)
		blocks_per_chunk = 1;


	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel num_threads(num_threads)
	{
		sum_private = calloc(R, sizeof(T));
	}

	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		#pragma omp for schedule(dynamic, 1)
		// #pragma omp single
		for (I i = 0; i < n_brow; ++i) {
			// #pragma omp task firstprivate(i)
			if (Ap[i + 1] - Ap[i] > 0) {
				I start, stop;
				I end = Ap[i + 1] - (Ap[i + 1] - Ap[i]) % blocks_per_chunk;
				for (start = Ap[i], stop = Ap[i] + blocks_per_chunk; start < end; start = stop, stop += blocks_per_chunk) {
					#pragma omp task firstprivate(i, start, stop)
					{
						for (I bi = 0; bi < R; bi++) {
							sum_private[bi] = 0;
						}
						// if (stop > Ap[i + 1])
						// 	stop = Ap[i + 1];
						for (I jj = start; jj < stop; ++jj) {
							const I j = Aj[jj];
							const T * A = Ax + RC * jj;
							const T * x = Xx + (npy_intp)C * j;
							gemv(R, C, A, x, sum_private);
						}
						T * y = Yx + (npy_intp)R * i;
						for (I bi = 0; bi < R; bi++) {
							#pragma omp atomic
							y[bi] += sum_private[bi] * alpha;
						}
					}
				}
				if (stop > Ap[i + 1])
					stop = Ap[i + 1];
				#pragma omp task firstprivate(i, end)
				{
					for (I bi = 0; bi < R; bi++) {
						sum_private[bi] = 0;
					}
					for (I jj = end; jj < Ap[i + 1]; ++jj) {
						const I j = Aj[jj];
						const T * A = Ax + RC * jj;
						const T * x = Xx + (npy_intp)C * j;
						gemv(R, C, A, x, sum_private);
					}
					T * y = Yx + (npy_intp)R * i;
					for (I bi = 0; bi < R; bi++) {
						#pragma omp atomic
						y[bi] += sum_private[bi] * alpha;
					}
				}
			}
		}

		free(sum_private);
	}
	// free(sum);
}

// The arguments `Ap`, `Aj, `Ax`, `Xx`, `Yx`, `R`, `C`, `RC, `sum_private`, and `sum` bear the same meanings as in the `bsr_matvec` function
// `i` is the row of blocks on which the section of blocks between the indices `start` and `stop` is
inline __attribute__((always_inline)) void one_row_of_blocks(
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const I R,
	const I C,
	const npy_intp RC,
	const I i,
	const I start,
	const I stop,
	T *const sum_private,
	const T alpha
) {
	// Initialize initial sum section with zeros
	// Initialize the sum segment for the current row to zero
	for (I bi = 0; bi < R; bi++) {
		sum_private[bi] = 0;
	}

	// Go from the first untill the last block from the `start`-`stop` sequence
	for (I jj = start; jj < stop; ++jj) {
		const I j = Aj[jj];
		const T *A = Ax + RC * jj;
		const T *x = Xx + (npy_intp)C * j;
		gemv(R, C, A, x, sum_private);
	}
	// Add the accumulated sum to the final vector with atomic operations
	T *const y = Yx + (npy_intp)R * i;
	for (I bi = 0; bi < R; bi++) {
		#pragma omp atomic
		y[bi] += sum_private[bi] * alpha;
	}
}

// `n_brow` and `n_bcol` represent how many rows and columns of blocks there are
// `R` and `C` represent the number of rows and columns from each block
// `Ap`, `Aj`, and `Ax` bear the same meaning as the arrays indptr, indices, and data of a scipy.sparse BSR matrix. Lets refer that BSR matrix as `A` in our case
// `num_threads` represents the number of threads that will be used
// Basically, this function performs the following SAXPY operation: Yx[:] = alpha * (A @ Xx) + beta * Yx
// the "_fair" part from the function name refers to balanced distribution of the workload across the threads
extern void bsr_matvec_fair(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const int32_t chunks_per_thread)
{
	assert(R > 0 && C > 0);

	if (num_threads > 0)
		omp_set_num_threads(num_threads);

	// T *const sum = calloc(n_brow * R, sizeof(T));
	// if (!sum) {
	// 	perror("sum calloc failed");
	// 	exit(EXIT_FAILURE);
	// }
	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;

	I chunks_cnt = num_threads * chunks_per_thread;
	I blocks_per_chunk = Ap[n_brow] / chunks_cnt;
	if (blocks_per_chunk < 1)
		blocks_per_chunk = 1;

	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel num_threads(num_threads)
	{
		// We have the result of the BSR matrix vector multiplication stored in `sum`, so we just have to operate on the `Yx` vector
		// No atomic operations needed because an index can only be processed by one thread
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		sum_private = malloc(R * sizeof(T));
		if (!sum_private) {
			perror("sum_private malloc failed");
			exit(EXIT_FAILURE);
		}

		#pragma omp single
		{
			// The indexes of the first and last row a task needs to deal with (they may be equal)
			I start_index = 0;
			I stop_index = 0;
			I start = 0;
			// Total number of blocks
			const I end = Ap[n_brow];

			// Split the workload in chunks
			for (I chunk_index = 0; chunk_index < chunks_cnt; ++chunk_index) {
				I stop = ((npy_intp)Ap[n_brow] * (chunk_index + 1)) / chunks_cnt;

				while (Ap[stop_index + 1] < stop)
					++stop_index;

				// Create a task which will go from the block on the index `start` (from the `start_index` row) to the block on the index `stop` (from the `stop_index` row)
				#pragma omp task firstprivate(start_index, stop_index, start, stop)
				{

					if (start_index == stop_index) {
						one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, start_index, start, stop, sum_private, alpha);
					}
					else {
						I i = start_index;
						one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, start, Ap[i + 1], sum_private, alpha);
						for (I i = start_index + 1; i < stop_index; ++i) {
							one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i], Ap[i + 1], sum_private, alpha);
						}
						i = stop_index;
						one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i], stop, sum_private, alpha);
					}
				}
				start_index = stop_index;
				start = stop;
			}
		}

		free(sum_private);
	}
	// free(sum);
}

// `n_brow` and `n_bcol` represent how many rows and columns of blocks there are
// `R` and `C` represent the number of rows and columns from each block
// `Ap`, `Aj`, and `Ax` bear the same meaning as the arrays indptr, indices, and data of a scipy.sparse BSR matrix. Lets refer that BSR matrix as `A` in our case
// `num_threads` represents the number of threads that will be used
// Basically, this function performs the following SAXPY operation: Yx[:] = alpha * (A @ Xx) + beta * Yx
// the "_fair" part from the function name refers to balanced distribution of the workload across the threads
extern void bsr_matvec_fair_batched(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const I chunks_cnt,
	const npy_intp chunk_index[],
	const I chunk_row[])
{
	assert(R > 0 && C > 0);

	if (num_threads > 0)
		omp_set_num_threads(num_threads);

	// T *const sum = calloc(n_brow * R, sizeof(T));
	// if (!sum) {
	// 	perror("sum calloc failed");
	// 	exit(EXIT_FAILURE);
	// }
	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;

	// // How many blocks are in a chunk of 10000 elements
	// I blocks_per_chunk = Ap[n_brow] / (num_threads * 100);
	// if (blocks_per_chunk < 1)
	// 	blocks_per_chunk = 1;

	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel num_threads(num_threads)
	{
		// We have the result of the BSR matrix vector multiplication stored in `sum`, so we just have to operate on the `Yx` vector
		// No atomic operations needed because an index can only be processed by one thread
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		sum_private = malloc(R * sizeof(T));
		if (!sum_private) {
			perror("sum_private malloc failed");
			exit(EXIT_FAILURE);
		}

		#pragma omp for schedule(dynamic, 1)
		for (I chunk = 0; chunk < chunks_cnt; ++chunk) {
			I start_row = chunk_row[chunk];
			I stop_row = chunk_row[chunk + 1];

			I start_index = chunk_index[chunk] / RC;
			I stop_index = chunk_index[chunk + 1] / RC;

			// If `start_row` and `stop_row` are identical then the `start_index` and `stop_index` blocks are on the same row
			if (start_row == stop_row) {
				one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, start_row, start_index, stop_index, sum_private, alpha);
			}
			else {
				// Handle the first row of blocks from the task
				I i = start_row;
				one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, start_index, Ap[i + 1], sum_private, alpha);

				// Handle the complete rows from between the `start_row` and `stop_row` ones
				for (I i = start_row + 1; i < stop_row; ++i) {
					one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i], Ap[i + 1], sum_private, alpha);
				}

				// Handle the last row of blocks from the task
				i = stop_row;
				one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i], stop_index, sum_private, alpha);
			}
		}

		free(sum_private);
	}
	// free(sum);
}


#define FAST_32_DIVU(x, m, p) (((uint64_t)(x) * (m)) >> (p))

void magicgu32(uint32_t nmax, uint32_t d, uint32_t *m, uint32_t *p) {
    if (d <= 1) {
        printf("magicgu32: invalid divisor\n");
        return;
    }

    const uint64_t nc = ((nmax + 1) / d) * d - 1;
    const uint32_t nbits = 32 - __builtin_clz(nmax);

    for (*p = 0; *p <= 2 * nbits; ++(*p)) {
        const uint64_t pow2p = 1ULL << *p;
        if (pow2p > nc * (d - 1 - (pow2p - 1) % d)) {
            *m = ((pow2p + d - 1 - (pow2p - 1) % d) / d);
            return;
        }
    }
    printf("magicgu32: Can't find p, something is wrong.\n");
}




// The arguments `Ap`, `Aj, `Ax`, `Xx`, `Yx`, `R`, `C`, `RC, `sum_private`, and `sum` bear the same meanings as in the caller function
// `i` is the index of the row of blocks on which the section of blocks between the indices `start` and `stop` is
inline __attribute__((always_inline)) void one_row_of_blocks_unblocked_nodivisions(
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const I R,
	const I C,
	const npy_intp RC,
	const I i,
	const npy_intp start_index,
	const npy_intp stop_index,
	T *const sum_private,
	T const alpha,
	const uint32_t m_RC,
	const uint32_t p_RC,
	const uint32_t m_C,
	const uint32_t p_C
) {
	for (I bi = 0; bi < R; bi++) {
		sum_private[bi] = 0;
	}

	for (npy_intp index = start_index; index < stop_index; ++index) {
        // Compute the index of the block to which the element belongs
        const I jj = FAST_32_DIVU(index, m_RC, p_RC);	// index / RC
        // Compute the index of the element within its block
        const I index_b = index - jj * RC;	// index % RC

        // Store the column index of the block
        const I j = Aj[jj];

        // Store the row index of the element within its block
        const I i_b = FAST_32_DIVU(index_b, m_C, p_C);	// index_b / C;
        // Store the column index of the element within its block
        const I j_b = index_b - i_b * C;	// index_b % C

        // Store the row and column coordinates of the element within the whole matrix
        // const I i_final = R * i + i_b;
        const I j_final = C * j + j_b;

        // Reminder that we are working with an auxiliary array meant only for one row of blocks
        // So we add to the element on the same index as the row index of the current element into its block, not into the whole matrix
        sum_private[i_b] += Ax[index] * Xx[j_final];
	}

	// Add the accumulated sum to the final vector with atomic operations
	T *const y = Yx + (npy_intp)R * i;
	for (I bi = 0; bi < R; bi++) {
		sum_private[bi] *= alpha;
		#pragma omp atomic
		y[bi] += sum_private[bi];
	}
}






// The arguments `Ap`, `Aj, `Ax`, `Xx`, `Yx`, `R`, `C`, `RC, `sum_private`, and `sum` bear the same meanings as in the caller function
// `i` is the index of the row of blocks on which the section of blocks between the indices `start` and `stop` is
inline __attribute__((always_inline)) void one_row_of_blocks_unblocked(
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const I R,
	const I C,
	const npy_intp RC,
	const I i,
	const npy_intp start_index,
	const npy_intp stop_index,
	T *const sum_private,
	T const alpha
) {
	for (I bi = 0; bi < R; bi++) {
		sum_private[bi] = 0;
	}

	for (npy_intp index = start_index; index < stop_index; ++index) {
        // Compute the index of the block to which the element belongs
        const I jj = index / RC;
        // Compute the index of the element within its block
        const I index_b = index % RC;

        // Store the column index of the block
        const I j = Aj[jj];

        // Store the row index of the element within its block
        const I i_b = index_b / C;
        // Store the column index of the element within its block
        const I j_b = index_b % C;

        // Store the row and column coordinates of the element within the whole matrix
        // const I i_final = R * i + i_b;
        const I j_final = C * j + j_b;

        // Reminder that we are working with an auxiliary array meant only for one row of blocks
        // So we add to the element on the same index as the row index of the current element into its block, not into the whole matrix
        sum_private[i_b] += Ax[index] * Xx[j_final];
	}

	// Add the accumulated sum to the final vector with atomic operations
	T *const y = Yx + (npy_intp)R * i;
	for (I bi = 0; bi < R; bi++) {
		sum_private[bi] *= alpha;
		#pragma omp atomic
		y[bi] += sum_private[bi];
	}
}




// `n_brow` and `n_bcol` represent how many rows and columns of blocks there are
// `R` and `C` represent the number of rows and columns from each block
// `Ap`, `Aj`, and `Ax` bear the same meaning as the arrays indptr, indices, and data of a scipy.sparse BSR matrix. Lets refer that BSR matrix as `A` in our case
// `num_threads` represents the number of threads that will be used
// Basically, this function performs the following SAXPY operation: Yx[:] = alpha * (A @ Xx) + beta * Yx
// the "_fair" part from the function name refers to balanced distribution of the workload across the threads
extern void bsr_matvec_fair_unblocked(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const I chunks_per_thread)
{
	assert(R > 0 && C > 0);

	if (num_threads > 0)
		omp_set_num_threads(num_threads);

	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;
	const I chunks_cnt = num_threads * chunks_per_thread;

	const npy_intp nnz = Ap[n_brow] * RC;


	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel num_threads(num_threads)
	{
		// We have the result of the BSR matrix vector multiplication stored in `sum`, so we just have to operate on the `Yx` vector
		// No atomic operations needed because an index can only be processed by one thread
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		sum_private = malloc(R * sizeof(T));
		if (!sum_private) {
			perror("sum_private malloc failed");
			exit(EXIT_FAILURE);
		}

		I start_row = 0;
		I stop_row = 0;
		npy_intp start_index = 0;

		#pragma omp single
		for (I chunk = 0; chunk < chunks_cnt; ++chunk) {
			npy_intp stop_index = (nnz * (chunk + 1)) / chunks_cnt;

			while (Ap[stop_row + 1] * RC < stop_index) {
				// printf("%d * %ld < %ld\n", Ap[stop_row + 1], RC, stop_index);
				++stop_row;
			}

			#pragma omp task firstprivate(start_row, stop_row, start_index, stop_index)
			{
				if (start_row == stop_row) {
					one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, start_row, start_index, stop_index, sum_private, alpha);
				}
				else {
					// Handle the first row of blocks from the task
					I i = start_row;
					one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, start_index, Ap[i + 1] * RC, sum_private, alpha);

					// Handle the complete rows from between the `start_row` and `stop_row` ones
					for (I i = start_row + 1; i < stop_row; ++i) {
						one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i] * RC, Ap[i + 1] * RC, sum_private, alpha);
					}

					// Handle the last row of blocks from the task
					i = stop_row;
					one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i] * RC, stop_index, sum_private, alpha);
				}
			}
			start_index = stop_index;
			start_row = stop_row;
		}

		free(sum_private);
	}
	// free(sum);
}




// `n_brow` and `n_bcol` represent how many rows and columns of blocks there are
// `R` and `C` represent the number of rows and columns from each block
// `Ap`, `Aj`, and `Ax` bear the same meaning as the arrays indptr, indices, and data of a scipy.sparse BSR matrix. Lets refer that BSR matrix as `A` in our case
// `num_threads` represents the number of threads that will be used
// Basically, this function performs the following SAXPY operation: Yx[:] = alpha * (A @ Xx) + beta * Yx
// the "_fair" part from the function name refers to balanced distribution of the workload across the threads
extern void bsr_matvec_fair_batched_unblocked(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const I chunks_cnt,
	const npy_intp chunk_index[],
	const I chunk_row[])
{
	assert(R > 0 && C > 0);

	if (num_threads > 0)
		omp_set_num_threads(num_threads);

	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;

	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel num_threads(num_threads)
	{
		// We have the result of the BSR matrix vector multiplication stored in `sum`, so we just have to operate on the `Yx` vector
		// No atomic operations needed because an index can only be processed by one thread
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		sum_private = malloc(R * sizeof(T));
		if (!sum_private) {
			perror("sum_private malloc failed");
			exit(EXIT_FAILURE);
		}

		I start_row = 0;
		I stop_row = 0;
		I start_index = 0;

		#pragma omp for schedule(dynamic, 1)
		for (I chunk = 0; chunk < chunks_cnt; ++chunk) {
			I start_row = chunk_row[chunk];
			I stop_row = chunk_row[chunk + 1];

			I start_index = chunk_index[chunk];
			I stop_index = chunk_index[chunk + 1];

			// If `start_row` and `stop_row` are identical then the `start_index` and `stop_index` blocks are on the same row
			if (start_row == stop_row) {
				one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, start_row, start_index, stop_index, sum_private, alpha);
			}
			else {
				// Handle the first row of blocks from the task
				I i = start_row;
				one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, start_index, Ap[i + 1] * RC, sum_private, alpha);

				// Handle the complete rows from between the `start_row` and `stop_row` ones
				for (I i = start_row + 1; i < stop_row; ++i) {
					one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i] * RC, Ap[i + 1] * RC, sum_private, alpha);
				}

				// Handle the last row of blocks from the task
				i = stop_row;
				one_row_of_blocks_unblocked(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i] * RC, stop_index, sum_private, alpha);
			}
		}

		free(sum_private);
	}
	// free(sum);
}




// `n_brow` and `n_bcol` represent how many rows and columns of blocks there are
// `R` and `C` represent the number of rows and columns from each block
// `Ap`, `Aj`, and `Ax` bear the same meaning as the arrays indptr, indices, and data of a scipy.sparse BSR matrix. Lets refer that BSR matrix as `A` in our case
// `num_threads` represents the number of threads that will be used
// Basically, this function performs the following SAXPY operation: Yx[:] = alpha * (A @ Xx) + beta * Yx
// the "_fair" part from the function name refers to balanced distribution of the workload across the threads
extern void bsr_matvec_fair_batched_unblocked_nodivisions(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads,
	const I chunks_cnt,
	const npy_intp chunk_index[],
	const I chunk_row[])
{
	assert(R > 0 && C > 0);

	if (num_threads > 0)
		omp_set_num_threads(num_threads);

	const npy_intp RC = (npy_intp)R * C;
	const I n_row = R * n_brow;

	uint32_t m_RC, p_RC;
	uint32_t m_C, p_C;

	magicgu32(Ap[n_brow] * RC, RC, &m_RC, &p_RC);
	magicgu32(RC, C, &m_C, &p_C);

	static T *sum_private;
	#pragma omp threadprivate(sum_private)

	#pragma omp parallel num_threads(num_threads)
	{
		// We have the result of the BSR matrix vector multiplication stored in `sum`, so we just have to operate on the `Yx` vector
		// No atomic operations needed because an index can only be processed by one thread
		#pragma omp for schedule(guided, 1)
		for (I i = 0; i < n_row; ++i)
			Yx[i] *= beta;

		sum_private = malloc(R * sizeof(T));
		if (!sum_private) {
			perror("sum_private malloc failed");
			exit(EXIT_FAILURE);
		}

		I start_row = 0;
		I stop_row = 0;
		I start_index = 0;

		#pragma omp for schedule(dynamic, 1)
		for (I chunk = 0; chunk < chunks_cnt; ++chunk) {
			I start_row = chunk_row[chunk];
			I stop_row = chunk_row[chunk + 1];

			I start_index = chunk_index[chunk];
			I stop_index = chunk_index[chunk + 1];

			// If `start_row` and `stop_row` are identical then the `start_index` and `stop_index` blocks are on the same row
			if (start_row == stop_row) {
				one_row_of_blocks_unblocked_nodivisions(Ap, Aj, Ax, Xx, Yx, R, C, RC, start_row, start_index, stop_index, sum_private, alpha, m_RC, p_RC, m_C, p_C);
			}
			else {
				// Handle the first row of blocks from the task
				I i = start_row;
				one_row_of_blocks_unblocked_nodivisions(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, start_index, Ap[i + 1] * RC, sum_private, alpha, m_RC, p_RC, m_C, p_C);

				// Handle the complete rows from between the `start_row` and `stop_row` ones
				for (I i = start_row + 1; i < stop_row; ++i) {
					one_row_of_blocks_unblocked_nodivisions(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i] * RC, Ap[i + 1] * RC, sum_private, alpha, m_RC, p_RC, m_C, p_C);
				}

				// Handle the last row of blocks from the task
				i = stop_row;
				one_row_of_blocks_unblocked_nodivisions(Ap, Aj, Ax, Xx, Yx, R, C, RC, i, Ap[i] * RC, stop_index, sum_private, alpha, m_RC, p_RC, m_C, p_C);
			}
		}

		free(sum_private);
	}
	// free(sum);
}










extern void bsr_matvec_1000(
	const I n_brow,
	const I n_bcol,
	const I R,
	const I C,
	const I Ap[],
	const I Aj[],
	const T Ax[],
	const T Xx[],
		  T Yx[],
	const double alpha,
	const double beta,
	const int32_t num_threads)
 {
	for (int i = 1; i <= 1000; ++i) {
		bsr_matvec_fair(
			n_brow,
			n_bcol,
			R,
			C,
			Ap,
			Aj,
			Ax,
			Xx,
			Yx,
			alpha,
			beta,
			num_threads,
			100
		);
		if (i % 200 == 0)
			printf("Done iteration %d\n", i);
	}
}





void *read_file_to_memory(const char* const filename, npy_intp *const elems, const I elem_size) {
	*elems = 0;
	FILE* file = fopen(filename, "rb"); // Open the file in binary mode
	if (!file) {
		perror("Error opening file");
		return NULL;
	}

	// Seek to the end of the file to determine its size
	if (fseek(file, 0, SEEK_END) != 0) {
		perror("Error seeking in file");
		fclose(file);
		return NULL;
	}

	long file_size = ftell(file); // Get the file size
	if (file_size < 0) {
		perror("Error getting file size");
		fclose(file);
		return NULL;
	}

	rewind(file); // Go back to the start of the file

	// Allocate memory to hold the file contents
	void* buffer = malloc(file_size);
	if (!buffer) {
		perror("Error allocating memory");
		fclose(file);
		return NULL;
	}

	// Read the file contents into the buffer
	size_t bytes_read = fread(buffer, 1, file_size, file);
	if (bytes_read != (size_t)file_size) {
		perror("Error reading file");
		free(buffer);
		fclose(file);
		return NULL;
	}
	// for (int x = 0; x < file_size; ++x) {
	// 	printf("%3llu ", ((uint8_t*)buffer)[x]);
	// }
	// printf("\n");
	*elems = file_size / elem_size;

	fclose(file); // Close the file
	return buffer; // Return the buffer
}

int big_test() {
	const I M = 16000;
	const I N = 18000;
	const T alpha = -1.0;
	const T beta = 1.0;
	const I R = 10;
	const I C = 10;

	if ((M % R) || (N % C)) {
		printf("Row number doesn't devide by block height or column number doesn't divide by block width: (%d, %d) vs (%d, %d)\n", M, N, R, C);
		return 0;
	}

	const I n_brow = M / R;
	const I n_bcol = N / C;
	int32_t num_threads = 2;

	npy_intp dummy;

	printf("reading Ap\n");
	I *Ap = read_file_to_memory("Ap", &dummy, dummy);
	printf("reading Aj\n");
	I *Aj = read_file_to_memory("Aj", &dummy, dummy);
	printf("reading Ax\n");
	T *Ax = read_file_to_memory("Ax", &dummy, dummy);
	printf("reading Xx\n");
	T *Xx = read_file_to_memory("/home/student/Desktop/Pt_Facultate/Master_2/Semestrul_1/Dizertatie/Xx", &dummy, dummy);
	printf("reading Yx\n");
	T *Yx = read_file_to_memory("/home/student/Desktop/Pt_Facultate/Master_2/Semestrul_1/Dizertatie/Yx", &dummy, dummy);

	printf("Multiplying\n");
	bsr_matvec_1000(
		n_brow,
		n_bcol,
		R,
		C,
		Ap,
		Aj,
		Ax,
		Xx,
		Yx,
		alpha,
		beta,
		num_threads
	);

	printf("GATA!\n");
	free(Ap);
	free(Aj);
	free(Ax);
	free(Xx);
	free(Yx);
}



#include <string.h>

int small_test() {
	const T alpha = -1.0;
	const T beta = 1.0;
	const I R = 10;
	const I C = 10;

	int32_t num_threads = 2;

	const char *const directory = "test_chunked/";
	const int dir_len = strlen(directory);
	char file[100];
	strcpy(file, directory);


	printf("reading Ap\n");
	strcpy(file + dir_len, "Ap");
	npy_intp elems;
	const I *const Ap = read_file_to_memory(file, &elems, sizeof(I));
	const I n_brow = elems - 1;
	const I n_bcol = n_brow;

	printf("reading Aj\n");
	strcpy(file + dir_len, "Aj");
	const I *const Aj = read_file_to_memory(file, &elems, sizeof(I));
	const npy_intp nblocks = elems;


	printf("reading Ax\n");
	strcpy(file + dir_len, "Ax");
	const T *const Ax = read_file_to_memory(file, &elems, sizeof(T));
	const npy_intp nnz = elems;
	assert(nblocks * R * C == nnz);

	printf("reading Xx\n");
	strcpy(file + dir_len, "Xx");
	const T *const Xx = read_file_to_memory(file, &elems, sizeof(T));
	const I N = elems;

	printf("reading Yx\n");
	strcpy(file + dir_len, "Yx");
	T *const Yx = read_file_to_memory(file, &elems, sizeof(T));
	const I M = elems;

	printf("reading etalon\n");
	strcpy(file + dir_len, "etalon_matvec_100_100_0.1_1000.bin");
	T *etalon = read_file_to_memory(file, &elems, sizeof(T));
	assert(elems == M);



	printf("Multiplying\n");
	bsr_matvec_1000(
		n_brow,
		n_bcol,
		R,
		C,
		Ap,
		Aj,
		Ax,
		Xx,
		Yx,
		alpha,
		beta,
		num_threads
	);

	printf("GATA!\n");

	char correct = 1;
	#define epsilon 0.00001
	for (I i = 0; i < N; ++i) {
		T diff = Yx[i] - etalon[i];
		if (diff > epsilon || diff < - epsilon) {
			correct = 0;
			printf("correct = False");
			break;
		}
	}
	#undef epsilon

	if (correct) {
		printf("correct = True\n");
	}

	free((void*)etalon);
	free((void*)Ap);
	free((void*)Aj);
	free((void*)Ax);
	free((void*)Xx);
	free((void*)Yx);
}



#undef I
#undef T
#undef npy_intp






int main() {
	small_test();
}
