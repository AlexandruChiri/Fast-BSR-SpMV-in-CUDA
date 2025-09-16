#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

typedef int32_t I;
typedef double T;
typedef int64_t npy_intp;



#define GROUP_SIZE 32  // Must divide R and C well, but we’ll generalize later



// Handles one row of blocks starting from the block index `start` and going untill `stop`
__device__ __forceinline__ void one_row_of_blocks(
    const I Ap[],
    const I Aj[],
    const T Ax[],
    const T Xx[],
          T sum[],
    const I i,
    const npy_intp start_index,
    const npy_intp stop_index,
    T *const sum_private,
    const I R,
    const I C,
    const npy_intp RC
) {
    // Initialize initial sum section with zeros
    __syncthreads();
    for (I bi = threadIdx.x; bi < R; bi += blockDim.x) {
        sum_private[bi] = 0;
    }
    __syncthreads();

    // Go from the first untill the last block from the `start`-`stop` sequence
    // const npy_intp start_index = start * RC;
    // const npy_intp stop_index = stop * RC;



    for (npy_intp index = start_index + threadIdx.x; index < stop_index; index += blockDim.x) {
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


        // Lets process the element on 
        const I j_final = C * j + j_b;

        // HERE
        atomicAdd(&sum_private[i_b], Ax[index] * __ldg(&Xx[j_final]));
        // sum_private[i_b] += Ax[index] * Xx[j_final];
    }
    __syncthreads();

    // Add the accumulated sum to the final vector with atomic operations
    T *const y = sum + (npy_intp)R * i;
    for (I bi = threadIdx.x; bi < R; bi += blockDim.x) {
        atomicAdd(&y[bi], sum_private[bi]);
        // y[bi] += sum_private[bi];
    }
}





__global__ void bsr_matvec_kernel(
    const I n_brow,
    const I Ap[],
    const I Aj[],
    const T Ax[],
    const T Xx[],
          T Yx[],
    const I R,
    const I C,
    const npy_intp RC,
    const npy_intp chunk_index[],
    const I chunk_row[],
    const I chunks_cnt)
{
    // Block id
    const int bid = blockIdx.x;
    const int B = gridDim.x;

    // extern __shared__ uint8_t shared[];
    extern __shared__ T sum_private[];
    // extern T *x_private = (T*)shared + R;



    for (I chunk = bid; chunk < chunks_cnt; chunk += B) {
        const npy_intp start_index = chunk_index[chunk];
        const npy_intp stop_index = chunk_index[chunk + 1];

        const I start_row = chunk_row[chunk];
        const I stop_row = chunk_row[chunk + 1];

        if (start_row == stop_row) {
            one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, start_row, start_index, stop_index, sum_private, R, C, RC);
        }
        else {
            // Handle the first row of blocks from the task
            I i = start_row;
            one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, i, start_index, Ap[i + 1] * RC, sum_private, R, C, RC);

            // Handle the complete rows from between the `start_row` and `stop_row` ones
            for (I i = start_row + 1; i < stop_row; ++i) {
                one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, i, Ap[i] * RC, Ap[i + 1] * RC, sum_private, R, C, RC);
            }

            // Handle the last row of blocks from the task
            i = stop_row;
            one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, i, Ap[i] * RC, stop_index, sum_private, R, C, RC);
        }
    }
}



__global__ void add_sum_to_Y_kernel(
    int32_t n_row,
    double sum[],
    double Yx[],
    double alpha,
    double beta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int P = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < n_row; i += P) {
        Yx[i] = beta * Yx[i] + alpha * sum[i];
    }
}






// Expose this wrapper function for Python
extern "C" void bsr_matvec(
    const I n_brow,
    const I n_bcol,
    const I R,
    const I C,
    const I Ap[],
    const I Aj[],
    const T Ax[],
    const T Xx[],
          T Yx[],
    const T alpha,
    const T beta,
    const int threadsPerBlock,
    const int blocksPerGrid,
    const I chunks_cnt,
    const npy_intp chunk_index[],
    const I chunk_row[]
) {
    assert(R > 0 && C > 0);

    const I n_row = n_brow * R;
    // const I n_col = n_bcol * C;

    T *sum; // Device pointer for sum
    cudaMalloc(&sum, n_row * sizeof(T));
    cudaMemset(sum, 0, n_row * sizeof(T)); // Initialize to zero

    const npy_intp RC = (npy_intp)R * C;

    // Do the actual multiplication
    bsr_matvec_kernel<<<blocksPerGrid, threadsPerBlock, R * sizeof(T)>>>(n_brow, Ap, Aj, Ax, Xx, sum, R, C, RC, chunk_index, chunk_row, chunks_cnt);
    cudaDeviceSynchronize();

    // Yx[:] = beta * Yx[:] + alpha * sum[:]
    add_sum_to_Y_kernel<<<blocksPerGrid, threadsPerBlock>>>(n_row, sum, Yx, alpha, beta);
    cudaDeviceSynchronize();

    cudaFree(sum);
}

#undef I
#undef T
#undef npy_intp
