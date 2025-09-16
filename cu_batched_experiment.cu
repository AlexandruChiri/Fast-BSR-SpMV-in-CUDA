#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

typedef int32_t I;
typedef double T;
typedef int64_t npy_intp;


#define LANE_ID (threadIdx.x & 31)
#define WARP_MASTER (LANE_ID == 0)
#define WARP_ID (threadIdx.x >> 5)


__device__ T block_reduce(T val) {
    __shared__ T aux[32];

    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        aux[i] = 0;
    __syncthreads();

    #pragma unroll
    for (int offset = 16; offset != 0; offset = offset >> 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (WARP_MASTER)
        aux[WARP_ID] = val;

    __syncthreads();

    if (WARP_ID == 0) {
        val = aux[LANE_ID];
        #pragma unroll
        for (int offset = 16; offset != 0; offset = offset >> 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}



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
    const npy_intp RC,
    const T alpha
) {
    __syncthreads();
    // Initialize `sum_private`
    for (I bi = threadIdx.x; bi < R; bi += blockDim.x) {
        sum_private[bi] = 0;
    }
    __syncthreads();

    T acc[5] = {0};

    for (npy_intp index = start_index + threadIdx.x; index < stop_index; index += blockDim.x) {
        // The block index
        const I jj = index / RC;
        // The intra-block index
        const I index_b = index % RC;
        // The block column
        const I j = Aj[jj];
        // The intra-block row and column indices
        const I i_b = index_b / C;
        const I j_b = index_b % C;
        // The column coordinate within the matrix
        const I j_final = C * j + j_b;
        // Atomicity required
        acc[i_b] += Ax[index] * Xx[j_final];
    }

    __syncthreads();

    // Add the accumulated sum to the final vector with atomic operations
    T *const y = sum + (npy_intp)R * i;
    for (I bi = 0; bi < R; ++bi) {
        const T val = block_reduce(acc[bi]);
        if (threadIdx.x == 0)
            atomicAdd(&y[bi], val * alpha);
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
    const I chunks_cnt,
    const T alpha)
{
    // Block id
    const int bid = blockIdx.x;
    // Block count
    const int bcount = gridDim.x;

    // Array of block-private partial results in shared memory
    extern __shared__ T sum_private[];



    for (I chunk = bid; chunk < chunks_cnt; chunk += bcount) {
        const npy_intp start_index = chunk_index[chunk];
        const npy_intp stop_index = chunk_index[chunk + 1];

        const I start_row = chunk_row[chunk];
        const I stop_row = chunk_row[chunk + 1];

        if (start_row == stop_row) {
            one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, start_row, start_index, stop_index, sum_private, R, C, RC, alpha);
        }
        else {
            // Handle the first row of blocks from the task
            I i = start_row;
            one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, i, start_index, Ap[i + 1] * RC, sum_private, R, C, RC, alpha);

            // Handle the complete rows from between the `start_row` and `stop_row` ones
            for (I i = start_row + 1; i < stop_row; ++i) {
                one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, i, Ap[i] * RC, Ap[i + 1] * RC, sum_private, R, C, RC, alpha);
            }

            // Handle the last row of blocks from the task
            i = stop_row;
            one_row_of_blocks(Ap, Aj, Ax, Xx, Yx, i, Ap[i] * RC, stop_index, sum_private, R, C, RC, alpha);
        }
    }
}



__global__ void scale_y(
    int32_t n_row,
    T Yx[],
    T beta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int P = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < n_row; i += P) {
        Yx[i] *= beta;
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
    const I n_row = n_brow * R;
    const npy_intp RC = (npy_intp)R * C;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, blocksPerGrid * threadsPerBlock * R * sizeof(T));

    scale_y<<<blocksPerGrid, threadsPerBlock>>>(n_row, Yx, beta);
    bsr_matvec_kernel<<<blocksPerGrid, threadsPerBlock, R * sizeof(T)>>>(n_brow, Ap, Aj, Ax, Xx, Yx, R, C, RC, chunk_index, chunk_row, chunks_cnt, alpha);
}

#undef I
#undef T
#undef npy_intp
