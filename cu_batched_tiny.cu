#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

typedef int32_t I;
typedef double T;
typedef int64_t npy_intp;



#define GROUP_SIZE 32
#define OFFSET_START (GROUP_SIZE >> 1)
#define CHECK_MASTER (GROUP_SIZE - 1)
#define MASTER ((threadIdx.x & CHECK_MASTER) == 0)
// (16 / sizeof(double)) DO NOT MODIFY
#define ROWS_AT_ONCE 8


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
    // Initialize initial sum section with zeros
    // Initialize the sum segment for the current row to zero

    __syncthreads();
    for (I bi = threadIdx.x; bi < R; bi += blockDim.x) {
        sum_private[bi] = 0;
    }
    __syncthreads();

    // const npy_intp tiled_RC = C;

    const npy_intp tiled_start = (start_index / RC) * C;
    const npy_intp tiled_stop = (stop_index / RC) * C;

    const unsigned step = blockDim.x;

    // I pre_R = R - R % ROWS_AT_ONCE;

    I offset = 0;
    I pre_R = 0;
#pragma unroll
for (int rows_at_once = ROWS_AT_ONCE; rows_at_once > 0; rows_at_once -= 1)
{
    if (R - offset < rows_at_once)
        continue;
    pre_R = R - R % rows_at_once;


    for (I i_b = offset; i_b < pre_R; i_b += rows_at_once) {
        T acc[ROWS_AT_ONCE] = {0};
        const I row_start_offset = i_b * C;

        for (npy_intp index = tiled_start + threadIdx.x; index < tiled_stop; index += step) {
            // Compute the index of the block to which the element belongs
            const I jj = index / C;
            // Compute the index of the element within its block
            const I j_b = index % C;

            // Store the column index of the block
            const I j = __ldg(&Aj[jj]);
            const I j_final = C * j + j_b;

            const npy_intp new_index = jj * RC + row_start_offset + j_b;
            const double X_factor = Xx[j_final];

            // HERE
            #pragma unroll
            for (int acc_i = 0; acc_i < rows_at_once; ++acc_i) {
                acc[acc_i] += Ax[new_index + C * acc_i] * X_factor;
            }
        }

        #pragma unroll
        for (int acc_i = 0; acc_i < rows_at_once; ++acc_i) {
            #pragma unroll
            for (int offset = OFFSET_START; offset != 0; offset = offset >> 1)
                acc[acc_i] += __shfl_down_sync(0xffffffff, acc[acc_i], offset);
            if (MASTER)
                atomicAdd(&sum_private[i_b + acc_i], acc[acc_i]);
        }
    }
    if (pre_R > offset)
        offset = pre_R;
}

    __syncthreads();

    // Add the accumulated sum to the final vector with atomic operations
    T *const y = sum + (npy_intp)R * i;
    for (I bi = threadIdx.x; bi < R; bi += blockDim.x) {
        atomicAdd(&y[bi], sum_private[bi] * alpha);
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

    scale_y<<<blocksPerGrid, threadsPerBlock>>>(n_row, Yx, beta);
    bsr_matvec_kernel<<<blocksPerGrid, threadsPerBlock, R * sizeof(T)>>>(n_brow, Ap, Aj, Ax, Xx, Yx, R, C, RC, chunk_index, chunk_row, chunks_cnt, alpha);
}

#undef I
#undef T
#undef npy_intp
