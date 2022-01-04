#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include "TensorCoreGemm.cuh"
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define CHUNK_K 4

using namespace nvcuda;
namespace MNN {
namespace CUDA {

__global__ void GemmPrearrange(const MatMulParam* param,
        const float* A,
        __half* AP,
        const float* B,
        __half* BP
        ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int e = param->elh[0];
    int l = param->elh[1];
    int h = param->elh[2];
    int lIndex = i % l;
    int oIndex = i / l;
    int lU = lIndex / 16;
    int lR = lIndex % 16;
    int eU = oIndex / 16;
    int eR = oIndex % 16;

    if (i < e * l) {
        float value = A[oIndex * param->aStride[0] + lIndex * param->aStride[1]];
        __half* dst = AP + eU * param->elhPack[1] * 16 * 16 + lU * 16 * 16 + lR + eR * 16;
        dst[0] = value;
    }
    if (i < h * l) {
        float value = B[oIndex * param->bStride[2] + lIndex * param->bStride[1]];
        int hU = eU;
        int hR = eR;
        __half* dst = BP + hU * param->elhPack[1] * 16 * 16 + lU * 16 * 16 + lR + hR * 16;
        dst[0] = value;
    }
}

void GemmPrepareRerange(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, const float* A, __half* AP, const float* B, __half* BP) {
    int maxCount = std::max(cpuParam->elh[0] * cpuParam->elh[1], cpuParam->elh[1] * cpuParam->elh[2]);
    int block_num = runtime->blocks_num(maxCount);
    int threads_num = runtime->threads_num();
    if (nullptr != AP) {
        runtime->memset(AP, 0, cpuParam->elhPack[0] * cpuParam->elhPack[1] * 256 * sizeof(__half));
    }
    if (nullptr != BP) {
        runtime->memset(BP, 0, cpuParam->elhPack[2] * cpuParam->elhPack[1] * 256 * sizeof(__half));
    }
    GemmPrearrange<<<block_num, threads_num>>>(param, A, AP, B, BP);
}

__global__ void GemmPacked(const MatMulParam* param, float *c, const half *a, const half *b, const float* biasPtr) {
    int eU = param->elhPack[0];
    int lU = param->elhPack[1];
    int hU = param->elhPack[2];
    int maxCount = eU * hU * warpSize;
    extern __shared__ float sharedMemory[];
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int subIndex = index / warpSize;
        int wrapId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;
        int warpM = subIndex % eU;
        int warpN = subIndex / eU;
        float* cache = sharedMemory + wrapId * 16 * 16;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        const half* aStart = a + warpM * lU * 16 * 16;
        const half* bStart = b + warpN * lU * 16 * 16;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (int i = 0; i < lU; ++i) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, aStart + i * 256, 16);
            wmma::load_matrix_sync(b_frag, bStart + i * 256, 16);
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(cache, acc_frag, 16, wmma::mem_row_major);
        //wmma::store_matrix_sync(c + warpM * 16 * param->elh[2] + 16 * warpN, acc_frag, param->elh[2], wmma::mem_row_major);
        int eSta = warpM * 16;
        int eEnd = min(eSta + 16, param->elh[0]);
        int hSta = warpN * 16;
        int hEnd = min(hSta + 16, param->elh[2]);
        int eC = eEnd - eSta;
        int hC = hEnd - hSta;
        float* dstStart = c + hSta * param->cStride[2];
        if (nullptr != biasPtr) {
            for (int tId = laneId; tId < eC * hC; tId += warpSize) {
                int y = tId % eC;
                int x = tId / eC;
                int ye = y + eSta;
                int yi = ye % param->split[2];
                int yc = ye / param->split[2];
                dstStart[yc * param->cStride[0] + yi * param->cStride[1] + x * param->cStride[2]] = min(max(cache[16 * y + x] + biasPtr[hSta + x], param->minValue), param->maxValue);
            }
        } else {
            for (int tId = laneId; tId < eC * hC; tId += warpSize) {
                int y = tId % eC;
                int x = tId / eC;
                int ye = y + eSta;
                int yi = ye % param->split[2];
                int yc = ye / param->split[2];
                dstStart[yc * param->cStride[0] + yi * param->cStride[1] + x * param->cStride[2]] = min(max(cache[16 * y + x], param->minValue), param->maxValue);
            }
        }
    }
}

void GemmPackedMain(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, float *c, const half *a, const half *b, const float* biasPtr) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int cores = prop.multiProcessorCount;
    int sharedMemorySize = 16 * 16 * sizeof(float) * threads_num / prop.warpSize;
    cudaFuncSetAttribute(GemmPacked, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    GemmPacked<<<cores, threads_num, sharedMemorySize>>>(param, c, a, b, biasPtr);
}

}
}
