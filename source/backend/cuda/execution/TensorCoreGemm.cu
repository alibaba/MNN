#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include "TensorCoreGemm.cuh"
#include "MNNCUDAFunction.cuh"
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define CHUNK_L 4
#define CHUNK_E 4
#define CHUNK_H 4
#define PACK_NUMBER 16
#define PACK_NUMBER_C2 (PACK_NUMBER/2)

using namespace nvcuda;
namespace MNN {
namespace CUDA {

template<typename T>
__global__ void GemmPrearrange(MatMulParam paramV,
        const T* OA,
        __half* OAP,
        const T* OB,
        __half* OBP,
        DivModFast lA
        ) {
    int b = blockIdx.x;
    auto param = &paramV;
    int lAlign = param->elhPack[1] * 16;
    int eAlign = param->elhPack[0] * 16;
    int hAlign = param->elhPack[2] * 16;
    __half* BP = OBP + b * param->elhPack[1] * param->elhPack[2] * 16 * 16;
    __half* AP = OAP + b * param->elhPack[1] * param->elhPack[0] * 16 * 16;
    const T* A = OA + b * param->elh[0] * param->elh[1];
    const T* B = OB + b * param->elh[2] * param->elh[1];
    int mc = param->elhPack[0] * param->elhPack[1] * 256;
    int e = param->elh[0];
    int l = param->elh[1];
    int h = param->elh[2];
    for (size_t index = threadIdx.x; index < mc && OA != nullptr; index += blockDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);

        half value = 0.0;
        if (oIndex < e && lIndex < l) {
            value = A[oIndex * param->aStride[0] + lIndex * param->aStride[1]];
        }
        AP[index] = value;
    }
    mc = param->elhPack[2] * param->elhPack[1] * 256;
    for (size_t index = threadIdx.x; index < mc && OB != nullptr; index += blockDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);
        half value = 0.0;
        if (oIndex < h && lIndex < l) {
            value = B[oIndex * param->bStride[2] + lIndex * param->bStride[1]];
        }
        BP[index] = value;
    }
}

template<typename T>
__global__ void GemmPrearrange_OPT(MatMulParam paramV, const int maxCount,
        const int AreaPackA, const int AreaPackB, const int AreaA, const int AreaB,
        const T* OA,
        __half* OAP,
        const T* OB,
        __half* OBP,
        DivModFast lA,
        DivModFast pM
        ) {
    int index, b;
    size_t indexT = blockIdx.x*blockDim.x+threadIdx.x;
    pM.divmod(indexT, b, index);
    int indexCopy = index;
    
    auto param = &paramV;
    int e = param->elh[0];
    int l = param->elh[1];
    int h = param->elh[2];
    for (; index < AreaPackA && OA != nullptr; index += blockDim.x*gridDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);

        __half* AP = OAP + b * AreaPackA;
        const T* A = OA + b * AreaA;
        half value = 0.0;
        if (oIndex < e && lIndex < l) {
            value = A[oIndex * param->aStride[0] + lIndex * param->aStride[1]];
        }
        AP[index] = value;
    }

    index = indexCopy;
    for (; index < AreaPackB && OB != nullptr; index += blockDim.x*gridDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);
        
        __half* BP = OBP + b * AreaPackB;
        const T* B = OB + b * AreaB;
        half value = 0.0;
        if (oIndex < h && lIndex < l) {
            value = B[oIndex * param->bStride[2] + lIndex * param->bStride[1]];
        }
        BP[index] = value;
    }
}

void GemmPrepareRerange(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, const void* A, __half* AP, const void* B, __half* BP, int bytes) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int unit_threads_num = ALIMAX(cpuParam->elhPack[0], cpuParam->elhPack[2]) * cpuParam->elhPack[1] * 256;
    threads_num = ALIMIN(threads_num, unit_threads_num);

    const int AreaPackA = cpuParam->elhPack[0] * cpuParam->elhPack[1] * 256;
    const int AreaPackB = cpuParam->elhPack[1] * cpuParam->elhPack[2] * 256;
    const int AreaA     = cpuParam->elh[0] * cpuParam->elh[1];
    const int AreaB     = cpuParam->elh[1] * cpuParam->elh[2];

    const int maxPack = ALIMAX(AreaPackA, AreaPackB);
    const int maxCount = cpuParam->batch * maxPack;
    DivModFast pM(maxPack);
    int block_num = runtime->blocks_num(maxCount);
    int block_size = runtime->threads_num();
    DivModFast lA(cpuParam->elhPack[1] * 16);
    if (bytes == 4) {
        //GemmPrearrange<<<cpuParam->batch, threads_num>>>(*cpuParam, (float*)A, AP, (float*)B, BP, lA);
        GemmPrearrange_OPT<<<block_num, block_size>>>(*cpuParam, maxCount, AreaPackA, AreaPackB,  AreaA, AreaB, (float*)A, AP, (float*)B, BP, lA, pM);
        checkKernelErrors;
    } else {
        MNN_ASSERT(bytes == 2);
        //GemmPrearrange<<<cpuParam->batch, threads_num>>>(*cpuParam, (half*)A, AP, (half*)B, BP, lA);
        GemmPrearrange_OPT<<<block_num, block_size>>>(*cpuParam, maxCount, AreaPackA, AreaPackB,  AreaA, AreaB, (half*)A, AP, (half*)B, BP, lA, pM);
        checkKernelErrors;
    }
}

template<typename T, typename LayoutA, typename LayoutB>
__global__ void GemmPacked(const MatMulParam* param, T *bc, const half *ba, const half *bb, const T* biasPtr) {
    int eU = param->elhPack[0];
    int lU = param->elhPack[1];
    int hU = param->elhPack[2];
    int maxCount = eU * hU * warpSize * param->batch;
    extern __shared__ float sharedMemory[];
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int oIndex = index / warpSize;
        int subIndex = oIndex % (eU * hU);
        int bIndex = oIndex / (eU * hU);
        int wrapId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;
        int warpM = subIndex % eU;
        int warpN = subIndex / eU;
        T* c = bc + bIndex * param->elh[0] * param->elh[2];
        const half* a = ba + bIndex * param->elhPack[1] * param->elhPack[0] * 16 * 16;
        const half* b = bb + bIndex * param->elhPack[1] * param->elhPack[2] * 16 * 16;
        float* cache = sharedMemory + wrapId * 16 * 16;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, LayoutA>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, LayoutB>
            b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        const half* aStart = a + warpM * param->aPStride[0];
        const half* bStart = b + warpN * param->bPStride[0];
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (int i = 0; i < lU; ++i) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, aStart + i * param->aPStride[1], param->aPStride[2]);
            wmma::load_matrix_sync(b_frag, bStart + i * param->bPStride[1], param->bPStride[2]);
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(cache, acc_frag, 16, wmma::mem_row_major);
        int eSta = warpM * 16;
        int eEnd = min(eSta + 16, param->elh[0]);
        int hSta = warpN * 16;
        int hEnd = min(hSta + 16, param->elh[2]);
        int eC = eEnd - eSta;
        int hC = hEnd - hSta;
        T* dstStart = c + hSta * param->cStride[2];
        if (nullptr != biasPtr) {
            for (int tId = laneId; tId < eC * hC; tId += warpSize) {
                int y = tId % eC;
                int x = tId / eC;
                int ye = y + eSta;
                float value = cache[16 * y + x];
                float biasValue = biasPtr[hSta + x];
                dstStart[ye * param->cStride[0] + x * param->cStride[2]] = value + biasValue;
            }
        } else {
            for (int tId = laneId; tId < eC * hC; tId += warpSize) {
                int y = tId % eC;
                int x = tId / eC;
                int ye = y + eSta;
                float value = cache[16 * y + x];
                dstStart[ye * param->cStride[0] + x * param->cStride[2]] = value;
            }
        }
    }
}

void GemmPackedMain(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const void* biasPtr, int bytes, bool transposeA, bool transposeB) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int cores = prop.multiProcessorCount;
    int sharedMemorySize = 16 * 16 * sizeof(float) * threads_num / prop.warpSize;
    if (bytes == 4) {
        if (transposeA) {
            if (transposeB) {
                GemmPacked<float, wmma::col_major, wmma::row_major><<<cores, threads_num, sharedMemorySize>>>(param, (float*)c, a, b, (float*)biasPtr);
            } else {
                GemmPacked<float, wmma::col_major, wmma::col_major><<<cores, threads_num, sharedMemorySize>>>(param, (float*)c, a, b, (float*)biasPtr);
            }
        } else {
            if (transposeB) {
                GemmPacked<float, wmma::row_major, wmma::row_major><<<cores, threads_num, sharedMemorySize>>>(param, (float*)c, a, b, (float*)biasPtr);
            } else {
                GemmPacked<float, wmma::row_major, wmma::col_major><<<cores, threads_num, sharedMemorySize>>>(param, (float*)c, a, b, (float*)biasPtr);
            }
        }
    } else {
        if (transposeA) {
            if (transposeB) {
                GemmPacked<half, wmma::col_major, wmma::row_major><<<cores, threads_num, sharedMemorySize>>>(param, (half*)c, a, b, (half*)biasPtr);
            } else {
                GemmPacked<half, wmma::col_major, wmma::col_major><<<cores, threads_num, sharedMemorySize>>>(param, (half*)c, a, b, (half*)biasPtr);
            }
        } else {
            if (transposeB) {
                GemmPacked<half, wmma::row_major, wmma::row_major><<<cores, threads_num, sharedMemorySize>>>(param, (half*)c, a, b, (half*)biasPtr);
            } else {
                GemmPacked<half, wmma::row_major, wmma::col_major><<<cores, threads_num, sharedMemorySize>>>(param, (half*)c, a, b, (half*)biasPtr);
            }
        }
    }
    checkKernelErrors;
}
}
}
