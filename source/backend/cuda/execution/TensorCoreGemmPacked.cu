
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include "TensorCoreGemm.cuh"

using namespace nvcuda;
namespace MNN {
namespace CUDA {

template<typename T>
__global__ void GemmPackedFull(const MatMulParam* param, const int iBlock, T *c, const half *a, const half *b, const T* biasPtr) {
    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t maxCount = eU * hU * warpSize;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];

    T* cache = (T*)(sharedMemory + wrapId * 16 * 16);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, T> acc_frag;
    
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        size_t subIndex = index / warpSize;
        size_t warpM = subIndex % eU;
        size_t warpN = subIndex / eU;

        wmma::load_matrix_sync(acc_frag, biasPtr + 16 * warpN, 0, wmma::mem_row_major);
        const half* aStart = a + warpM * lU * 16 * 16;
        const half* bStart = b + warpN * lU * 16 * 16;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            half* aTemp = ((half *)(aStart+i*256));//aStart + (i << 8) + (laneId << 1);
            half* bTemp = ((half *)(bStart+i*256));//bStart + (i << 8) + (laneId << 1);

            wmma::load_matrix_sync(a_frag, aStart + i * 256, 16);
            wmma::load_matrix_sync(b_frag, bStart + i * 256, 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        for(size_t t=0; t<acc_frag.num_elements; t++){
            acc_frag.x[t] = max(acc_frag.x[t], param->minValue);
            acc_frag.x[t] = min(acc_frag.x[t], param->maxValue);
        }

        size_t eSta = (warpM + iBlock*eU) * 16;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);

        size_t eC = eEnd - eSta;
        T* dstStart = (T*)(c + warpN * 16 * (size_t)param->elh[0] + eSta * 16);
        wmma::store_matrix_sync(cache, acc_frag, 16, wmma::mem_row_major);

        if (warpSize % 16 == 0) {
            if(sizeof(T) == 4) {
                size_t r = warpSize / 16;
                size_t x = laneId / r;
                size_t ysta = laneId % r;
                for (size_t y = ysta; y < eC; y+=r) {
                    float value = *((T*)(cache + 16 * y + x));
                    dstStart[y * 16 + x] = value;
                }
            } else {
                size_t xsta = (laneId % 8) * 2;
                size_t ysta = laneId / 8;
                for (size_t y = ysta; y < eC; y+=4) {
                    dstStart[y * 16 + xsta]     = *((T*)(cache + 16 * y + xsta));
                    dstStart[y * 16 + xsta + 1] = *((T*)(cache + 16 * y + xsta + 1));
                }
            }
        } else {
            for (size_t tId = laneId; tId < eC * 16; tId += warpSize) {
                size_t y = tId % eC;
                size_t x = tId / eC;
                float value = *((T*)(cache + 16 * y + x));
                dstStart[y * 16 + x] = value;
            }
        }
    }
}

template<typename T>
__global__ void GemmPackedFull16x32(const MatMulParam* param, const int iBlock, T *c, const half *a, const half *b, const T* biasPtr) {
    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t threadCount = blockDim.x / warpSize;
    size_t maxCount = eU * hU;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];
    T* cache = (T*)(sharedMemory + wrapId * 16 * 32);
    for (size_t index = blockIdx.x * threadCount + wrapId; index < maxCount; index += gridDim.x * threadCount) {
        size_t warpM = index % eU;
        size_t warpN = index / eU;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            MA0;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            MB0;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            MB1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC1;

        wmma::load_matrix_sync(MC0, biasPtr + 32 * warpN + 0, 0, wmma::mem_row_major);
        wmma::load_matrix_sync(MC1, biasPtr + 32 * warpN + 16, 0, wmma::mem_row_major);
        const half* aStart = a + warpM * lU * 16 * 16;
        const half* bStart = b + warpN * lU * 16 * 32;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            wmma::load_matrix_sync(MA0, aStart + i * 256 + 0, 16);
            wmma::load_matrix_sync(MB0, bStart + i * 512, 16);
            wmma::load_matrix_sync(MB1, bStart + i * 512 + 256, 16);
            wmma::mma_sync(MC0, MA0, MB0, MC0);
            wmma::mma_sync(MC1, MA0, MB1, MC1);
        }
        for(size_t t=0; t<MC0.num_elements; t++){
            MC0.x[t] = max(MC0.x[t], param->minValue);
            MC0.x[t] = min(MC0.x[t], param->maxValue);
        }
        for(size_t t=0; t<MC1.num_elements; t++){
            MC1.x[t] = max(MC1.x[t], param->minValue);
            MC1.x[t] = min(MC1.x[t], param->maxValue);
        }
        size_t eSta = (warpM + iBlock*eU) * 16;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);
        size_t eC = eEnd - eSta;
        T* dst0 = (T*)(c + warpN * 32 * (size_t)param->elh[0] + eSta * 16);
        T* dst1 = (T*)(c + (warpN * 32 + 16) * (size_t)param->elh[0] + eSta * 16);
        // First 8x32
        wmma::store_matrix_sync(cache, MC0, 16, wmma::mem_row_major);
        // Second 8x32
        wmma::store_matrix_sync(cache + 256, MC1, 16, wmma::mem_row_major);
        auto dst = dst0;
        auto src = cache;
        if (laneId >= 16) {
            dst = dst1;
            src = cache + 256;
        }
        size_t x = laneId % 16;
        for (size_t y = 0; y < eC; ++y) {
            dst[y * 16 + x] = src[y * 16 + x];
        }
    }
}

template<typename T>
__global__ void GemmPackedFull32x16(const MatMulParam* param, const int iBlock, T *c, const half *a, const half *b, const T* biasPtr) {
    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t threadCount = blockDim.x / warpSize;
    size_t maxCount = eU * hU;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];
    T* cache = (T*)(sharedMemory + wrapId * 32 * 16);
    for (size_t index = blockIdx.x * threadCount + wrapId; index < maxCount; index += gridDim.x * threadCount) {
        size_t warpN = index % hU;
        size_t warpM = index / hU;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            MA0;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            MA1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            MB0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC1;

        wmma::load_matrix_sync(MC0, biasPtr + 16 * warpN + 0, 0, wmma::mem_row_major);
        for(size_t t=0; t<MC0.num_elements; t++){
            MC1.x[t] = MC0.x[t];
        }

        const half* aStart = a + warpM * lU * 32 * 16;
        const half* bStart = b + warpN * lU * 16 * 16;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            wmma::load_matrix_sync(MA0, aStart + i * 512 + 0, 16);
            wmma::load_matrix_sync(MA1, aStart + i * 512 + 256, 16);
            wmma::load_matrix_sync(MB0, bStart + i * 256 + 0, 16);
            wmma::mma_sync(MC0, MA0, MB0, MC0);
            wmma::mma_sync(MC1, MA1, MB0, MC1);
        }
        for(size_t t=0; t<MC0.num_elements; t++){
            MC0.x[t] = max(MC0.x[t], param->minValue);
            MC0.x[t] = min(MC0.x[t], param->maxValue);
        }
        for(size_t t=0; t<MC1.num_elements; t++){
            MC1.x[t] = max(MC1.x[t], param->minValue);
            MC1.x[t] = min(MC1.x[t], param->maxValue);
        }
        size_t eSta = (warpM + iBlock*eU) * 32;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);
        size_t eC = eEnd - eSta;
        T* dst0 = (T*)(c + warpN * 16 * (size_t)param->elh[0] + eSta * 16);
        T* dst1 = (T*)(dst0 + 256);
        // First 8x32
        wmma::store_matrix_sync(cache, MC0, 16, wmma::mem_row_major);
        // Second 8x32
        wmma::store_matrix_sync(cache + 256, MC1, 16, wmma::mem_row_major);
        auto dst = dst0;
        auto src = cache;
        if (laneId >= 16) {
            dst = dst1;
            src = cache + 256;
        }
        size_t x = laneId % 16;
        for (size_t y = 0; y < eC; ++y) {
            dst[y * 16 + x] = src[y * 16 + x];
        }
    }
}

void GemmPackedFullMain(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const half* biasPtr, int bytes, int iBlock) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    // MNN_PRINT("%d: %d - %d - %d  - %d\n", iBlock, cpuParam->elhPack[0], cpuParam->elhPack[1], cpuParam->elhPack[2], cpuParam->elh[0]);
    {
        int maxThreadInWarp = UP_DIV(cpuParam->elhPack[0] * cpuParam->elhPack[2], cores);
        int threads_num = std::min(prop.maxThreadsPerBlock, maxThreadInWarp * prop.warpSize);
        int basicMemory = 16 * 16 * sizeof(float) * prop.maxThreadsPerBlock / prop.warpSize;
        if (4 == bytes) {
            cudaFuncSetAttribute(GemmPackedFull<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, prop.sharedMemPerMultiprocessor);
            GemmPackedFull<<<cores, threads_num, basicMemory>>>(param, iBlock, (float*)c, a, b, (float*)biasPtr);
            checkKernelErrors;
        } else {
            //MNN_PRINT("%d - %d, %d- %d\n", cpuParam->elhPack[0], cpuParam->elhPack[2], cpuParam->elh[0], cpuParam->elh[2]);
            cudaFuncSetAttribute(GemmPackedFull<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, prop.sharedMemPerMultiprocessor);
            GemmPackedFull<<<cores, threads_num, basicMemory>>>(param, iBlock, (half*)c, a, b, (half*)biasPtr);
            checkKernelErrors;
        }
    }
}


void GemmPacked16x32(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const half* biasPtr, int bytes, int iBlock) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    // MNN_PRINT("%d - %d - %d\n", cpuParam->elhPack[0], cpuParam->elhPack[1], cpuParam->elhPack[2]);
    {
        int hUP = cpuParam->elhPack[2];
        int maxThreadInWarp = UP_DIV(cpuParam->elhPack[0] * hUP, cores);
        int threads_num = ALIMIN(512, maxThreadInWarp * prop.warpSize);
        //MNN_PRINT("GemmPacked16x32：%d-%d-%d-%d-%d\n\n", hUP, cpuParam->elhPack[0], cpuParam->elhPack[2], cpuParam->elhPack[0]*cpuParam->elhPack[2], threads_num);
        threads_num = ALIMIN(prop.maxThreadsPerBlock, threads_num);
        int basicMemory = 32 * 16 * sizeof(float) * (threads_num / prop.warpSize);
        if (4 == bytes) {
            cudaFuncSetAttribute(GemmPackedFull16x32<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, basicMemory);
            GemmPackedFull16x32<<<cores, threads_num, basicMemory>>>(param, iBlock, (float*)c, a, b, (float*)biasPtr);
            checkKernelErrors;
        } else {
            cudaFuncSetAttribute(GemmPackedFull16x32<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, basicMemory);
            GemmPackedFull16x32<<<cores, threads_num, basicMemory>>>(param, iBlock, (half*)c, a, b, (half*)biasPtr);
            checkKernelErrors;
        }
    }
}

void GemmPacked32x16(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const half* biasPtr, int bytes, int iBlock) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    // MNN_PRINT("%d - %d - %d\n", cpuParam->elhPack[0], cpuParam->elhPack[1], cpuParam->elhPack[2]);
    {
        int eUP = cpuParam->elhPack[0];
        int maxThreadInWarp = UP_DIV(eUP * cpuParam->elhPack[2], cores);
        int threads_num = ALIMIN(512, maxThreadInWarp * prop.warpSize);
        //MNN_PRINT("GemmPacked32x16：%d-%d-%d-%d-%d\n\n", eUP, cpuParam->elhPack[0], cpuParam->elhPack[2], cpuParam->elhPack[0]*cpuParam->elhPack[2], threads_num);
        threads_num = ALIMIN(prop.maxThreadsPerBlock, threads_num);
        int basicMemory = 32 * 16 * sizeof(float) * (threads_num / prop.warpSize);
        if (4 == bytes) {
            cudaFuncSetAttribute(GemmPackedFull32x16<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, basicMemory);
            GemmPackedFull32x16<<<cores, threads_num, basicMemory>>>(param, iBlock, (float*)c, a, b, (float*)biasPtr);
            checkKernelErrors;
        } else {
            cudaFuncSetAttribute(GemmPackedFull32x16<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, basicMemory);
            GemmPackedFull32x16<<<cores, threads_num, basicMemory>>>(param, iBlock, (half*)c, a, b, (half*)biasPtr);
            checkKernelErrors;
        }
    }
}

}
}