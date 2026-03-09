#include "TopKV2Execution.hpp"
#include <memory>

namespace MNN {
namespace CUDA {

// Sift Down
template<typename indexT, typename valueT>
__device__ inline void siftDown(const int K, const int descendFlag, valueT * valuesThread, indexT * indicesThread) {
    int parent = 0;
    while (true) {
        int child = 2 * parent + 1;
        if (child >= K) break;
        if (child + 1 < K && (valueT)(descendFlag) * valuesThread[child + 1] < (valueT)(descendFlag) * valuesThread[child]) {
            child++;
        }
        if ((valueT)(descendFlag) * valuesThread[parent] > (valueT)(descendFlag) * valuesThread[child]) {
            valueT tmpV = valuesThread[parent];
            valuesThread[parent] = valuesThread[child];
            valuesThread[child] = tmpV;

            indexT tmpI = indicesThread[parent];
            indicesThread[parent] = indicesThread[child];
            indicesThread[child] = tmpI;

            parent = child;
        } else {
            break;
        }
    }
}


// rank TopK in the corresponding thead
template<typename indexT, typename valueT>
__device__ void TopKInThread(const valueT * inputDevice, indexT * indicesThread, valueT * valuesThread, const int K, const int numElePerRow, const valueT minValue, const int descendFlag) {
    for (int i = 0 ; i < K; i++) {
        indicesThread[i] = -1;
        valuesThread[i] = (valueT)(descendFlag) * minValue;
    }

    int idxFirstEleInRow = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if we can use vectorized load (address alignment and size)
    // We assume inputDevice is aligned to at least 4 bytes. 
    // Note: Since we use grid-stride loop, the data accessed by a single thread is NOT contiguous.
    // So we cannot achieve true vectorized load (LD.E.128) which requires contiguous memory.
    // However, manual unrolling helps Instruction-Level Parallelism (ILP) and Latency Hiding.

    // Main loop with unrolling (Process 4 elements per step if possible)
    indexT i = idxFirstEleInRow;
    /* 
       Note: The data[0..3] are separated by gridDim.x * blockDim.x, so they are not contiguous.
       We unroll the loop to issue multiple independent memory loads, allowing the GPU 
       to hide memory latency and improve ILP.
    */
    for (; i + gridDim.x * blockDim.x * 3 < numElePerRow; i += gridDim.x * blockDim.x * 4) {
        valueT data[4];
        data[0] = inputDevice[i];
        data[1] = inputDevice[i + gridDim.x * blockDim.x];
        data[2] = inputDevice[i + gridDim.x * blockDim.x * 2];
        data[3] = inputDevice[i + gridDim.x * blockDim.x * 3];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            valueT val = data[k];
            // Compare with the root of the heap
            if ((valueT)(descendFlag) * val > (valueT)(descendFlag) * valuesThread[0]) {
                valuesThread[0] = val;
                indicesThread[0] = i + gridDim.x * blockDim.x * k;

                siftDown(K, descendFlag, valuesThread, indicesThread);
            }
        }
    }

    // Handle remaining elements
    for (; i < numElePerRow; i += gridDim.x * blockDim.x) {
        valueT data = inputDevice[i];

        // Compare with the root of the heap (which is the minimum of the current Top K)
        // If data is larger than the minimum, it might be in the Top K.
        if ((valueT)(descendFlag) * data > (valueT)(descendFlag) * valuesThread[0]) {
            // Replace root
            valuesThread[0] = data;
            indicesThread[0] = i;

            siftDown(K, descendFlag, valuesThread, indicesThread);
        }
    }

    // Sort the Heap to produce a sorted array (Descending)
    // We pop elements from the Min-Heap one by one.
    // The popped element is the minimum of the remaining heap.
    // We place it at the end of the array (filling from K-1 down to 0).
    // This results in a descending sorted array: [Max, ..., Min]
    for (int i = K - 1; i > 0; i--) {
        // Move current root (min) to the end (i)
        valueT tmpV = valuesThread[0];
        valuesThread[0] = valuesThread[i];
        valuesThread[i] = tmpV;

        indexT tmpI = indicesThread[0];
        indicesThread[0] = indicesThread[i];
        indicesThread[i] = tmpI;

        // Restore heap property for the remaining [0, i) elements
        siftDown(i, descendFlag, valuesThread, indicesThread);
    }

    return;
}


// reduce TopK results of two offsets
template<typename indexT, typename valueT>
__device__ void ReduceTopK(indexT * indicesArray, valueT * valuesArray, const int offset1, const int offset2, const int K, const int descendFlag) {
    indexT idx1 = offset1 + K - 1;
    indexT idx2 = offset2 + K - 1;
    indexT idxVirtual = offset1 + 2 * K -1;

    while (idx2 >= offset2) {
        if (idx1 < offset1) {
            while (idxVirtual >= offset1) {
                indicesArray[idxVirtual] = indicesArray[offset2 + (idxVirtual - offset1)];
                valuesArray[idxVirtual] = valuesArray[offset2 + (idxVirtual - offset1)];
                idxVirtual --;
            }
            break;
        }

        if ((valueT)(descendFlag) * valuesArray[idx1] <= (valueT)(descendFlag) * valuesArray[idx2]) {
            if (idxVirtual <= offset1 + K - 1) {
                indicesArray[idxVirtual] = indicesArray[idx1];
                valuesArray[idxVirtual] = valuesArray[idx1];
            }
            idx1 --;
        } else {
            if (idxVirtual <= offset1 + K - 1) {
                indicesArray[idxVirtual] = indicesArray[idx2];
                valuesArray[idxVirtual] = valuesArray[idx2];
            }
            idx2 --;
        }
        idxVirtual --;
    }

    return;
}


// get results of all blocks' TopK in one row
template<typename indexT, typename valueT>
__device__ void TopKOneRow(const valueT * inputDevice, indexT * indicesBlock, valueT * valuesBlock, indexT * tempIndicesDevice, valueT * tempValuesDevice, const int K, const int lengthRow, valueT minValue, const int descendFlag) {
    indexT * indicesThread = indicesBlock + threadIdx.x * K;
    valueT * valuesThread = valuesBlock + threadIdx.x * K;

    // rank TopK
    TopKInThread<indexT, valueT>(inputDevice, indicesThread, valuesThread, K, lengthRow, minValue, descendFlag);

    __syncthreads();

    // reduce
    for(int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            ReduceTopK<indexT, valueT>(indicesBlock, valuesBlock, threadIdx.x * K, (threadIdx.x + stride) * K, K, descendFlag);
        }
        __syncthreads();
    }

    // move data from block's smem to global memory(prepare for the next kernel function)
    if (threadIdx.x == 0) {
        for(int i = 0; i < K; i++) {
            tempIndicesDevice[K * blockIdx.x + i] = indicesBlock[i];
            tempValuesDevice[K * blockIdx.x + i] = valuesBlock[i];
        }
    }

    return;
}


// get results of the final TopK from all block's TopK in a row
template<typename indexT, typename valueT>
__device__ void GetResultOneRow(indexT * outputIndicesDevice, valueT * outputValuesDevice, indexT * tempIndicesDevice, valueT * tempValuesDevice, indexT * finalIndices, valueT * finalValues, const int K, const int reduceLength, const int descendFlag) {
    // move data from global memory to a block's smem
    if (threadIdx.x < reduceLength) {
        for (int i = 0; i < K; i++) {
            finalIndices[threadIdx.x * K + i] = tempIndicesDevice[threadIdx.x * K + i];
            finalValues[threadIdx.x * K + i] = tempValuesDevice[threadIdx.x * K + i];
        }
    }
    __syncthreads();

    // the first round of reducing needs special action
    int stride = blockDim.x >> 1;
    if ((threadIdx.x < stride) && (threadIdx.x + stride < reduceLength)) {
        ReduceTopK<indexT, valueT>(finalIndices, finalValues, threadIdx.x * K, (threadIdx.x + stride) * K, K, descendFlag);
    }
    __syncthreads();
    stride >>= 1;

    // the remaining rounds of reducing
    for (; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ReduceTopK<indexT, valueT>(finalIndices, finalValues, threadIdx.x * K, (threadIdx.x + stride) * K, K, descendFlag);
        }
        __syncthreads();
    }

    //move data from a block's smem to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < K; i++) {
            outputIndicesDevice[i] = finalIndices[i];
            outputValuesDevice[i] = finalValues[i];
        }
    }

    return;
}


// allocate addresses for each row and call <TopKOneRow>
template<typename indexT, typename valueT>
__global__ void TopKAllRows(const valueT * inputDevice, indexT * tempIndicesDevice, valueT * tempValuesDevice, const int K, const int lengthRow, valueT minValue, const int descendFlag) {
    extern __shared__ char smem[];
    indexT * indicesBlock = reinterpret_cast<indexT *>(smem);
    valueT * valuesBlock = reinterpret_cast<valueT *>(&smem[blockDim.x * K * sizeof(indexT)]);

    int idxRow = blockIdx.y;

    const valueT * inputDeviceThisRow = inputDevice + idxRow * lengthRow;
    indexT * tempIndicesDeviceThisRow = tempIndicesDevice + idxRow * gridDim.x * K;
    valueT * tempValuesDeviceThisRow = tempValuesDevice + idxRow * gridDim.x * K;

    TopKOneRow<indexT, valueT>(inputDeviceThisRow, indicesBlock, valuesBlock, tempIndicesDeviceThisRow, tempValuesDeviceThisRow, K, lengthRow, minValue, descendFlag);
    __syncthreads();

    return;
}


// allocate addresses for each row and call <GetResultOneRow>
// This kernel assumes that each row of data corresponds to one block.
template<typename indexT, typename valueT>
__global__ void GetResultAllRows(indexT * outputIndicesDevice, valueT * outputValuesDevice, indexT * tempIndicesDevice, valueT * tempValuesDevice, const int K, const int numBlockPerRow, const int descendFlag) {
    extern __shared__ char smem[];
    indexT * finalIndices = reinterpret_cast<indexT *>(smem);
    valueT * finalValues = reinterpret_cast<valueT *>(&smem[numBlockPerRow * K * sizeof(indexT)]);

    int idxRow = blockIdx.x; // each block corresponds to a row

    indexT * outputIndicesDeviceThisRow = outputIndicesDevice + idxRow * K;
    valueT * outputValuesDeviceThisRow = outputValuesDevice + idxRow * K;
    indexT * tempIndicesDeviceThisRow = tempIndicesDevice + idxRow * numBlockPerRow * K;
    valueT * tempValuesDeviceThisRow = tempValuesDevice + idxRow * numBlockPerRow * K;

    GetResultOneRow<indexT, valueT>(outputIndicesDeviceThisRow, outputValuesDeviceThisRow, tempIndicesDeviceThisRow, tempValuesDeviceThisRow, finalIndices, finalValues, K, numBlockPerRow, descendFlag);

    return;
}


// The inequality "numThreadPerBlock * K * (sizeof(indexT) + sizeof(valueT)) <= smemPerBlock" must be guaranteed, which means numThreadPerBlock depends on K.
template<typename indexT, typename valueT>
int CalculateNumThreadPerBlock(const int K, const int smemPerBlock) {
    int temp = smemPerBlock / (K * (sizeof(indexT) + sizeof(valueT)));
    int numCalculate = std::pow(2, (std::floor(std::log2(temp))));
    int numLimit = 1024;
    return ALIMIN(numLimit, numCalculate);
}


// The inequality "numBlockPerRow * K * (sizeof(indexT) + sizeof(valueT)) <= smemPerBlock" must be guaranteed by restricting numElePerThread.
template<typename indexT, typename valueT>
int CalcualteNumElePerThread(const int K, const int numElePerRow, const int numThreadPerBlock, const int smemPerBlock) {
    int numLimit = K;
    int numCalculate = UP_DIV(numElePerRow, (smemPerBlock / (K * (sizeof(indexT) + sizeof(valueT))))-1);
    return ALIMAX(numLimit,numCalculate);
}


TopKV2Execution::TopKV2Execution(const Op* op, Backend* backend) : Execution(backend) {
    mOp = op;
}


ErrorCode TopKV2Execution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // prepare some params for the kernel function
    Tensor * inputTensor = inputs[0];

    int lengthRow = inputTensor->buffer().dim[inputTensor->buffer().dimensions - 1].extent;
    int numRow = inputTensor->elementSize() / lengthRow;

    mParams.mLengthRow = lengthRow;
    mParams.mNumRow = numRow;

    auto boolDescendFlag = mOp->main_as_TopKV2();
    if (boolDescendFlag != nullptr) {
        mParams.mDescendFlag = boolDescendFlag ? 1 : -1;
    }

    mParams.mNumElePerRow = mParams.mLengthRow;
    mParams.mNumK = outputs[0]->buffer().dim[outputs[0]->buffer().dimensions-1].extent;
    auto smemLimit = static_cast<CUDABackend*>(backend())->getCUDARuntime()->smemPerBlock();
    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
        mParams.mNumThreadPerBlock = CalculateNumThreadPerBlock<int, int>(mParams.mNumK, smemLimit);
        mParams.mNumElePerThread = CalcualteNumElePerThread<int, int>(mParams.mNumK, mParams.mNumElePerRow, mParams.mNumThreadPerBlock, smemLimit);
    } else if (static_cast<CUDABackend*>(backend())->useFp16()) {
        mParams.mNumThreadPerBlock = CalculateNumThreadPerBlock<int, half>(mParams.mNumK, smemLimit);
        mParams.mNumElePerThread = CalcualteNumElePerThread<int, half>(mParams.mNumK, mParams.mNumElePerRow, mParams.mNumThreadPerBlock, smemLimit);
    } else {
        mParams.mNumThreadPerBlock = CalculateNumThreadPerBlock<int, float>(mParams.mNumK, smemLimit);
        mParams.mNumElePerThread = CalcualteNumElePerThread<int, float>(mParams.mNumK, mParams.mNumElePerRow, mParams.mNumThreadPerBlock, smemLimit);
    }
    mParams.mNumElePerBlock = mParams.mNumElePerThread * mParams.mNumThreadPerBlock;
    mParams.mNumBlockPerRow = (mParams.mNumElePerRow - 1 + mParams.mNumElePerBlock) / mParams.mNumElePerBlock;
    mParams.mNumBlockFinal = mParams.mNumRow;
    mParams.mNumThreadFinal = std::pow(2, (std::ceil(std::log2(mParams.mNumBlockPerRow))));
    mParams.mNumBlockTotal = mParams.mNumBlockPerRow * mParams.mNumRow;

    // prepare temp buffer
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();

    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
        auto bufferIndices = pool->alloc(mParams.mNumBlockTotal * mParams.mNumK * sizeof(int));
        mParams.mBufferIndices = (void*)((uint8_t*)bufferIndices.first + bufferIndices.second);
        auto  bufferValues = pool->alloc(mParams.mNumBlockTotal * mParams.mNumK * sizeof(int));
        mParams.mBufferValues = (void*)((uint8_t*)bufferValues.first + bufferValues.second);
        pool->free(bufferIndices);
        pool->free(bufferValues);
    } else if (static_cast<CUDABackend*>(backend())->useFp16()) {
        auto bufferIndices = pool->alloc(mParams.mNumBlockTotal * mParams.mNumK * sizeof(int));
        mParams.mBufferIndices = (void*)((uint8_t*)bufferIndices.first + bufferIndices.second);
        auto bufferValues = pool->alloc(mParams.mNumBlockTotal * mParams.mNumK * sizeof(half));
        mParams.mBufferValues = (void*)((uint8_t*)bufferValues.first + bufferValues.second);
        pool->free(bufferIndices);
        pool->free(bufferValues);
    } else {
        auto bufferIndices = pool->alloc(mParams.mNumBlockTotal * mParams.mNumK * sizeof(int));
        mParams.mBufferIndices = (void*)((uint8_t*)bufferIndices.first + bufferIndices.second);
        auto bufferValues = pool->alloc(mParams.mNumBlockTotal * mParams.mNumK * sizeof(float));
        mParams.mBufferValues = (void*)((uint8_t*)bufferValues.first + bufferValues.second);
        pool->free(bufferIndices);
        pool->free(bufferValues);

    }

    return NO_ERROR;
}


ErrorCode TopKV2Execution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // get input and output pointers
    void * inputDeviceAddr = reinterpret_cast<void *>(inputs[0]->deviceId());
    void * outputIndicesDeviceAddr = reinterpret_cast<void *>(outputs[1]->deviceId());
    void * outputValuesDeviceAddr = reinterpret_cast<void *>(outputs[0]->deviceId());

    // configure threads
    dim3 grid1 = {(unsigned int)mParams.mNumBlockPerRow, (unsigned int)mParams.mNumRow};
    dim3 block1 = {(unsigned int)mParams.mNumThreadPerBlock, (unsigned int)1};
    int smemSize1 = mParams.mNumThreadPerBlock * mParams.mNumK;
    dim3 grid2 = {(unsigned int)mParams.mNumBlockFinal};
    dim3 block2 = {(unsigned int)mParams.mNumThreadFinal};
    int smemSize2 = mParams.mNumBlockPerRow * mParams.mNumK;

    if (inputs[0]->getType().code == halide_type_int && inputs[0]->getType().bits == 32) {
        TopKAllRows<int, int><<<grid1, block1, smemSize1 * (sizeof(int) + sizeof(int))>>>(static_cast<const int *>(inputDeviceAddr), static_cast<int *>(mParams.mBufferIndices), static_cast<int *>(mParams.mBufferValues), mParams.mNumK, mParams.mLengthRow, mParams.mMinInt, mParams.mDescendFlag);
        checkKernelErrors;
        GetResultAllRows<int, int><<<grid2, block2, smemSize2 * (sizeof(int) + sizeof(int))>>>(static_cast<int *>(outputIndicesDeviceAddr), static_cast<int *>(outputValuesDeviceAddr), static_cast<int *>(mParams.mBufferIndices), static_cast<int *>(mParams.mBufferValues), mParams.mNumK, mParams.mNumBlockPerRow, mParams.mDescendFlag);
        checkKernelErrors;
    } else if (static_cast<CUDABackend*>(backend())->useFp16()) {
        TopKAllRows<int, half><<<grid1, block1, smemSize1 * (sizeof(half) + sizeof(int))>>>(static_cast<const half *>(inputDeviceAddr), static_cast<int *>(mParams.mBufferIndices), static_cast<half *>(mParams.mBufferValues), mParams.mNumK, mParams.mLengthRow, mParams.mMinHalf, mParams.mDescendFlag);
        checkKernelErrors;
        GetResultAllRows<int, half><<<grid2, block2, smemSize2 * (sizeof(half) + sizeof(int))>>>(static_cast<int *>(outputIndicesDeviceAddr), static_cast<half *>(outputValuesDeviceAddr), static_cast<int *>(mParams.mBufferIndices), static_cast<half *>(mParams.mBufferValues), mParams.mNumK, mParams.mNumBlockPerRow, mParams.mDescendFlag);
        checkKernelErrors;
    } else {
        TopKAllRows<int, float><<<grid1, block1, smemSize1 * (sizeof(float) + sizeof(int))>>>(static_cast<const float *>(inputDeviceAddr), static_cast<int *>(mParams.mBufferIndices), static_cast<float *>(mParams.mBufferValues), mParams.mNumK, mParams.mLengthRow, mParams.mMinFloat, mParams.mDescendFlag);
        checkKernelErrors;
        GetResultAllRows<int, float><<<grid2, block2, smemSize2 * (sizeof(float) + sizeof(int))>>>(static_cast<int *>(outputIndicesDeviceAddr), static_cast<float *>(outputValuesDeviceAddr), static_cast<int *>(mParams.mBufferIndices), static_cast<float *>(mParams.mBufferValues), mParams.mNumK, mParams.mNumBlockPerRow, mParams.mDescendFlag);
        checkKernelErrors;
    }

    return NO_ERROR;
}


class TopKV2Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new TopKV2Execution(op, backend);
    }
};


static CUDACreatorRegister<TopKV2Creator> __init(OpType_TopKV2);


}
}
