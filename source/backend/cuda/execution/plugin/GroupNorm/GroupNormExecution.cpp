//
//  GroupNormExecution.cpp
//  MNN
//
//  Created by MNN on 2023/09/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "GroupNormExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

static int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor)
{
    int32_t maxDivisor = -1;
    for (int32_t i = 1; i <= std::sqrt(n); i++)
    {
        if (n % i == 0)
        {
            int32_t divisor1 = n / i;
            int32_t divisor2 = i;

            if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor)
            {
                maxDivisor = divisor1;
            }
            if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor)
            {
                maxDivisor = divisor2;
            }
        }
    }
    return maxDivisor;
}

GroupNormExecution::GroupNormExecution(const MNN::Op* op, Backend* backend) : Execution(backend) {
    auto group_norm_param = op->main_as_GroupNorm();

    mEpsilon = group_norm_param->epsilon();
    mBSwish = group_norm_param->bSwish();
    mGroup = group_norm_param->group();
    if (group_norm_param->gamma() && group_norm_param->beta()) {
        int size = group_norm_param->gamma()->size();
        mGammaTensor.reset(Tensor::createDevice<int32_t>({size}));
        auto status = backend->onAcquireBuffer(mGammaTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in CudaLayerNorm.\n");
        }

        mDeviceGamma = (void *)mGammaTensor.get()->buffer().device;
        const float* gamma_data = group_norm_param->gamma()->data();
        cudaMemcpy(mDeviceGamma, gamma_data, size * sizeof(float), cudaMemcpyHostToDevice);

        if (group_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in CudaLayerNorm.\n");
        }
        mBetaTensor.reset(Tensor::createDevice<int32_t>({size}));
        status = backend->onAcquireBuffer(mBetaTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when beta is acquired in CudaLayerNorm.\n");
        }
     
        mDeviceBeta = (void *)mBetaTensor.get()->buffer().device;
        const float* beta_data = group_norm_param->beta()->data();
        cudaMemcpy(mDeviceBeta, beta_data, size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

size_t GroupNormExecution::getWorkspaceSizeInBytes() const {
    return (sizeof(float) * 2) * mBatch * mGroup; // sizeof(float2) * maxBatchSize * maxNumberOfGroup. float2
                                          // contians two buffers for sum and squared sum
}

ErrorCode GroupNormExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();

    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(input->dimensions() == 4);
    MNN_ASSERT(output->dimensions() == 4);
    mBatch = input->length(0);
    if(inputs.size() > 1) {
        MNN_ASSERT(inputs[1]->dimensions() == 2);
        MNN_ASSERT(inputs[1]->length(0) == inputs[0]->length(0));
        MNN_ASSERT(inputs[1]->length(1) == inputs[0]->length(1));
    }
    auto size = getWorkspaceSizeInBytes(); 
    auto buffer_ws = pool->alloc(size);
    mWorkSpacePtr = (void*)((uint8_t*)buffer_ws.first + buffer_ws.second);
    runtime->memset(mWorkSpacePtr, 0, size);
    return NO_ERROR;
}

ErrorCode GroupNormExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start GroupNormExecution onExecute...");
#endif
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0];
    auto output = outputs[0];

    runtime->memset(mWorkSpacePtr, 0, getWorkspaceSizeInBytes());
    int32_t cPerBlock = 320;
    int32_t maxBlocksPerHW = 1024;

    switch (input->length(1)) {
        case 960:
        case 1920: 
            cPerBlock = 480; break;
        case 512:
        case 256: 
            cPerBlock = 256; break;
        case 128: 
            cPerBlock = 128; break;
        default: 
            cPerBlock = 320;
    }

    mParams.withSwish = bool(mBSwish);
    mParams.dst = static_cast<half*>((void *)output->deviceId());
    if(inputs.size() > 1) {
        mParams.src = nullptr;
        mParams.src_0 = static_cast<half const*>((void *)inputs[0]->deviceId());
        mParams.src_1 = static_cast<half const*>((void *)inputs[1]->deviceId());
    } else {
        mParams.src = static_cast<half const*>((void *)input->deviceId());
    }
    mParams.gamma = static_cast<float const*>(mDeviceGamma);
    mParams.beta = static_cast<float const*>(mDeviceBeta);
    mParams.redBuffer = static_cast<float*>(mWorkSpacePtr);
    mParams.n = input->length(0);
    mParams.h = input->length(2);
    mParams.w = input->length(3);
    mParams.c = input->length(1);

    // Kernel format is NHWC, OP format NC4HW4(NHWC8)
    MNN_ASSERT(mParams.c % 8 == 0);
    mParams.groups = mGroup;
    mParams.hw = mParams.h * mParams.w;
    const int32_t blocksPerHW = findMaxDivisor(mParams.hw, maxBlocksPerHW);
    mParams.hwPerBlock = UP_DIV(mParams.hw, blocksPerHW);
    mParams.cPerBlock = cPerBlock;
    mParams.cPerGroup = mParams.c / mParams.groups;
    mParams.hwc = mParams.hw * mParams.c;
    mParams.invHWC = 1.F / (float) (mParams.hw * mParams.cPerGroup);
    mParams.groupsPerBlock = cPerBlock / mParams.cPerGroup;

    groupNormNHWCSum(mParams);
    checkKernelErrors;

    groupNormNHWCScale(mParams);

    checkKernelErrors;
#ifdef LOG_VERBOSE
    MNN_PRINT("end GroupNormExecution onExecute...");
#endif
    return NO_ERROR;
}


class GroupNormCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if(!static_cast<CUDABackend*>(backend)->useFp16()) {
            MNN_PRINT("CUDA GroupNorm only support fp16 now!\n");
            return nullptr;
        }
        return new GroupNormExecution(op, backend);
    }
};

CUDACreatorRegister<GroupNormCreator> __GroupNormExecution(OpType_GroupNorm);
} // namespace CUDA
} // namespace MNN
#endif