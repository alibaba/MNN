//
//  ReductionExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ReductionExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReductionExecution::ReductionExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto reduct = op->main_as_ReductionParam();
    if (nullptr != reduct->dim()) {
        for (int i = 0; i < reduct->dim()->size(); ++i) {
            mAxis.insert(reduct->dim()->data()[i]);
        }
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            mReductType = "0";
            break;
        case ReductionType_MAXIMUM:
            mReductType = "1";
            break;
        case ReductionType_MINIMUM:
            mReductType = "2";
            break;
        case ReductionType_PROD:
            mReductType = "3";
            break;
        case ReductionType_SUM:
            mReductType = "4";
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionExecution init !\n");
#endif
}

ErrorCode ReductionExecution::generateReductionGWSLWS(const std::vector<uint32_t> &paramArray) {
    if (paramArray.empty()) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::vector<int> kernelArray(3);
    if (runtime->getGpuType() == ADRENO) {
        uint32_t waveSize;
        if (mAxis.size() == 1) {
            waveSize = static_cast<uint32_t>(runtime->GetKernelWaveSize(mReduct1DKernel));
        } else {
            waveSize = static_cast<uint32_t>(runtime->GetKernelWaveSize(mReduct2DKernel));
        }
        mGlobalWorkSize = {4, (waveSize / 4), paramArray[0] * paramArray[2]};
    } else {
        mGlobalWorkSize = {4, paramArray[1] / 16, paramArray[0] * paramArray[2]};
        if (mGlobalWorkSize[1] == 0) {
            mGlobalWorkSize[1] = 1;
        } else if (mGlobalWorkSize[1] > 16) {
            mGlobalWorkSize[1] = 16;
        }
    }
    mLocalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], 1};

    return NO_ERROR;
}

ErrorCode ReductionExecution::prepareReduction1Dkernel(const std::vector<int> &inputArray,
                                                        const Tensor *input, const Tensor *output) {
    if (inputArray.empty()) {
        return INVALID_VALUE;
    }
    const int batch = inputArray[0];
    const int reductSize = inputArray[1];
    const int workNum = inputArray[2];
    const int groupNum = inputArray[3];
    const int channels = inputArray[4];
    std::vector<uint32_t> paramArray = {static_cast<uint32_t>(batch), static_cast<uint32_t>(workNum), static_cast<uint32_t>(groupNum)};
    auto code = generateReductionGWSLWS(paramArray);
    if (NO_ERROR != code) {
        return INVALID_VALUE;
    }
    const int groupWorkSize = mLocalWorkSize[0] * mLocalWorkSize[1] * mLocalWorkSize[2];
    // Each kernel intends to compute computeNum elements.
    const int computeNum = (workNum + groupWorkSize - 1) / groupWorkSize;
    const int lastNum = workNum % groupWorkSize;
    uint32_t idx = 0;
    mReduct1DKernel.setArg(idx++, mGlobalWorkSize[0]);
    mReduct1DKernel.setArg(idx++, mGlobalWorkSize[1]);
    mReduct1DKernel.setArg(idx++, mGlobalWorkSize[2]);
    mReduct1DKernel.setArg(idx++, openCLImage(input));
    mReduct1DKernel.setArg(idx++, openCLImage(output));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(groupWorkSize));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(computeNum));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(lastNum));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(reductSize));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(workNum));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(groupNum));
    mReduct1DKernel.setArg(idx++, channels);

    return NO_ERROR;
}

ErrorCode ReductionExecution::prepareReduction2Dkernel(const std::vector<int> &inputArray,
                                                        const Tensor *input, const Tensor *output) {
    if (inputArray.empty()) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto bufferPool = mOpenCLBackend->getBufferPool();
    const int batch = inputArray[0];
    const int inputHeight = inputArray[1];
    const int inputWidth = inputArray[2];
    const int leftSize = inputArray[3];
    const int channels = inputArray[4];
    cl::Buffer* leftBuffer = bufferPool->alloc(leftSize);
    const uint32_t imageSize = static_cast<uint32_t >(inputHeight * inputWidth);
    std::vector<uint32_t> paramArray = {static_cast<uint32_t>(batch), imageSize, static_cast<uint32_t>(leftSize)};
    auto code = generateReductionGWSLWS(paramArray);
    if (NO_ERROR != code) {
        return INVALID_VALUE;
    }
    const int groupWorkSize = mLocalWorkSize[0] * mLocalWorkSize[1] * mLocalWorkSize[2];
    // Each kernel intends to compute computeNum elements.
    const int computeNum = (imageSize + groupWorkSize - 1) / groupWorkSize;
    const int lastNum = imageSize % groupWorkSize;
    uint32_t idx = 0;
    mReduct2DKernel.setArg(idx++, mGlobalWorkSize[0]);
    mReduct2DKernel.setArg(idx++, mGlobalWorkSize[1]);
    mReduct2DKernel.setArg(idx++, mGlobalWorkSize[2]);
    mReduct2DKernel.setArg(idx++, openCLImage(input));
    mReduct2DKernel.setArg(idx++, openCLImage(output));
    mReduct2DKernel.setArg(idx++, (groupWorkSize * 4 * sizeof(float)), nullptr);
    mReduct2DKernel.setArg(idx++, *leftBuffer);
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(groupWorkSize));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(computeNum));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(lastNum));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(leftSize));
    mReduct2DKernel.setArg(idx++, channels);

    return NO_ERROR;
}

ErrorCode ReductionExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input = inputs[0];
    auto output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    
    // For fast test, when there is inputs[1] represent axis to reduce, assume it reduce all dim
    // TODO: remove the assumption, support general dims
    if (inputs.size() >= 2) {
        mAxis.clear();
        for (int i = 0; i < input->dimensions(); ++i) {
            mAxis.insert(i);
        }
    }
    
    mUnits.resize(1);
    if (mAxis.size() == input->dimensions() && mReductType == "4") {
        auto kernel = runtime->buildKernel("reduction", "reduce_sum_all", {});
        kernel.setArg(0, openCLImage(input));
        kernel.setArg(1, openCLImage(output));
        kernel.setArg(2, inputShape[2]);
        kernel.setArg(3, inputShape[3]);
        mUnits[0].kernel = kernel;
        mUnits[0].localWorkSize = cl::NullRange;
        mUnits[0].globalWorkSize = {static_cast<uint32_t>(1)};
        return NO_ERROR;
    }
    
    // only support channel
    if (mAxis.size() == 3 && mAxis.find(0) != mAxis.end() && mReductType == "4") {
        auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
        bool support = true;
        if (layout == MNN_DATA_FORMAT_NCHW || layout == MNN_DATA_FORMAT_NC4HW4) {
            if (mAxis.find(1) != mAxis.end()) {
                support = false;
            }
        } else {
            if (mAxis.find(3) != mAxis.end()) {
                support = false;
            }
        }
        if (!support) {
            return NOT_SUPPORT;
        }
        
        bool useLocal = false;
        /* on Mac Intel Iris Pro 1536 MB, 16x32x128x128 (NCHW) input
         * reduce_use_local_along_channel: 140.30ms
         * reduce_along_channel: 221.34ms
         * NOTE: time cost above include data transfer between CPU and GPU
         */
        cl::Kernel kernel;
        if (useLocal) {
            kernel = runtime->buildKernel("reduction", "reduce_use_local_along_channel", {});
        } else {
            kernel = runtime->buildKernel("reduction", "reduce_along_channel", {});
        }
        
        kernel.setArg(0, openCLImage(input));
        kernel.setArg(1, openCLImage(output));
        kernel.setArg(2, inputShape[2]);
        kernel.setArg(3, (mReductType == "0" ? 1 : 0));
        
        if (useLocal) {
            const int N_H = inputShape[0] * inputShape[1];
            const int tile = ALIMIN(N_H, 16), step = UP_DIV(N_H, tile);
            kernel.setArg(4, step);
            kernel.setArg(5, cl::Local(tile * 4 * sizeof(float)));
            kernel.setArg(6, tile * 4);
            mUnits[0].localWorkSize = {static_cast<uint32_t>(tile), static_cast<uint32_t>(1)};
            mUnits[0].globalWorkSize = {
                static_cast<uint32_t>(tile),
                static_cast<uint32_t>(UP_DIV(inputShape[3], 4))
            };
        } else {
            mUnits[0].localWorkSize = cl::NullRange;
            mUnits[0].globalWorkSize = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4))};
        }
        
        mUnits[0].kernel = kernel;
        
        return NO_ERROR;
    }
    
    int batch         = inputShape.at(0);
    int inputHeight   = inputShape.at(1);
    int inputWidth    = inputShape.at(2);
    int channels      = inputShape.at(3);
    int channelBlocks = UP_DIV(channels, 4);
    std::vector<int> inputArray(5);

    std::set<std::string> buildOptions;
    buildOptions.emplace("-DREDUCE_TYPE=" + mReductType);
    if (runtime->getGpuType() == ADRENO) {
        buildOptions.emplace("-DNON_QUALCOMM_ADRENO");
    }
    if (mAxis.size() == 0) {
        return NOT_SUPPORT;
    }

    if (mAxis.size() == 1) {
        auto iter = mAxis.find(0);
        if (iter != mAxis.end()) {
            return NOT_SUPPORT;
        }
        // reduct H axis
        iter = mAxis.find(1);
        if (iter != mAxis.end()) {
            buildOptions.emplace("-DREDUCTION_H");
            inputArray = {batch, inputHeight, inputWidth, channelBlocks, channels};
        }
        // reduct W axis
        iter = mAxis.find(2);
        if (iter != mAxis.end()) {
            buildOptions.emplace("-DREDUCTION_W");
            inputArray = {batch, inputWidth, inputHeight, channelBlocks, channels};
        }
        // reduct C axis
        iter = mAxis.find(3);
        if (iter != mAxis.end()) {
            buildOptions.emplace("-DREDUCTION_C");
            inputArray = {batch,  channelBlocks, inputWidth, inputHeight, channels};
        }
        if (mReduct1DKernel.get() == nullptr) {
            mReduct1DKernel = runtime->buildKernel("reduction", "reduct_1d", buildOptions);
        }
        prepareReduction1Dkernel(inputArray, inputs[0], outputs[0]);
        return NO_ERROR;
    }

    if (mAxis.size() == 2) {
        auto iter = mAxis.find(0);
        if (iter != mAxis.end()) {
            return NOT_SUPPORT;
        }
        iter = mAxis.find(1);
        if (iter == mAxis.end()) {
            buildOptions.emplace("-DREDUCTION_WC");
            inputArray = {batch, channelBlocks, inputWidth, inputHeight, channels};
        }
        iter = mAxis.find(2);
        if (iter == mAxis.end()) {
            buildOptions.emplace("-DREDUCTION_HC");
            inputArray = {batch, inputHeight, channelBlocks, inputWidth, channels};
        }
        iter = mAxis.find(3);
        if (iter == mAxis.end()) {
            buildOptions.emplace("-DREDUCTION_HW");
            inputArray = {batch, inputHeight, inputWidth, channelBlocks, channels};
        }
        if (mReduct2DKernel.get() == nullptr) {
            mReduct2DKernel = runtime->buildKernel("reduction", "reduct_2d", buildOptions);
        }
        prepareReduction2Dkernel(inputArray, inputs[0], outputs[0]);
        return NO_ERROR;
    }

    if (mAxis.size() == 3) {
        auto iter = mAxis.find(0);
        if (iter != mAxis.end()) {
            return NOT_SUPPORT;
        }
        buildOptions.emplace("-DREDUCTION_HC");
        buildOptions.emplace("-DREDUCE_W=1");
        inputArray = {batch, inputHeight, channelBlocks, inputWidth, channels};
        if (mReduct2DKernel.get() == nullptr) {
            mReduct2DKernel = runtime->buildKernel("reduction", "reduct_2d", buildOptions);
        }
        prepareReduction2Dkernel(inputArray, inputs[0], outputs[0]);
        return NO_ERROR;
    }
    return NOT_SUPPORT;
}

ErrorCode ReductionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionExecution onExecute !\n");
#endif
    if (mReductType == "4" && (mAxis.size() == inputs[0]->dimensions() || (mAxis.size() == 3 && mAxis.find(0) != mAxis.end()))) {
        return CommonExecution::onExecute(inputs, outputs);
    }
    
    if (mAxis.size() == 1) {
        run3DKernelDefault(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    } else {
        run3DKernelDefault(mReduct2DKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ReductionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ReductionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                 const MNN::Op *op, Backend *backend) const override {
        if (inputs[0]->getDimensionType() == Tensor::TENSORFLOW) {
            return new ReductionExecution(op, backend);
        }
        return NULL;
    }
};

OpenCLCreatorRegister<ReductionCreator> __reduction_op(OpType_Reduction);
} // namespace OpenCL
} // namespace MNN
