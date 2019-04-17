//
//  LrnExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "LrnExecution.hpp"
#include <Macro.h>
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

LrnExecution::LrnExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start LrnExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto lrn       = op->main_as_LRN();
    mRegionType    = lrn->regionType();
    mLocalSize     = lrn->localSize();
    mAlpha         = lrn->alpha() / (float)mLocalSize;
    mBeta          = lrn->beta();
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    std::string kernelName = "lrn_buffer";
    mKernel                = runtime->buildKernel("lrn", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end LrnExecution init !\n");
#endif
}

ErrorCode LrnExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto bufferPool = mOpenCLBackend->getBufferPool();
    mInputTemp.reset(Tensor::createDevice<float>(tensorShapeFormat(inputs[0])));
    mOutputTemp.reset(Tensor::createDevice<float>(tensorShapeFormat(outputs[0])));
    auto inputBuffer             = bufferPool->alloc(mInputTemp->size());
    auto outputBuffer            = bufferPool->alloc(mOutputTemp->size());
    mInputTemp->buffer().device  = (uint64_t)inputBuffer;
    mOutputTemp->buffer().device = (uint64_t)outputBuffer;

    bufferPool->recycle(inputBuffer);
    bufferPool->recycle(outputBuffer);

    return NO_ERROR;
}

ErrorCode LrnExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start LrnExecution onExecute !\n");
#endif

    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int oN = outputShape.at(0);
    int oH = outputShape.at(1);
    int oW = outputShape.at(2);
    int oC = outputShape.at(3);

    convertImageToNCHWBuffer(input, mInputTemp.get(), mImageToBufferKernel, mOpenCLBackend->getOpenCLRuntime());
    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(oW), static_cast<uint32_t>(oH), static_cast<uint32_t>(oC)};
        const std::vector<uint32_t> lws = {16, 16, 1};
        int32_t shape[4]                = {oW, oH, oC, oN};
        {
            uint32_t idx = 0;
            mKernel.setArg(idx++, openCLBuffer(mInputTemp.get()));
            mKernel.setArg(idx++, openCLBuffer(mOutputTemp.get()));
            mKernel.setArg(idx++, shape);
            mKernel.setArg(idx++, mLocalSize);
            mKernel.setArg(idx++, mAlpha);
            mKernel.setArg(idx++, mBeta);
        }
        run3DKernelDefault(mKernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());
    }
    convertNCHWBufferToImage(mOutputTemp.get(), output, mBufferToImageKernel, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end LrnExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class LRNCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto lrn = op->main_as_LRN();
        if (lrn->regionType() != 0) {
            return nullptr;
        }

        return new LrnExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<LRNCreator> __lrn_op(OpType_LRN);

} // namespace OpenCL
} // namespace MNN
