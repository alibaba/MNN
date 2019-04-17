//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/UnaryExecution.hpp"
#include <Macro.h>
#include "TensorUtils.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

UnaryExecution::UnaryExecution(const std::string& compute, Backend* backend) : Execution(backend) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend);
    std::set<std::string> buildOptions;
    buildOptions.emplace(" -DOPERATOR=" + compute);
    // FUNC_PRINT_ALL(buildOptions.begin()->c_str(), s);
    auto runtime      = openCLBackend->getOpenCLRuntime();
    mKernel           = runtime->buildKernel("unary", "unary", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

    mAreadySetArg = false;
}

ErrorCode UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...");
#endif
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    if (!mAreadySetArg) {
        int batch        = outputShape.at(0);
        int outputHeight = outputShape.at(1);
        int outputWidth  = outputShape.at(2);
        int channels     = outputShape.at(3);

        int channelBlocks = (channels + 3) / 4;

        mGlobalWorkSize = {
            static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(batch * outputHeight),
        };

        uint32_t idx = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, mGlobalWorkSize[2]);
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(output));

        mAreadySetArg = true;
    }

    const std::vector<uint32_t> lws =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime());
    run3DKernelDefault(mKernel, mGlobalWorkSize, lws, openCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...");
#endif
    return NO_ERROR;
}
class UnaryCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_RSQRT:
                    return new UnaryExecution("rsqrt(in)", backend);
                case UnaryOpOperation_ABS:
                    return new UnaryExecution("fabs(in)", backend);
                default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution("native_recip((float4)1+native_exp(-in))", backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution("tanh(in)", backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
OpenCLCreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
OpenCLCreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
} // namespace OpenCL
} // namespace MNN
