
//
//  SplitGeluBufExecution.cpp
//  MNN
//
//  Created by MNN on 2024/06/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/SplitGeluBufExecution.hpp"

namespace MNN {
namespace OpenCL {

SplitGeluBufExecution::SplitGeluBufExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}

ErrorCode SplitGeluBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<OpenCLBackend*>(backend())->getOpenCLRuntime();

    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(input->dimensions() == 3);
    MNN_ASSERT(output->dimensions() == 3);
    if(inputs.size() > 1) {
        MNN_ASSERT(inputs[1]->dimensions() == 1);
        MNN_ASSERT(inputs[1]->length(0) == inputs[0]->length(2));
    }
    
    mUnits.clear();
    mUnits.resize(1);
    std::vector<int> outputShape = tensorShapeFormat(output);
    int shape[4] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};
    std::set<std::string> buildOptions;
    if(inputs.size() > 1) {
        buildOptions.emplace("-DDOUBLE_INPUTS");
    }
    int pack_wh = 1;
    if(shape[2] % 4 == 0) {
        pack_wh = 4;
        buildOptions.emplace("-DWH_4");
    }
    auto &unit = mUnits[0];
    std::string kernelName = "splitgelu_buf";
    unit.kernel = runtime->buildKernel("splitgelu_buf", kernelName, buildOptions);
    
    auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    mGWS = {static_cast<uint32_t>(shape[0]),
            static_cast<uint32_t>(UP_DIV(shape[1], 4)),
            static_cast<uint32_t>(UP_DIV(shape[2],pack_wh))};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    if(inputs.size() > 1) {
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[1]));
    }
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, sizeof(shape), shape);

    MNN_CHECK_CL_SUCCESS(ret, "setArg SplitGeluBufExecution");
    
    mLWS = localWS3DDefault(mGWS, maxWorkGroupSize, runtime, "splitgelu_buf", unit.kernel).first;

    unit.globalWorkSize  = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize   = {mLWS[0], mLWS[1], mLWS[2]};
    
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    mOpenCLBackend->endRecord(mRecording);
    return NO_ERROR;
    
}


class SplitGeluBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~SplitGeluBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        
        return new SplitGeluBufExecution(op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(SplitGeluBufCreator, OpType_SplitGeLU, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
