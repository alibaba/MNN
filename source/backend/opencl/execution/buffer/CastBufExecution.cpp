//
//  CastBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/CastBufExecution.hpp"

namespace MNN {
namespace OpenCL {

CastBufExecution::CastBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::string& compute, const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    mBuildOptions.emplace(compute);
    auto runtime = static_cast<OpenCLBackend*>(backend)->getOpenCLRuntime();
    unit.kernel = runtime->buildKernel("cast_buf", "cast_buf", mBuildOptions, inputs[0], outputs[0]);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
}
ErrorCode CastBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto &unit = mUnits[0];
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);

    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(outputHeight),
        static_cast<uint32_t>(batch * channelBlocks),
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, outputWidth);
    ret |= unit.kernel->get().setArg(idx++, outputHeight);
    ret |= unit.kernel->get().setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg CastBufExecution");

    std::string kernelName = "cast_buf";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
    return NO_ERROR;
}

static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}

class CastBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto cast = op->main_as_CastParam();
        // cast param srcT is invalid
        // auto srcT = _mapDataType(cast->srcT());
        auto dstT = _mapDataType(cast->dstT());

        const auto &inputDataType = inputs[0]->getType();
        if (inputDataType.bytes() == 4 && cast->dstT() == MNN::DataType_DT_BOOL) {
            return new CastBufExecution(inputs, outputs, "-DTO_BOOL", op, backend);
        } else {
            return new CastBufExecution(inputs, outputs, "", op, backend);
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(CastBufCreator, OpType_Cast, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
