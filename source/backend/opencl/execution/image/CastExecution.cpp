//
//  CastExecution.cpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/CastExecution.hpp"

namespace MNN {
namespace OpenCL {

CastExecution::CastExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::string& compute, const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    mBuildOptions.emplace(compute);
    auto openCLBackend = static_cast<OpenCLBackend*>(backend);
    auto runtime       = openCLBackend->getOpenCLRuntime();
    mUnits.resize(1);
    auto &unit = mUnits[0];
    unit.kernel = openCLBackend->getOpenCLRuntime()->buildKernel("cast", "cast", mBuildOptions, inputs[0], outputs[0]);
    mMaxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
}
ErrorCode CastExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();

    auto &unit = mUnits[0];
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
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    ret |= unit.kernel->get().setArg(idx++, outputWidth);
    ret |= unit.kernel->get().setArg(idx++, outputHeight);
    ret |= unit.kernel->get().setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg CastExecution");

    std::string kernelName = "cast";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
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

class CastCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto cast = op->main_as_CastParam();
        // cast param srcT is invalid
        // auto srcT = _mapDataType(cast->srcT());
        auto dstT = _mapDataType(cast->dstT());

        const auto &inputDataType = inputs[0]->getType();
        if (inputDataType.bytes() == 4 && cast->dstT() == MNN::DataType_DT_BOOL) {
            return new CastExecution(inputs, outputs, "-DTO_BOOL", op, backend);
        } else {
            return new CastExecution(inputs, outputs, "", op, backend);
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(CastCreator, OpType_Cast, IMAGE);
} // namespace OpenCL
} // namespace MNN
