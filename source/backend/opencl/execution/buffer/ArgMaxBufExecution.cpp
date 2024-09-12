//
//  ArgMaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/ArgMaxBufExecution.hpp"

namespace MNN {
namespace OpenCL {

ArgMaxBufExecution::ArgMaxBufExecution(const std::string &compute, const MNN::Op* op, Backend* backend, const int axis) : CommonExecution(backend, op) {
    mBuildOptions.emplace(compute);
    mAxis = axis;
    // Do nothing
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    std::set<std::string> buildOptions = mBuildOptions;
    buildOptions.emplace("-DARGMAX_LOCAL_SIZE=512");
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("argmax_buf", "argmax_buf", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

int ArgMaxBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode ArgMaxBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.clear();
    auto runtime       = mOpenCLBackend->getOpenCLRuntime();
    auto MaxLocalSize = std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize);
    auto input = inputs[0];
    auto output = outputs[0];
    
    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;
    if (mNeedUnpackC4) {
        int inputTotalSize = 1, outputTotalSize = 1;
        for (int i = 1; i < input->dimensions(); ++i) {
            inputTotalSize *= input->length(i);
        }
        for (int i = 1; i < output->dimensions(); ++i) {
            outputTotalSize *= output->length(i);
        }
        mTempInputTensor.reset(Tensor::createDevice<float>({inputTotalSize}));
        mTempOutputTensor.reset(Tensor::createDevice<float>({outputTotalSize}));
        mOpenCLBackend->onAcquireBuffer(mTempInputTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempOutputTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempInputTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempOutputTensor.get(), Backend::DYNAMIC);
        
    }
    if(mAxis < 0){
        mAxis = input->dimensions() + mAxis;
    }
    int inside = 1;
    int outside = 1;
    for(int i = 0; i < mAxis; ++i){
        outside *= input->length(i);
    }
    for(int i = mAxis + 1; i < input->dimensions(); ++i){
        inside *= input->length(i);
    }
    int dim = input->length(mAxis);

    // NC4HW4 -> NCHW
    if(mNeedUnpackC4){
        Unit unit;
        std::vector<int> outputShape = tensorShapeFormat(input);
        int shape[4] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//N C H W
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DINPUT_FORMAT=MNN_DATA_FORMAT_NC4HW4");
        buildOptions.emplace("-DOUTPUT_FORMAT=MNN_DATA_FORMAT_NCHW");
        unit.kernel = runtime->buildKernel("buffer_convert_buf", "buffer_convert_to_buffer", buildOptions, input, output);
        mGlobalWorkSize = {static_cast<uint32_t>(shape[2] * shape[3]), static_cast<uint32_t>(shape[1]), static_cast<uint32_t>(shape[0])};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, sizeof(shape), shape);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTempInputTensor.get()));
        MNN_CHECK_CL_SUCCESS(ret, "setArg buffer_convert_to_buffer");

        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mLocalSize = {16, std::max((uint32_t)1, maxWorkGroupSize / 16), 1};
        
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
        mUnits.emplace_back(unit);
    }
    
    // Argmax
    {
        Unit unit;
        int localSize = getLocalSize(dim, MaxLocalSize);
        if(localSize < 4){
            localSize = 1;
        }
        std::set<std::string> buildOptions = mBuildOptions;
        buildOptions.emplace("-DARGMAX_LOCAL_SIZE=" + std::to_string(localSize));
        std::string kernelName;
        if(inside % 4 == 0){
            kernelName = "argmax_v4_buf";
            unit.kernel = runtime->buildKernel("argmax_buf", kernelName, buildOptions);
            mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(UP_DIV(inside, 4)), static_cast<uint32_t>(outside)};
        }else {
            kernelName = "argmax_buf";
            unit.kernel = runtime->buildKernel("argmax_buf", kernelName, buildOptions);
            mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(inside), static_cast<uint32_t>(outside)};
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mLocalSize = {(uint32_t)(localSize), 1, 1};
        
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        if(mNeedUnpackC4){
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTempInputTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTempOutputTensor.get()));
        }else{
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        }
        ret |= unit.kernel->get().setArg(idx++, inside);
        ret |= unit.kernel->get().setArg(idx++, outside);
        ret |= unit.kernel->get().setArg(idx++, dim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg ArgMaxBufExecution");
        
        if(localSize == 1){
            mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
        }
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
        mUnits.emplace_back(unit);
    }
    
    // NCHW -> NC4HW4
    if(mNeedUnpackC4){
        Unit unit;
        std::vector<int> outputShape = tensorShapeFormat(output);
        int shape[4] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//N C H W
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DINPUT_FORMAT=MNN_DATA_FORMAT_NCHW");
        buildOptions.emplace("-DOUTPUT_FORMAT=MNN_DATA_FORMAT_NC4HW4");
        unit.kernel = runtime->buildKernel("buffer_convert_buf", "buffer_convert_to_buffer", buildOptions, input, output);
        mGlobalWorkSize = {static_cast<uint32_t>(shape[2] * shape[3]), static_cast<uint32_t>(shape[1]), static_cast<uint32_t>(shape[0])};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTempOutputTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, sizeof(shape), shape);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        MNN_CHECK_CL_SUCCESS(ret, "setArg buffer_convert_to_buffer");

        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mLocalSize = {16, std::max((uint32_t)1, maxWorkGroupSize / 16), 1};
        
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
        mUnits.emplace_back(unit);
    }
    return NO_ERROR;
}

class ArgMaxBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto inputDimensionFromat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if(inputDimensionFromat == MNN_DATA_FORMAT_NC4HW4){
            return nullptr;
        }
        int axis = op->main_as_ArgMax()->axis();
        if (op->type() == OpType_ArgMax) {
            return new ArgMaxBufExecution("-DARGMAX", op, backend, axis);
        }else{
            return new ArgMaxBufExecution("", op, backend, axis);
        }
    }
};

REGISTER_OPENCL_OP_CREATOR(ArgMaxBufCreator, OpType_ArgMax, BUFFER);
REGISTER_OPENCL_OP_CREATOR(ArgMaxBufCreator, OpType_ArgMin, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
