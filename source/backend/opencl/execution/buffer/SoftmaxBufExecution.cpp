//
//  SoftmaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/SoftmaxBufExecution.hpp"

namespace MNN {
namespace OpenCL {

SoftmaxBufExecution::SoftmaxBufExecution(const std::vector<Tensor *> &inputs, int axis, const MNN::Op* Op, Backend *backend)
    : CommonExecution(backend, Op) {
    mAxis          = axis;
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"});
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

int SoftmaxBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode SoftmaxBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    
    const auto dims = input->buffer().dimensions;
    auto runtime       = mOpenCLBackend->getOpenCLRuntime();

    auto MaxLocalSize = std::min(std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize), static_cast<uint32_t>(256));

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;
    if (mNeedUnpackC4) {
        int totalSize = 1;
        for (int i = 1; i < dims; ++i) {
            totalSize *= input->length(i);
        }
        mTempTensor.reset(Tensor::createDevice<float>({totalSize}));
        mOpenCLBackend->onAcquireBuffer(mTempTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempTensor.get(), Backend::DYNAMIC);
    }
    
    int inside  = 1;
    int outside = 1;
    int channel = 1;
    for (int i = 0; i < mAxis; ++i) {
        outside *= input->length(i);
    }
    channel = input->length(mAxis);
    for (int i = mAxis + 1; i < dims; ++i) {
        inside *= input->length(i);
    }
    
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
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        MNN_CHECK_CL_SUCCESS(ret, "setArg buffer_convert_to_buffer");

        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mLocalWorkSize = {16, std::max((uint32_t)1, maxWorkGroupSize / 16), 1};
        
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
    }
    
    // softmax
    {
        Unit unit;
        int localSize = getLocalSize(channel, MaxLocalSize);
        if(localSize < 4){
            localSize = 1;
        }
        std::set<std::string> buildOptions = mBuildOptions;
        buildOptions.emplace("-DARGMAX_LOCAL_SIZE=" + std::to_string(localSize));
        std::string kernelName;
        if(inside == 1){
            buildOptions.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
            unit.kernel = runtime->buildKernel("self_attention_buf", "softmax_inside", buildOptions, inputs[0], outputs[0]);
            mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(outside), static_cast<uint32_t>(1)};
        }
        else if(inside % 4 == 0){
            unit.kernel = runtime->buildKernel("softmax_buf", "softmax_v4_buf", buildOptions);
            mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(UP_DIV(inside, 4)), static_cast<uint32_t>(outside)};
        }else {
            unit.kernel = runtime->buildKernel("softmax_buf", "softmax_buf", buildOptions);
            mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(inside), static_cast<uint32_t>(outside)};
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mLocalWorkSize = {(uint32_t)(localSize), 1, 1};
        
        cl_int ret = CL_SUCCESS;
        
        uint32_t idx    = 0;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        if(mNeedUnpackC4){
            ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
            ret |= unit.kernel->get().setArg(idx++, openCLImage(mTempTensor.get()));
        }else{
            ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
            ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        }
        if(inside == 1){
            ret |= unit.kernel->get().setArg(idx++, channel);
            int shape[4] = {1, outside, channel, 1};
            ret |= unit.kernel->get().setArg(idx++, shape);
        } else {
            ret |= unit.kernel->get().setArg(idx++, inside);
            ret |= unit.kernel->get().setArg(idx++, outside);
            ret |= unit.kernel->get().setArg(idx++, channel);
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg SoftmaxBufExecution");
        if(localSize == 1){
            mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "softmax_buf", unit.kernel).first;
        }
        
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
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
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTempTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, sizeof(shape), shape);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        MNN_CHECK_CL_SUCCESS(ret, "setArg buffer_convert_to_buffer");

        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mLocalWorkSize = {16, std::max((uint32_t)1, maxWorkGroupSize / 16), 1};
        
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
    }
    
    return NO_ERROR;
}

class SoftmaxBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto dimType = inputs[0]->getDimensionType();
        if (dimType == Tensor::TENSORFLOW && inputs[0]->dimensions() == 4) {
            int index[4] = {0, 2, 3, 1};
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }

            axis = index[axis];
            //1 : channel //2 : height
            if (1 == axis || 2 == axis || 3 == axis) {
                return new SoftmaxBufExecution(inputs, axis, op, backend);
            }
            return nullptr;
        } else {
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }

            if (1 == axis || 2 == axis || 3 == axis) {
                return new SoftmaxBufExecution(inputs, axis, op, backend);
            }
            return nullptr;
        }
    }
};
REGISTER_OPENCL_OP_CREATOR(SoftmaxBufCreator, OpType_Softmax, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_OPENCL_BUFFER_CLOSED */

