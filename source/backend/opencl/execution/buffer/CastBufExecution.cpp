//
//  CastBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/CastBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

CastBufExecution::CastBufExecution(const std::string& compute, Backend* backend) : Execution(backend) {
    mBuildOptions.emplace(compute);
}
ErrorCode CastBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    mKernel = runtime->buildKernel("cast_buf", "cast_buf", mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

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
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, outputWidth);
    ret |= mKernel.setArg(idx++, outputHeight);
    ret |= mKernel.setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg CastBufExecution");

    std::string kernelName = "cast_buf";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    return NO_ERROR;
}

ErrorCode CastBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start CastBufExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Cast", event});
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end CastBufExecution onExecute...");
#endif
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
            return new CastBufExecution("-DTO_BOOL", backend);
        }
        if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<int32_t>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<int8_t>() == inputDataType) {
            return new CastBufExecution("", backend);
        }
        MNN_PRINT("Don't support cast form %d, %d to %d\n", inputDataType.code, inputDataType.bits, cast->dstT());
        return nullptr;
    }
};

OpenCLCreatorRegister<CastBufCreator> __CastBuf__(OpType_Cast, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
