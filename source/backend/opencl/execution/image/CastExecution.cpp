//
//  CastExecution.cpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/CastExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

CastExecution::CastExecution(const std::string& compute, Backend* backend) : Execution(backend) {
    mBuildOptions.emplace(compute);
}
ErrorCode CastExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    openCLBackend->startRecord(mRecording);
    mKernel = runtime->buildKernel("cast", "cast", mBuildOptions);
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
    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, openCLImage(output));
    ret |= mKernel.setArg(idx++, outputWidth);
    ret |= mKernel.setArg(idx++, outputHeight);
    ret |= mKernel.setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg CastExecution");

    std::string kernelName = "cast";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    openCLBackend->recordKernel3d(mKernel, mGlobalWorkSize, mLocalSize);
    openCLBackend->endRecord(mRecording);
    return NO_ERROR;
}

ErrorCode CastExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start CastExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Cast", event});
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        if(mOpenCLBackend->isDevideOpRecord())
            mOpenCLBackend->addRecord(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End CastExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end CastExecution onExecute...");
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
            return new CastExecution("-DTO_BOOL", backend);
        }
        if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<int32_t>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
            return new CastExecution("", backend);
        }
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<int8_t>() == inputDataType) {
            return new CastExecution("", backend);
        }
        MNN_PRINT("Don't support cast form %d, %d to %d\n", inputDataType.code, inputDataType.bits, cast->dstT());
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(CastCreator, OpType_Cast, IMAGE);
} // namespace OpenCL
} // namespace MNN
