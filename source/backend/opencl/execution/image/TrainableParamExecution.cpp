//
//  TrainableParamExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include "backend/opencl/execution/image/TrainableParamExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

TrainableParamExecution::TrainableParamExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op), mInitialized(false) {
}

TrainableParamExecution::~TrainableParamExecution() {
    // do nothing
}

ErrorCode TrainableParamExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    if (mInitialized) {
        return NO_ERROR;
    }
    mInitialized = true;

    auto output = outputs[0];
    const int blobSize = output->elementSize();
    const float* blobData = mOp->main_as_Blob()->float32s()->data();

    auto openclBackend = static_cast<OpenCLBackend *>(backend());
    auto runtime = openclBackend->getOpenCLRuntime();
    cl::Buffer buffer(runtime->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, blobSize * sizeof(float));
    cl_int error;
    auto bufferPtr = runtime->commandQueue().enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, blobSize * sizeof(float), nullptr, nullptr, &error);
    if (bufferPtr != nullptr && error == CL_SUCCESS) {
        ::memcpy(bufferPtr, blobData, blobSize * sizeof(float));
    } else {
        MNN_ERROR("Map error bufferPtr == nullptr \n");
        return OUT_OF_MEMORY;
    }
    runtime->commandQueue().enqueueUnmapMemObject(buffer, bufferPtr);

    auto format = TensorUtils::getDescribe(output)->dimensionFormat;
    if (format != MNN_DATA_FORMAT_NCHW && format != MNN_DATA_FORMAT_NHWC) {
        MNN_ERROR("Variable's blob dataFormat should be MNN_DATA_FORMAT_NCHW or MNN_DATA_FORMAT_NHWC\n");
        return NOT_SUPPORT;
    }
    std::shared_ptr<Tensor> bufferTensor;
    MNN::OpenCL::ImageBufferConvertor convertor(runtime);
    if (format == MNN_DATA_FORMAT_NCHW) {
        bufferTensor.reset(new Tensor(output, Tensor::CAFFE, false));
        bufferTensor->buffer().device = (uint64_t)(&buffer);
        convertor.convertBufferToImage(bufferTensor.get(), MNN::OpenCL::NCHW_BUFFER, output, true);
    } else {
        bufferTensor.reset(new Tensor(output, Tensor::TENSORFLOW, false));
        bufferTensor->buffer().device = (uint64_t)(&buffer);
        convertor.convertBufferToImage(bufferTensor.get(), MNN::OpenCL::NHWC_BUFFER, output, true);
    }

    return NO_ERROR;
}

using TrainableParamCreator = TypedCreator<TrainableParamExecution>;
REGISTER_OPENCL_OP_CREATOR(TrainableParamCreator, OpType_TrainableParam, IMAGE);
}
}
