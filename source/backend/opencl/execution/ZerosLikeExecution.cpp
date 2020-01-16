//
//  ZerosLikeExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <array>
#include "backend/opencl/execution/ZerosLikeExecution.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenCL {

ZerosLikeExecution::ZerosLikeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : Execution(backend) {
    // do nothing
}

ErrorCode ZerosLikeExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto& output = openCLImage(outputs[0]);
    const size_t imageWidth = output.getImageInfo<CL_IMAGE_WIDTH>();
    const size_t imageHeight = output.getImageInfo<CL_IMAGE_HEIGHT>();
    std::array<size_t, 3> origin = {0, 0, 0}, region = {imageWidth, imageHeight, 1};
    size_t row_pitch;
    cl_int error;
    auto commandQueue = ((OpenCLBackend*)backend())->getOpenCLRuntime()->commandQueue();
    auto dataMapped = commandQueue.enqueueMapImage(output, true, CL_MAP_WRITE, origin, region, &row_pitch, nullptr, nullptr, nullptr, &error);
    if (dataMapped == nullptr || error != CL_SUCCESS) {
        MNN_ERROR("ZerosLike data map failed\n");
        return OUT_OF_MEMORY;
    }
    ::memset(dataMapped, 0, imageHeight * row_pitch);
    commandQueue.enqueueUnmapMemObject(output, dataMapped);
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ZerosLikeExecution>> __ZerosLikeExecution(OpType_ZerosLike);

} // namespace OpenCL
} // namespace MNN
