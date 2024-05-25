//
//  ScaleExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/ScaleExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

ScaleExecution::ScaleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution init !\n");
#endif
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto openclBackend        = (OpenCLBackend *)backend;
    mOpenCLBackend            = static_cast<OpenCLBackend *>(backend);
    const auto *scaleParams   = op->main_as_Scale();
    int scaleSize             = scaleParams->scaleData()->size();
    const float *scaleDataPtr = scaleParams->scaleData()->data();
        
    size_t buffer_size = ALIGN_UP4(scaleSize) * sizeof(float);
    cl::Buffer scaleBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto scalePtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        scaleBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(nullptr != scalePtrCL && error == CL_SUCCESS){
        ::memset(scalePtrCL, 0, buffer_size);
        ::memcpy(scalePtrCL, scaleDataPtr, scaleSize * sizeof(float));
    }else{
        MNN_ERROR("Map error scalePtrCL == nullptr \n");
    }
    openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(scaleBuffer, scalePtrCL);

    mScale.reset(Tensor::createDevice<float>({1, 1, 1, scaleSize}));
    backend->onAcquireBuffer(mScale.get(), Backend::STATIC);
    copyBufferToImage(openclBackend->getOpenCLRuntime(), scaleBuffer, openCLImage(mScale.get()), UP_DIV(scaleSize, 4),
                      1);

    std::set<std::string> buildOptions;
    if (nullptr != scaleParams->biasData() && nullptr != scaleParams->biasData()->data()) {
        int biasSize = scaleParams->biasData()->size();
        MNN_ASSERT(biasSize == scaleSize);
        const float *biasDataPtr = scaleParams->biasData()->data();
        
        int buffer_size = ALIGN_UP4(biasSize) * sizeof(float);
        cl::Buffer biasBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        cl_int error;
        auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(nullptr != biasPtrCL && error == CL_SUCCESS){
            ::memset(biasPtrCL, 0, buffer_size);
            ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
        }else{
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
        std::shared_ptr<Tensor> bias;
        bias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
        backend->onAcquireBuffer(bias.get(), Backend::STATIC);
        copyBufferToImage(openclBackend->getOpenCLRuntime(), biasBuffer, openCLImage(bias.get()), UP_DIV(biasSize, 4),
                          1);
        mBias = bias;
        buildOptions.emplace("-DHAS_BIAS");
        mHasBias = true;
    }
    std::string kernelName = "scale";
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();
    unit.kernel          = runtime->buildKernel("scale", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution init !\n");
#endif
}

ScaleExecution::~ScaleExecution() {
    if (nullptr != mBias) {
        mOpenCLBackend->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    mOpenCLBackend->onReleaseBuffer(mScale.get(), Backend::STATIC);
}

ErrorCode ScaleExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution onResize !\n");
#endif
    
    auto &unit = mUnits[0];
    std::vector<int> inputShape = tensorShapeFormat(inputs[0]);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);

    const std::vector<uint32_t> &gws = {static_cast<uint32_t>(channelBlocks),
                                        static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height * batch)};
    
    uint32_t idx                     = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, gws[0]);
    ret |= unit.kernel->get().setArg(idx++, gws[1]);
    ret |= unit.kernel->get().setArg(idx++, gws[2]);

    ret |= unit.kernel->get().setArg(idx++, openCLImage(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(mScale.get()));
    if (mHasBias) {
        ret |= unit.kernel->get().setArg(idx++, openCLImage(mBias.get()));
    }
    ret |= unit.kernel->get().setArg(idx++, openCLImage(outputs[0]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg ScaleExecution");

    std::string name = "scale";
    std::vector<uint32_t> mGWS{1, 1, 1, 1};
    std::vector<uint32_t> mLWS{1, 1, 1, 1};
    mLWS = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
    for (size_t i = 0; i < gws.size(); ++i) {
        mGWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, mLWS[i]));
    }
    
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution onResize !\n");
#endif
    return NO_ERROR;
}

using ScaleCreator = TypedCreator<ScaleExecution>;
REGISTER_OPENCL_OP_CREATOR(ScaleCreator, OpType_Scale, IMAGE);

} // namespace OpenCL
} // namespace MNN
