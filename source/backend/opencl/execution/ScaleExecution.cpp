//
//  ScaleExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/ScaleExecution.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

ScaleExecution::ScaleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution init !\n");
#endif
    auto openclBackend        = (OpenCLBackend *)backend;
    mOpenCLBackend            = static_cast<OpenCLBackend *>(backend);
    const auto *scaleParams   = op->main_as_Scale();
    int scaleSize             = scaleParams->scaleData()->size();
    const float *scaleDataPtr = scaleParams->scaleData()->data();
    cl::Buffer scaleBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                           UP_DIV(scaleSize, 4) * 4 * sizeof(float));
    cl_int error;
    auto scalePtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        scaleBuffer, true, CL_MAP_WRITE, 0, ALIGN_UP4(scaleSize) * sizeof(float), nullptr, nullptr, &error);
    if(nullptr != scalePtrCL && error == CL_SUCCESS){
        ::memset(scalePtrCL, 0, ALIGN_UP4(scaleSize) * sizeof(float));
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
        cl::Buffer biasBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              UP_DIV(biasSize, 4) * 4 * sizeof(float));
        cl_int error;
        auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            biasBuffer, true, CL_MAP_WRITE, 0, ALIGN_UP4(biasSize) * sizeof(float), nullptr, nullptr, &error);
        if(nullptr != biasPtrCL && error == CL_SUCCESS){
            ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
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
    mKernel                = runtime->buildKernel("scale", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

    mAreadySetArg = false;
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

ErrorCode ScaleExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution onResize !\n");
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution onResize !\n");
#endif
    std::vector<int> inputShape = tensorShapeFormat(inputs[0]);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);

    const std::vector<uint32_t> &gws = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height * batch)};
    uint32_t idx                     = 0;
    mKernel.setArg(idx++, gws[0]);
    mKernel.setArg(idx++, gws[1]);
    mKernel.setArg(idx++, gws[2]);

    mKernel.setArg(idx++, openCLImage(inputs[0]));
    mKernel.setArg(idx++, openCLImage(mScale.get()));
    if (mHasBias) {
        mKernel.setArg(idx++, openCLImage(mBias.get()));
    }
    mKernel.setArg(idx++, openCLImage(outputs[0]));
    return NO_ERROR;
}

ErrorCode ScaleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution onExecute !\n");
#endif
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);

    const std::vector<uint32_t> &gws = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height * batch)};

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());

    cl::Event event;
    cl_int error;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(
        mKernel, cl::NullRange, cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1], roundUpGroupWorkSize[2]),
        cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);

    MNN_CHECK_CL_SUCCESS(error);

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ScaleExecution>> __scale_op(OpType_Scale);

} // namespace OpenCL
} // namespace MNN
