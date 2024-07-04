//
//  DepthwiseConvBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/DepthwiseConvBufExecution.hpp"
#include "backend/opencl/execution/buffer/DepthwiseConvSubgroupBufExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseConvBufExecution::DepthwiseConvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mResource->mConv2dParams       = op->main_as_Convolution2D();
    mResource->mConv2dCommonParams = mResource->mConv2dParams->common();
    mResource->mStrides            = {mResource->mConv2dCommonParams->strideY(), mResource->mConv2dCommonParams->strideX()};
    mResource->mDilations          = {mResource->mConv2dCommonParams->dilateY(), mResource->mConv2dCommonParams->dilateX()};

    int kernelWidth   = mResource->mConv2dCommonParams->kernelX();
    int kernelHeight  = mResource->mConv2dCommonParams->kernelY();
    int outputChannel = mResource->mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

        
    const float* filterDataPtr = nullptr;
    int filterDataSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, mResource->mConv2dParams, &filterDataPtr, &filterDataSize);

    mResource->mFilter.reset(Tensor::createDevice<float>({1, ROUND_UP(filterImageShape[1], 2)/*for kernel C8 read*/, 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
    size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
    MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
        
    bool needTrans = true;
    bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mResource->mFilter.get(), needTrans);
    
    if (mResource->mConv2dCommonParams->relu() == true) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mConv2dCommonParams->relu6() == true) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
}

DepthwiseConvBufExecution::~DepthwiseConvBufExecution() {
    // Do nothing
}

DepthwiseConvBufExecution::DepthwiseConvBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend *backend)
    : ConvBufCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool DepthwiseConvBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DepthwiseConvBufExecution(mResource, op, bn);
    return true;
}

ErrorCode DepthwiseConvBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    if (mResource->mConv2dCommonParams->strideX() == 1 && mResource->mConv2dCommonParams->strideY() == 1 &&
        mResource->mConv2dCommonParams->dilateX() == 1 && mResource->mConv2dCommonParams->dilateY() == 1) {
        mStride_1 = true;
    }

    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int outputChannel  = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mResource->mConv2dParams->common()->kernelY();
    const int filterWidth        = mResource->mConv2dParams->common()->kernelX();
    
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outputChannel) + "_" + std::to_string(filterHeight) + "_" + std::to_string(filterWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    if(mStride_1) {
        // {"depthwise_conv2d_s1_c4h1w4", "depthwise_conv2d_s1_c8h1w4", "depthwise_conv2d_s1_c8h1w2"};
        const int total_kernel = 3;
        std::string kernelName[total_kernel] = {"depthwise_conv2d_s1_c4h1w4", "depthwise_conv2d_s1_c8h1w4", "depthwise_conv2d_s1_c8h1w2"};
        int itemC[total_kernel] = {4, 8, 8};
        int itemW[total_kernel] = {4, 4, 2};
        int itemH[total_kernel] = {1, 1, 1};

        int actual_kernel = total_kernel;
        
        
        if(kernelShape[0]==3 && kernelShape[1]==3 && paddingShape[0]==1 && paddingShape[1]==1) {
            //{"depthwise_conv2d_k3s1p1_c4h1w2", "depthwise_conv2d_k3s1p1_c4h2w2"}
            actual_kernel = 2;
            kernelName[0] = "depthwise_conv2d_k3s1p1_c4h1w2";
            itemC[0]      = 4;
            itemW[0]      = 2;
            itemH[0]      = 1;

            kernelName[1] = "depthwise_conv2d_k3s1p1_c4h2w2";
            itemC[1]      = 4;
            itemW[1]      = 2;
            itemH[1]      = 2;
        }
        
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
            actual_kernel = 1;
        }

        std::shared_ptr<KernelWrap> kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[knl_idx], mResource->mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                        
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
            
            uint32_t idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannels));
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(kernelShape), kernelShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(paddingShape), paddingShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(dilationShape), dilationShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(strideShape), strideShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputWidth, itemW[knl_idx]));
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputChannel, 4));
            MNN_CHECK_CL_SUCCESS(ret, "setArg DepthwiseConvBufExecution Stride_1 Kernel Select");

            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
            //printf("depthwiseCovs1 %d, %d\n", knl_idx, retTune.second);
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        unit.kernel     = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[min_index], mResource->mBuildOptions);
        
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
        ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputWidth, itemW[min_index]));
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputChannel, 4));
        MNN_CHECK_CL_SUCCESS(ret, "setArg DepthwiseConvBufExecution Stride_1");

        //printf("DepthwiseConvBufs1 %d, %d %d, %d %d, %d %d\n", min_index, mGlobalWorkSize[0], mGlobalWorkSize[1], mLocalWorkSize[0], mLocalWorkSize[1], outputChannel, outputWidth);

    } else {
        // {"depthwise_conv2d_c4h1w4", "depthwise_conv2d_c4h1w2", "depthwise_conv2d_c4h1w1"};
        const int total_kernel = 3;
        const std::string kernelName[total_kernel] = {"depthwise_conv2d_c4h1w1", "depthwise_conv2d_c4h1w4", "depthwise_conv2d_c4h1w2"};
        int itemC[total_kernel] = {4, 4, 4};
        int itemW[total_kernel] = {1, 4, 2};
        
        int actual_kernel = total_kernel;
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
            actual_kernel = 1;
        }

        std::shared_ptr<KernelWrap> kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[knl_idx], mResource->mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                        
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            
            uint32_t idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannels));
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(kernelShape), kernelShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(paddingShape), paddingShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(dilationShape), dilationShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(strideShape), strideShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputWidth, itemW[knl_idx]));
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputChannel, 4));
            MNN_CHECK_CL_SUCCESS(ret, "setArg DepthwiseConvBufExecution Kernel Select");

            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
            //printf("depthwiseCov!! %d, %d\n", knl_idx, retTune.second);
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        unit.kernel     = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[min_index], mResource->mBuildOptions);
        
        
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
        ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputWidth, itemW[min_index]));
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputChannel, 4));
        MNN_CHECK_CL_SUCCESS(ret, "setArg DepthwiseConvBufExecution");

        //printf("DepthwiseConvBuf!! %d, %d %d, %d %d, %d %d\n", min_index, mGlobalWorkSize[0], mGlobalWorkSize[1], mLocalWorkSize[0], mLocalWorkSize[1], outputChannel, outputWidth);
    }
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return NO_ERROR;
}

class DepthwiseConvolutionBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~DepthwiseConvolutionBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        
        MNN_ASSERT(inputs.size() <= 3);
        if (inputs.size() > 1) {
            //MNN_PRINT("multi input depthwise conv for opencl buffer not supoort!\n");
            return nullptr;
        }
        
        MNN_ASSERT(inputs.size() == 1);
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
        if (static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->isSupportedIntelSubgroup() &&
            outputs[0]->channel() >= 16) {
            auto conv2D  = op->main_as_Convolution2D();
            auto pads = ConvolutionCommon::convolutionPadFull(inputs[0], outputs[0], conv2D->common());
            TensorUtils::setTensorChannelPack(inputs[0], 16);
            TensorUtils::setTensorPad(inputs[0], std::get<0>(pads), std::get<2>(pads), 0, 0);
            return new DepthwiseConvSubgroupBufExecution(inputs, op, backend);
        }
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new DepthwiseConvBufExecution(inputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(DepthwiseConvolutionBufCreator, OpType_ConvolutionDepthwise, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
