//
//  ConvSubgroupBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP

#include "ConvBufExecution.hpp"
#include "ConvSubgroupBufExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

static float EstimateOccupancy(int blockWidth, int x, int y, int f, int b, int slm_div_factor, int maxThreadsPerDevice) {

    auto threads =  UP_DIV(x, blockWidth) * y * UP_DIV(f, 16) * slm_div_factor * b;

    return static_cast<float>(threads) / static_cast<float>(maxThreadsPerDevice);
}


static std::pair<int, int> GetTuningParams(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const uint32_t maxWorkGroupSize, const bool isSupportedFP16, const int maxThreadsPerDevice) {

    auto input  = inputs[0];
    auto output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int batch              = outputShape.at(0);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    size_t ic_blocks = UP_DIV(inputChannels, 16);

    size_t max_slm_div_factor = maxWorkGroupSize / 16;
    int blockWidth = 2;
    int slm_div_factor = 1;
    int xf = width * outChannel;
    if (xf <= 256) {
        if (width <= 8 || xf <= 128)
            blockWidth = 2;
        else
            blockWidth = 4;
    } else if (xf <= 1536) {
        blockWidth = 4;
    } else {
        if (width >= 8 && width < 12 && xf < 2600)
            blockWidth = 4;
        else if (width < 12 && xf < 8192)
            blockWidth = 8;
        else
            blockWidth =  8;
    }

    bool slm_exception = width == 3 && height == 3 && !isSupportedFP16 && outChannel <= 512;

    if (!slm_exception)
        while (ic_blocks % (slm_div_factor * 2) == 0 && (slm_div_factor * 2 <= max_slm_div_factor) &&
               EstimateOccupancy(blockWidth, width, height, outChannel, batch, slm_div_factor, maxThreadsPerDevice) <
                   4.0)
            slm_div_factor *= 2;

    return {blockWidth, slm_div_factor};
}

ConvSubgroupBuf::ConvSubgroupBuf(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                 const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvSubgroupBuf init !\n");
#endif
    mResource.reset(new ConvSubgroupBufResource);
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams                  = conv2dParams;
    mResource->mConv2dCommonParams            = conv2dCommonParams;
    mResource->mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};


    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second; // padY
    mPaddings[1] = padding.first;  // padX

    mResource->mKernelWidth           = conv2dCommonParams->kernelX();
    mResource->mKernelHeight          = conv2dCommonParams->kernelY();
    mResource->mOutputChannel         = conv2dCommonParams->outputCount();
    mResource->mInputChannel          = inputs[0]->channel();

    {
        // create tensor for intel filter
        mResource->mFilter.reset(Tensor::createDevice<float>(std::vector<int>{
            UP_DIV(mResource->mOutputChannel, 16), UP_DIV(mResource->mInputChannel, 16), mResource->mKernelWidth * mResource->mKernelHeight, 16, 16}));
        auto res = mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
        cl_int ret_code;
        if (!res) {
            mValid = false;
            return;
        }
        const float *FilterDataPtr = NULL;
        int weightSize = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        ConvolutionCommon::getConvParameters(&quanCommon, backend, conv2dParams, &FilterDataPtr, &weightSize);
        if (FilterDataPtr != nullptr) {
            std::shared_ptr<Tensor> sourceWeight(
                Tensor::create<float>(std::vector<int>{mResource->mOutputChannel, mResource->mInputChannel, mResource->mKernelWidth, mResource->mKernelHeight},
                                      (void *)FilterDataPtr, Tensor::CAFFE));
            std::shared_ptr<Tensor> destWeight(Tensor::create<float>(std::vector<int>{
                UP_DIV(mResource->mOutputChannel, 16), UP_DIV(mResource->mInputChannel, 16), mResource->mKernelWidth * mResource->mKernelHeight, 16, 16}));

            transformWeight(destWeight.get(), sourceWeight.get());
            auto weightDestSize = destWeight->size();

            auto buffer_size = destWeight->elementSize();
            if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                buffer_size *= sizeof(half_float::half);
            } else {
                buffer_size *= sizeof(float);
            }

            cl::Buffer &weightBuffer = *(cl::Buffer *)mResource->mFilter->buffer().device;

            auto runTime = mOpenCLBackend->getOpenCLRuntime();
            auto queue   = runTime->commandQueue();

            auto weight_ptr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr,
                                                     nullptr, &ret_code);
            if (weight_ptr != nullptr && ret_code == CL_SUCCESS) {
                if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                    for (int i = 0; i < destWeight->elementSize(); i++) {
                        ((half_float::half *)weight_ptr)[i] = (half_float::half)(destWeight->host<float>()[i]);
                    }
                } else {
                    ::memcpy(weight_ptr, destWeight->host<float>(), buffer_size);
                }
            } else {
                MNN_ERROR("Map error weightPtr == nullptr \n");
            }

            queue.enqueueUnmapMemObject(weightBuffer, weight_ptr);
        }
    } 
    {
        int biasSize    = conv2dParams->common()->outputCount();
        int buffer_size = ROUND_UP(biasSize, 16); // pack to 16
        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }

        mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(biasSize, 16)}));
        backend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
        cl::Buffer &biasBuffer = openCLBuffer(mResource->mBias.get());

        cl_int res;
        auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if (biasPtrCL != nullptr && res == CL_SUCCESS) {
            ::memset(biasPtrCL, 0, buffer_size);
            if (nullptr != conv2dParams->bias()) {
                const float *biasDataPtr = conv2dParams->bias()->data();
                if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                    for (int i = 0; i < biasSize; i++) {
                        ((half_float::half *)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                    }
                } else {
                    ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
                }
            }
        } else {
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    }

    if (mResource->mConv2dCommonParams->relu()) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mConv2dCommonParams->relu6()) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvSubgroupBuf init !\n");
#endif
}

void ConvSubgroupBuf::transformWeight(const Tensor *weightDest, const Tensor *source) {
    int co      = source->length(0);
    int ci     = source->length(1);
    int KernelY = source->length(2);
    int KernelX = source->length(3);

    ::memset(weightDest->host<float>(), 0, weightDest->size());

    auto weightPtr      = source->host<float>();
    for (int oz = 0; oz < co; ++oz) {
        auto srcOz = weightPtr + oz * ci * KernelY * KernelX;

        int ozC4 = oz / 16;
        int mx   = oz % 16;

        auto dstOz = weightDest->host<float>() + weightDest->stride(0) * ozC4 + mx;
        for (int sz = 0; sz < ci; ++sz) {
            int szC4         = sz / 16;
            int my           = sz % 16;
            auto srcSz       = srcOz + KernelY * KernelX * sz;
            auto dstSz = dstOz + szC4 * weightDest->stride(1) + my * 16;

            for (int i = 0; i < KernelY * KernelX; ++i) {
                *(dstSz + i * weightDest->stride(2)) = srcSz[i];
            }
        }
    }
}

ConvSubgroupBuf::~ConvSubgroupBuf() {
    // Do nothing
}

ConvSubgroupBuf::ConvSubgroupBuf(std::shared_ptr<ConvSubgroupBufResource> resource, const MNN::Op* op, Backend *backend) : CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool ConvSubgroupBuf::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvSubgroupBuf(mResource, op, bn);
    return true;
}

ErrorCode ConvSubgroupBuf::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvSubgroupBuf onResize !\n");
#endif
    mUnits.clear();
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    int in_c_pack                = TensorUtils::getTensorChannelPack(input);
    int out_c_pack               = TensorUtils::getTensorChannelPack(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    
    int input_width_pad = mResource->mStrides[1] * (8 - 1) + (mResource->mKernelWidth - 1) * mResource->mDilations[1] + 1 + width * mResource->mStrides[1] + mPaddings[1];
    int input_height_pad = inputHeight + 2 * mPaddings[0];
    uint32_t MaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->MaxWorkGroupSize());
    uint32_t MaxThreadsPerDevice = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->MaxThreadsPerDevice());
    bool isSupportedFP16 = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16();

    auto inputpad                      = TensorUtils::getDescribe(input)->mPads;
    auto outputpad                     = TensorUtils::getDescribe(output)->mPads;
    int inputImageShape[2]             = {inputHeight, inputWidth};
    int outputImageShape[2]            = {height, width};
    int kernelShape[2]                 = {mResource->mKernelHeight, mResource->mKernelWidth};
    int strideShape[2]                 = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]                = {mPaddings[0], mPaddings[1]};
    int dilationShape[2]               = {mResource->mDilations[0], mResource->mDilations[1]};
    auto tune_param = GetTuningParams(inputs, outputs, MaxWorkGroupSize, isSupportedFP16, MaxThreadsPerDevice);
    uint32_t blockWidth = tune_param.first;
    uint32_t sub_group_size = 16;
    uint32_t slm_div_factor = tune_param.second;
    uint32_t work_group_size = sub_group_size * slm_div_factor;
    uint32_t feature_block_size = 16;        
    uint32_t input_line_size = strideShape[1] * (blockWidth - 1) + (kernelShape[1] - 1) * dilationShape[1] + 1;
    uint32_t input_block_size = UP_DIV(input_line_size * kernelShape[0] * dilationShape[0], sub_group_size);
    uint32_t x_blocks = UP_DIV(outputImageShape[1], blockWidth);

    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(2), blockWidth) * outputShape.at(1)),
                       static_cast<uint32_t>(ROUND_UP(outputShape.at(3), sub_group_size) * slm_div_factor),
                       static_cast<uint32_t>(outputShape.at(0))};
    mLocalWorkSize  = {1, static_cast<uint32_t>(sub_group_size * slm_div_factor), 1};

    if (in_c_pack == 4) {
         Unit unit;
         mNeedTranse = true;
         if (inputChannels < 16) {
             mSource.reset(Tensor::createDevice<float>(std::vector<int>{inputShape.at(0), input->channel(), inputHeight, inputWidth}, Tensor::CAFFE_C4));
             mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
             mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
             unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("input_transe_buf", "conv_transe_c4_c1", {});
             
             uint32_t mMaxWGS_S = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
             
             mTranseGlobalWorkSize = {static_cast<uint32_t>(inputWidth * inputHeight),
                                      static_cast<uint32_t>(UP_DIV(inputShape.at(3), 4)),
                                      static_cast<uint32_t>(inputShape.at(0))};
             uint32_t idx          = 0;
             unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[0]);
             unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[1]);
             unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[2]);
             unit.kernel->get().setArg(idx++, openCLBuffer(input));
             unit.kernel->get().setArg(idx++, openCLBuffer(mSource.get()));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputWidth));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputHeight));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputChannels));
             unit.kernel->get().setArg(idx++, UP_DIV(inputShape.at(3), 4));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.left));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.right));
             
             mTranseLocalWorkSize = localWS3DDefault(mTranseGlobalWorkSize, mMaxWGS_S, mOpenCLBackend->getOpenCLRuntime(), "conv_transe_c4_c1", unit.kernel).first;
             mOpenCLBackend->recordKernel3d(unit.kernel, mTranseGlobalWorkSize, mTranseLocalWorkSize);
         } else {
             mSource.reset(Tensor::createDevice<float>(std::vector<int>{inputShape.at(0), UP_DIV(input->channel(), 16),inputHeight * inputWidth, 16}, Tensor::CAFFE_C4));
             mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
             mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
             unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("input_transe_buf", "conv_transe_c4_c16", {});
             
             uint32_t mMaxWGS_S = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
             
             mTranseGlobalWorkSize = {static_cast<uint32_t>(inputWidth * inputHeight),
                                      static_cast<uint32_t>(UP_DIV(inputShape.at(3), 4)),
                                      static_cast<uint32_t>(inputShape.at(0))};
             uint32_t idx          = 0;
             unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[0]);
             unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[1]);
             unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[2]);
             unit.kernel->get().setArg(idx++, openCLBuffer(input));
             unit.kernel->get().setArg(idx++, openCLBuffer(mSource.get()));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputWidth));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputHeight));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputChannels));
             unit.kernel->get().setArg(idx++, UP_DIV(inputShape.at(3), 4));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.left));
             unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.right));
             
             mTranseLocalWorkSize = localWS3DDefault(mTranseGlobalWorkSize, mMaxWGS_S, mOpenCLBackend->getOpenCLRuntime(), "conv_transe_c4_c16", unit.kernel).first;
             mOpenCLBackend->recordKernel3d(unit.kernel, mTranseGlobalWorkSize, mTranseLocalWorkSize);
         }
        unit.globalWorkSize = {mTranseGlobalWorkSize[0], mTranseGlobalWorkSize[1], mTranseGlobalWorkSize[2]};
        unit.localWorkSize = {mTranseLocalWorkSize[0], mTranseLocalWorkSize[1], mTranseLocalWorkSize[2]};
        mUnits.emplace_back(unit);
    }
    
    Unit unit;
    if (inputChannels < 16 && in_c_pack == 4) {
         std::set<std::string> buildOptions = mResource->mBuildOptions;
         buildOptions.emplace("-DINPUT_LINE_SIZE=" + std::to_string(input_line_size));
         buildOptions.emplace("-DINPUT_BLOCK_SIZE=" + std::to_string(input_block_size));
         buildOptions.emplace("-DINPUT_CHANNEL=" + std::to_string(inputChannels));
         buildOptions.emplace("-DFILTER_HEIGHT=" + std::to_string(kernelShape[0]));
         buildOptions.emplace("-DFILTER_WIDTH=" + std::to_string(kernelShape[1]));
         buildOptions.emplace("-DDILATION_HEIGHT=" + std::to_string(dilationShape[0]));
         buildOptions.emplace("-DDILATION_WIDTH=" + std::to_string(dilationShape[1]));
         buildOptions.emplace("-DSTRIDE_HEIGHT=" + std::to_string(strideShape[0]));
         buildOptions.emplace("-DSTRIDE_WIDTH=" + std::to_string(strideShape[1]));
         std::string kernelname = "conv_2d_buf_subgroup_c1_c" + std::to_string(out_c_pack) + "_b" + std::to_string(blockWidth);
         unit.kernel  = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_c1_subgroup_buf", kernelname, buildOptions);
    } else {
         std::set<std::string> buildOptions = mResource->mBuildOptions;
         buildOptions.emplace("-DINPUT_LINE_SIZE=" + std::to_string(input_line_size));
         buildOptions.emplace("-DSLM_DIV_FACTOR=" + std::to_string(slm_div_factor));
         buildOptions.emplace("-DWORK_GROUP_SIZE=" + std::to_string(work_group_size));
         buildOptions.emplace("-DIC_BLOCKS=" + std::to_string(UP_DIV(inputChannels, feature_block_size)));
         buildOptions.emplace("-DINPUT_CHANNEL=" + std::to_string(inputChannels));
         buildOptions.emplace("-DFILTER_HEIGHT=" + std::to_string(kernelShape[0]));
         buildOptions.emplace("-DFILTER_WIDTH=" + std::to_string(kernelShape[1]));
         buildOptions.emplace("-DDILATION_HEIGHT=" + std::to_string(dilationShape[0]));
         buildOptions.emplace("-DDILATION_WIDTH=" + std::to_string(dilationShape[1]));
         buildOptions.emplace("-DSTRIDE_HEIGHT=" + std::to_string(strideShape[0]));
         buildOptions.emplace("-DSTRIDE_WIDTH=" + std::to_string(strideShape[1]));
         std::string kernelname = "conv_2d_buf_subgroup_c16_c" + std::to_string(out_c_pack) + "_b" + std::to_string(blockWidth);
         unit.kernel  = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_c16_subgroup_buf", kernelname, buildOptions);
    }
    uint32_t idx    = 0;
    if (mNeedTranse) {
        unit.kernel->get().setArg(idx++, openCLBuffer(mSource.get()));
    } else {
        unit.kernel->get().setArg(idx++, openCLBuffer(input));
    }
    unit.kernel->get().setArg(idx++, openCLBuffer(output));
    unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(mPaddings[1]));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(mPaddings[0]));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputWidth));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputHeight));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(width));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(height));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outChannel));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(x_blocks));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.left));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.right));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputpad.left));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputpad.right));
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvSubgroupBuf onResize !\n");
#endif
    
    mOpenCLBackend->recordKernel3d(unit.kernel , mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
