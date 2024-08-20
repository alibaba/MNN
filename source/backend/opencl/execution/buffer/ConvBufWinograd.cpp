//
//  ConvBufWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ConvBufWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "math/WingoradGenerater.hpp"

#define UNIT 2
#define INTERP 1
namespace MNN {
namespace OpenCL {
bool ConvBufWinograd::valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, bool isIntel, int limit) {
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if(common->kernelX() != 3 || common->kernelY() != 3){
        return false;
    }
    if (isIntel) {
        return input->width() * input->height() <= 4096;
    }

    bool valid = input->channel() >= 32 && output->channel() >= 32 && input->width() < output->channel();
    valid = valid || (input->channel() >= 64 && output->channel() >= 64);
    return valid;
}
    
void ConvBufWinograd::convertWeightFormat(cl::Buffer& buffer, const int tileK, const int tileN) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    
    auto icPad  = ROUND_UP(mCi, tileK);
    auto ocPad  = ROUND_UP(mCo, tileN);
    
    auto kernel = runtime->buildKernel("winogradTransform_buf", "winoTransWeightBuf2_3_1", {});
    uint32_t gws[2] = {static_cast<uint32_t>(icPad), static_cast<uint32_t>(ocPad)};
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, buffer);
    ret |= kernel->get().setArg(idx++, openCLBuffer(mResource->mWeight.get()));
    ret |= kernel->get().setArg(idx++, mCi);
    ret |= kernel->get().setArg(idx++, mCo);
    ret |= kernel->get().setArg(idx++, icPad);
    ret |= kernel->get().setArg(idx++, ocPad);

    MNN_CHECK_CL_SUCCESS(ret, "setArg conv-winograd convertWeightFormat");
    const std::vector<uint32_t> lws = {8, 8};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "conv-winograd convertWeightFormat");
    //event.wait();
    return;
}

ConvBufWinograd::ConvBufWinograd(const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    mResource.reset(new ConvBufWinoResource);
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto conv2D  = op->main_as_Convolution2D();
    mResource->mCommon  = conv2D->common();
    MNN_ASSERT((3 == mResource->mCommon->kernelY() && 3 == mResource->mCommon->kernelX()));
    MNN_ASSERT(1 == mResource->mCommon->strideX() && 1 == mResource->mCommon->strideY());
    MNN_ASSERT(1 == mResource->mCommon->dilateX() && 1 == mResource->mCommon->dilateY());
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    int ky       = mResource->mCommon->kernelY();
    int kx       = mResource->mCommon->kernelX();

    int weightSize             = 0;
    const float* filterDataPtr = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, conv2D, &filterDataPtr, &weightSize);

    mCo     = mResource->mCommon->outputCount();
    mCi     = weightSize / mCo / mResource->mCommon->kernelX() / mResource->mCommon->kernelY();
    auto ocC4  = UP_DIV(mCo, 4);
    auto icC4  = UP_DIV(mCi, 4);
    auto queue = runTime->commandQueue();

    auto imageChannelType = CL_HALF_FLOAT;
    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_High) {
        imageChannelType = CL_FLOAT;
    }
    // Create Buffer Object
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    mResource->mUseSubgroup = runTime->isSupportedIntelSubgroup();
    if (mResource->mUseSubgroup) {
        // create buffer for intel subgroup
        cl_int ret_code;
        size_t bias_element = ALIGN_UP4(mCo);
        size_t buffer_size;
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size = bias_element * sizeof(half_float::half);
        } else {
            buffer_size = bias_element * sizeof(float);
        }
        
        mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, (int)ALIGN_UP4(mCo)}));
        mOpenCLBackend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
        cl::Buffer &bias_buffer = *(cl::Buffer *)mResource->mBias->buffer().device;

        auto bias_ptr = queue.enqueueMapBuffer(bias_buffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if(bias_ptr == nullptr || ret_code) {
            MNN_ERROR("clBuffer map error!\n");
        }
        ::memset(bias_ptr, 0, buffer_size);
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            for(int i=0; i<mCo; i++) {
                ((half_float::half *)bias_ptr)[i] = (half_float::half)conv2D->bias()->data()[i];
            }
        } else {
            ::memcpy(bias_ptr, conv2D->bias()->data(), mCo*sizeof(float));
        }
        queue.enqueueUnmapMemObject(bias_buffer, bias_ptr);


        auto ocC16 = UP_DIV(mCo, 16);
        auto icC16 = UP_DIV(mCi, 16);
        std::shared_ptr<Tensor> sourceWeight(
            Tensor::create<float>(std::vector<int>{mCo, mCi, ky, kx}, (void*)(filterDataPtr), Tensor::CAFFE));

        int unit       = UNIT;
        int kernelSize = kx;
        Math::WinogradGenerater generator(unit, kernelSize, INTERP);
        int alpha = unit + kernelSize - 1;
        auto weightDest = generator.allocTransformWeight(sourceWeight.get(), 16, 16);
        generator.transformWeight(weightDest.get(), sourceWeight.get());
        auto weightDestSize = weightDest->size();

        buffer_size = weightDest->elementSize();
        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }

        mResource->mWeight.reset(Tensor::createDevice<float>({alpha * alpha, ocC16, icC16, 16 * 16}, Tensor::CAFFE_C4)); // NHWC
        mOpenCLBackend->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);

        cl::Buffer& weightBuffer = *(cl::Buffer*)mResource->mWeight->buffer().device;

        auto weight_ptr =
            queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if (weight_ptr != nullptr && ret_code == CL_SUCCESS) {
            if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                for (int i = 0; i < weightDest->elementSize(); i++) {
                    ((half_float::half*)weight_ptr)[i] = (half_float::half)(weightDest->host<float>()[i]);
                }
            } else {
                ::memcpy(weight_ptr, weightDest->host<float>(), buffer_size);
            }
        } else {
            MNN_ERROR("Map error weightPtr == nullptr \n");
        }

        queue.enqueueUnmapMemObject(weightBuffer, weight_ptr);
    }else
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */    
    {

        cl_int ret_code;
        size_t bias_element = ALIGN_UP4(mCo);
        size_t buffer_size;
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size = bias_element * sizeof(half_float::half);
        } else {
            buffer_size = bias_element * sizeof(float);
        }
        
        mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, (int)ALIGN_UP4(mCo)}));
        mOpenCLBackend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
        cl::Buffer &bias_buffer = *(cl::Buffer *)mResource->mBias->buffer().device;
        
        auto bias_ptr = queue.enqueueMapBuffer(bias_buffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if(bias_ptr == nullptr || ret_code) {
            MNN_ERROR("clBuffer map error!\n");
        }
        ::memset(bias_ptr, 0, buffer_size);
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            for(int i=0; i<mCo; i++) {
                ((half_float::half *)bias_ptr)[i] = (half_float::half)conv2D->bias()->data()[i];
            }
        } else {
            ::memcpy(bias_ptr, conv2D->bias()->data(), mCo*sizeof(float));
        }
        queue.enqueueUnmapMemObject(bias_buffer, bias_ptr);
        
        int unit       = UNIT;
        int kernelSize = kx;
        int alpha       = unit + kernelSize - 1;
        
        int tileK = 4;
        int tileN = 32;

        std::shared_ptr<Tensor> tmpFilterTensor;
        tmpFilterTensor.reset(Tensor::createDevice<int32_t>({mCo * mCi * ky * kx}));
        mOpenCLBackend->onAcquireBuffer(tmpFilterTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(tmpFilterTensor.get(), Backend::DYNAMIC);

        mResource->mWeight.reset(Tensor::createDevice<float>({alpha * alpha * ROUND_UP(mCo, tileN) * ROUND_UP(mCi, tileK)}));//NHWC
        mOpenCLBackend->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
        
        buffer_size = mCo * mCi * ky * kx * sizeof(float);
        cl::Buffer& weightBufferCL = openCLBuffer(tmpFilterTensor.get());
        
        cl_int res;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(weightBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(ptrCL != nullptr && res == CL_SUCCESS) {
            ::memcpy(ptrCL, filterDataPtr, buffer_size);
        }else{
            MNN_ERROR("Map weightBufferCL error:%d, ptrCL == nullptr \n", res);
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(weightBufferCL, ptrCL);
        
        convertWeightFormat(weightBufferCL, tileK, tileN);
    }
}

ConvBufWinograd::~ConvBufWinograd() {
    // Do nothing
}

ConvBufWinograd::ConvBufWinograd(std::shared_ptr<ConvBufWinoResource> resource, const MNN::Op* op, Backend *backend) : CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto conv2D  = op->main_as_Convolution2D();
    mResource->mCommon = conv2D->common();
}

bool ConvBufWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvBufWinograd(mResource, op, bn);
    return true;
}
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
ErrorCode ConvBufWinograd::SubgroupOnResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto input  = inputs[0];
    auto output = outputs[0];
    int alpha  = mKernelX + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    auto pad = ConvolutionCommon::convolutionPad(input, output, mResource->mCommon);
    int padY = pad.second;
    int padX = pad.first;
    uint32_t total_num = input->batch();
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    std::string info = std::to_string(input->channel()) + "_" + std::to_string(output->channel());
    mSource.reset(Tensor::createDevice<float>(std::vector<int>{alpha * alpha, UP_DIV(input->channel(), 16), ROUND_UP(wUnit * hUnit, 8), 16}, Tensor::CAFFE_C4));
    mDest.reset(Tensor::createDevice<float>(std::vector<int>{alpha * alpha, UP_DIV(output->channel(), 16), ROUND_UP(wUnit * hUnit, 8), 16}, Tensor::CAFFE_C4));
    
    mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);
    
    auto icC4  = UP_DIV(input->channel(), 4);
    auto icC16 = UP_DIV(input->channel(), 16);
    auto ocC4  = UP_DIV(output->channel(), 4);
    auto ocC16     = UP_DIV(output->channel(), 16);
    auto inputpad  = TensorUtils::getDescribe(input)->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
    int in_c_pack  = TensorUtils::getTensorChannelPack(input);
    int out_c_pack = TensorUtils::getTensorChannelPack(output);
    
    std::set<std::string> basic;
    std::string srcTranseKernelname = "_c16_c16";
    std::string dstTranseKernelname = "_c16_c16";
    if (in_c_pack == 4) {
        srcTranseKernelname = "_c4_c16";
    }
    if (out_c_pack == 4) {
        dstTranseKernelname = "_c16_c4";
    }
    /*Create Kernel*/
    for (int i = 0; i < total_num; i++) {
        char format[20];
        ::memset(format, 0, sizeof(format));
        sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
        auto formatStr = std::string(format);
        mUnits[i * 3].kernel = runTime->buildKernel("winogradTransform_subgroup_buf", "winoTransSrcBuf" + formatStr + srcTranseKernelname, basic);
        mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mUnits[i * 3].kernel));
        {
            std::set<std::string> buildOptions = basic;
            if (mResource->mCommon->relu()) {
                buildOptions.emplace("-DRELU");
            }
            if (mResource->mCommon->relu6()) {
                buildOptions.emplace("-DRELU6");
            }
            if (output->width() % 2 != 0) {
                buildOptions.emplace("-DOUTPUT_LEFTOVERS");
            }
            mUnits[i * 3 + 2].kernel = runTime->buildKernel("winogradTransform_subgroup_buf", "winoTransDstBuf" + formatStr + dstTranseKernelname, buildOptions);
            mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mUnits[i * 3 + 2].kernel));
        }
    }
    
    for (int b = 0; b < input->batch(); ++b) {
        int hCount = hUnit;
        int wCount = wUnit;
        int width_pack = ROUND_UP(hCount * wCount, 8);
        
        // Source Transform
        {
            mGWS_S[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(input->channel())};
            int index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mUnits[b * 3].kernel->get().setArg(index++, mGWS_S[b][0]);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, mGWS_S[b][1]);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, openCLBuffer(input));
            ret |= mUnits[b * 3].kernel->get().setArg(index++, openCLBuffer(mSource.get()));
            ret |= mUnits[b * 3].kernel->get().setArg(index++, wCount);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, hCount);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, padX);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, padY);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, input->width());
            ret |= mUnits[b * 3].kernel->get().setArg(index++, input->height());
            ret |= mUnits[b * 3].kernel->get().setArg(index++, icC4);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, icC16);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, width_pack);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, b);
            ret |= mUnits[b * 3].kernel->get().setArg(index++, static_cast<uint32_t>(inputpad.left));
            ret |= mUnits[b * 3].kernel->get().setArg(index++, static_cast<uint32_t>(inputpad.right));
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf Source Trans");
            
            if (in_c_pack == 4) {
                mGWS_S[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(ROUND_UP(input->channel(), 16) / 4)};
                std::string kernelName = srcTranseKernelname + "_" + std::to_string(mGWS_S[b][0]) + "_" + std::to_string(mGWS_S[b][1]);
                mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName + info, mUnits[b * 3].kernel).first;
            } else {
                mLWS_S[b] = {1, 16};
            }
            mOpenCLBackend->recordKernel2d(mUnits[b * 3].kernel, mGWS_S[b], mLWS_S[b]);
            mUnits[b * 3].globalWorkSize = {mGWS_S[b][0], mGWS_S[b][1]};
            mUnits[b * 3].localWorkSize = {mLWS_S[b][0], mLWS_S[b][1]};
        }
        
        // MatMul
        {
            auto gemmHeight = ocC4;
            auto gemmWidth  = wCount * hCount;
            
            mGWS_M[b] = {static_cast<uint32_t>(UP_DIV(gemmWidth, 8)), static_cast<uint32_t>(ROUND_UP(output->channel(), 16)), static_cast<uint32_t>(alpha * alpha)};
            mLWS_M[b] = {1, 16, 1};
            std::set<std::string> buildOptions = basic;
            mUnits[b * 3 + 1].kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("winogradTransform_subgroup_buf", "gemm_buf_intel", buildOptions);
            
            int index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, openCLBuffer(mSource.get()));
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, openCLBuffer(mResource->mWeight.get()));
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, openCLBuffer(mDest.get()));
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, width_pack);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, ocC16);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, icC16);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(index++, alpha * alpha);
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf MatMul");
            mOpenCLBackend->recordKernel3d(mUnits[b * 3 + 1].kernel, mGWS_M[b], mLWS_M[b]);
            mUnits[b * 3 + 1].globalWorkSize = {mGWS_M[b][0], mGWS_M[b][1], mGWS_M[b][2]};
            mUnits[b * 3 + 1].localWorkSize = {mLWS_M[b][0], mLWS_M[b][1], mLWS_M[b][2]};
        }
        
        // Dest Transform
        {
            mGWS_D[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(output->channel())};
            
            int index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, mGWS_D[b][0]);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, mGWS_D[b][1]);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, openCLBuffer(mDest.get()));
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, openCLBuffer(mResource->mBias.get()));
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, openCLBuffer(output));
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, wCount);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, hCount);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, output->width());
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, output->height());
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, ocC4);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, ocC16);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, width_pack);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, b);
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.left));
            ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.right));
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf Dest Trans");
            
            if (out_c_pack == 4) {
                mGWS_D[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(ocC4)};
                std::string kernelName = dstTranseKernelname + "_" + std::to_string(mGWS_D[b][0]) + "_" + std::to_string(mGWS_D[b][1]);
                mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName + info, mUnits[b * 3 + 2].kernel).first;
            } else {
                mLWS_D[b] = {1, 16};
            }
            mOpenCLBackend->recordKernel2d(mUnits[b * 3 + 2].kernel, mGWS_D[b], mLWS_D[b]);
            mUnits[b * 3 + 2].globalWorkSize = {mGWS_D[b][0], mGWS_D[b][1]};
            mUnits[b * 3 + 2].localWorkSize = {mLWS_D[b][0], mLWS_D[b][1]};
        }
    }
    return NO_ERROR;
}
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */

ErrorCode ConvBufWinograd::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    mKernelX    = mResource->mCommon->kernelX();
    mKernelY    = mResource->mCommon->kernelY();
    mStrideX    = mResource->mCommon->strideX();
    mStrideY    = mResource->mCommon->strideY();
    
    int alpha  = mKernelX + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    
    auto pad = ConvolutionCommon::convolutionPad(input, output, mResource->mCommon);
    int padY = pad.second;
    int padX = pad.first;
    
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    std::string info = std::to_string(input->channel()) + "_" + std::to_string(output->channel());
    
    uint32_t total_num = input->batch();
    mUnits.resize(total_num * 3);
    mMaxWGS_S.resize(total_num);
    mMaxWGS_D.resize(total_num);
    mMaxWGS_M.resize(total_num);
    
    mGWS_S.resize(total_num);
    mGWS_D.resize(total_num);
    mGWS_M.resize(total_num);
    mLWS_S.resize(total_num);
    mLWS_D.resize(total_num);
    mLWS_M.resize(total_num);

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    if (mResource->mUseSubgroup) {
        return SubgroupOnResize(inputs, outputs);
    } else
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */    
    {
	int tileM = 16;
        int tileN = 32;
        int tileK = 4;
        mSource.reset(Tensor::createDevice<float>(
            std::vector<int>{alpha * alpha * ROUND_UP(input->channel(), tileK) * ROUND_UP(wUnit * hUnit, tileM)}));
        mDest.reset(Tensor::createDevice<float>(
            std::vector<int>{alpha * alpha * ROUND_UP(wUnit * hUnit, tileM) * ROUND_UP(output->channel(), tileN)}));

        mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

        auto icC4 = UP_DIV(input->channel(), 4);
        auto ocC4 = UP_DIV(output->channel(), 4);

        std::set<std::string> basic;
        /*Create Kernel*/
        for (int i = 0; i < total_num; i++) {
            char format[20];
            ::memset(format, 0, sizeof(format));
            sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
            auto formatStr      = std::string(format);
            mUnits[i * 3].kernel = runTime->buildKernel("winogradTransform_buf", "winoTransSrcBuf" + formatStr, basic);
            mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mUnits[i * 3].kernel));
            {
                std::set<std::string> buildOptions = basic;
                if (mResource->mCommon->relu()) {
                    buildOptions.emplace("-DRELU");
                }
                if (mResource->mCommon->relu6()) {
                    buildOptions.emplace("-DRELU6");
                }
                mUnits[i * 3 + 2].kernel = runTime->buildKernel("winogradTransform_buf", "winoTransDstBuf" + formatStr, buildOptions);
                mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mUnits[i * 3 + 2].kernel));
            }
        }

        int hCount = hUnit;
        int wCount = wUnit;
        int M_pack = ROUND_UP(wCount * hCount, tileM);
        int K_pack = ROUND_UP(input->channel(), tileK);
        int N_pack = ROUND_UP(output->channel(), tileN);
        for (int b = 0; b < input->batch(); ++b) {

            // Source Transform
            {
                mGWS_S[b] = {static_cast<uint32_t>(M_pack), static_cast<uint32_t>(UP_DIV(K_pack, 4))};
                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mUnits[b * 3].kernel->get().setArg(index++, mGWS_S[b][0]);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, mGWS_S[b][1]);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, openCLBuffer(input));
                ret |= mUnits[b * 3].kernel->get().setArg(index++, openCLBuffer(mSource.get()));
                ret |= mUnits[b * 3].kernel->get().setArg(index++, wCount);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, hCount);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, padX);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, padY);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, input->width());
                ret |= mUnits[b * 3].kernel->get().setArg(index++, input->height());
                ret |= mUnits[b * 3].kernel->get().setArg(index++, icC4);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, M_pack);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, K_pack);
                ret |= mUnits[b * 3].kernel->get().setArg(index++, b);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf Source Trans");

                std::string kernelName = "winoTransSrcBuf";
                mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName + info, mUnits[b * 3].kernel).first;
                mOpenCLBackend->recordKernel2d(mUnits[b * 3].kernel, mGWS_S[b], mLWS_S[b]);
                mUnits[b * 3].globalWorkSize = {mGWS_S[b][0], mGWS_S[b][1]};
                mUnits[b * 3].localWorkSize = {mLWS_S[b][0], mLWS_S[b][1]};
            }

            // MatMul
            
            {
                int loop = alpha * alpha;
                int e_pack = ROUND_UP(wCount * hCount, tileM);
                int l_pack = ROUND_UP(input->channel(), tileK);
                int h_pack = ROUND_UP(output->channel(), tileN);

                std::set<std::string> buildOptions;
                uint32_t layout = 4;
                auto param = getGemmParams({(uint32_t)e_pack, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)0}, {openCLBuffer(mSource.get()), openCLBuffer(mResource->mWeight.get()), openCLBuffer(mDest.get())}, mOpenCLBackend->getOpenCLRuntime());

                int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
                buildOptions.emplace("-DKWG=" + std::to_string(KWG));
                buildOptions.emplace("-DKWI=" + std::to_string(KWI));
                buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
                buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
                buildOptions.emplace("-DMWG=" + std::to_string(MWG));
                buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
                buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
                buildOptions.emplace("-DNWG=" + std::to_string(NWG));
                buildOptions.emplace("-DSA=" + std::to_string(SA));
                buildOptions.emplace("-DSB=" + std::to_string(SB));
                buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
                buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
                buildOptions.emplace("-DVWM=" + std::to_string(VWM));
                buildOptions.emplace("-DVWN=" + std::to_string(VWN));
                if(layout >= 4) {
                    buildOptions.emplace("-DOUTPUTMN");
                }
                
                int tileM = MWG;
                int tileN = NWG;
                int localM = MDIMC;
                int localN = NDIMC;
                
                if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
                    buildOptions.emplace("-DUSE_CL_MAD=1");
                    buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
                }
                
                mUnits[b * 3 + 1].kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
                
                int out_per_thread_m = tileM / localM;
                int out_per_thread_n = tileN / localN;
                
                mGWS_M[b] = {static_cast<uint32_t>(e_pack/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
                mLWS_M[b] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
                
                float alpha = 1.0f;
                float beta = 0.0f;
                int batch_offset_a = e_pack * l_pack;
                int batch_offset_b = h_pack * l_pack;
                int batch_offset_c = e_pack * h_pack;
                
                int idx            = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, static_cast<int>(e_pack));
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, static_cast<int>(h_pack));
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, static_cast<int>(l_pack));
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, alpha);
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, beta);
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, openCLBuffer(mSource.get()));
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, batch_offset_a);
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, openCLBuffer(mResource->mWeight.get()));
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, batch_offset_b);
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, openCLBuffer(mDest.get()));
                ret |= mUnits[b * 3 + 1].kernel->get().setArg(idx++, batch_offset_c);
                MNN_CHECK_CL_SUCCESS(ret, "setArg Winograd batchmatmul Kernel");
                
                mOpenCLBackend->recordKernel3d(mUnits[b * 3 + 1].kernel, mGWS_M[b], mLWS_M[b]);
                mUnits[b * 3 + 1].globalWorkSize = {mGWS_M[b][0], mGWS_M[b][1], mGWS_M[b][2]};
                mUnits[b * 3 + 1].localWorkSize = {mLWS_M[b][0], mLWS_M[b][1], mLWS_M[b][2]};
            }

            // Dest Transform
            {
                mGWS_D[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(ocC4)};

                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, mGWS_D[b][0]);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, mGWS_D[b][1]);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, openCLBuffer(mDest.get()));
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, openCLBuffer(mResource->mBias.get()));
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, openCLBuffer(output));
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, wCount);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, hCount);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, output->width());
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, output->height());
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, ocC4);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, M_pack);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, N_pack);
                ret |= mUnits[b * 3 + 2].kernel->get().setArg(index++, b);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf Dest Trans");
                
                std::string kernelName = "winoTransDstBuf";
                mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName + info, mUnits[b * 3 + 2].kernel).first;
                mOpenCLBackend->recordKernel2d(mUnits[b * 3 + 2].kernel, mGWS_D[b], mLWS_D[b]);
                mUnits[b * 3 + 2].globalWorkSize = {mGWS_D[b][0], mGWS_D[b][1]};
                mUnits[b * 3 + 2].localWorkSize = {mLWS_D[b][0], mLWS_D[b][1]};
            }
        }
    }
    
    return NO_ERROR;
}
    
ErrorCode ConvBufWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#else
    if(openCLBackend->isUseRecordQueue()){
        openCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    auto res = CL_SUCCESS;
    for (auto &unit : mUnits) {
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize,
                                                    nullptr,
                                                    &event);
        std::string name = "Conv-winograd";

        if(idx % 3 == 1) {
            name += "-batchgemm";
        } else {
            name += "-rearrange";
        }
        auto wUnit = UP_DIV(outputs[0]->width(), UNIT);
        auto hUnit = UP_DIV(outputs[0]->height(), UNIT);
        auto icC4 = ROUND_UP(inputs[0]->channel(), 4);

        auto ocC4 = ROUND_UP(outputs[0]->channel(), 4);
        int alpha  = mKernelX + UNIT - 1;

        auto gemmHeight = ocC4;
        auto gemmWidth  = ROUND_UP(wUnit * hUnit, 4);
        
        std::string b = std::to_string(alpha*alpha);
        std::string m = std::to_string(gemmWidth);
        std::string n = std::to_string(gemmHeight);
        std::string k = std::to_string(icC4);
        std::string total = std::to_string(1.0 / 1000000 * alpha*alpha * gemmWidth * gemmHeight * icC4);
        name += "-b" + b + "m" + m + "n" + n + "k" + k + "-total:" + total + "*10^6";
        runtime->pushEvent({name.c_str(), event});
        idx++;
    #else
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize);
    #endif
        MNN_CHECK_CL_SUCCESS(res, "Conv-Winograd execute");
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
