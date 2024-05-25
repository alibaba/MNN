//
//  ConvWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/ConvWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "math/WingoradGenerater.hpp"
#define UNIT 2
#define INTERP 1
namespace MNN {
namespace OpenCL {
bool ConvWinograd::valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, int maxWidth, int maxHeight, int limit) {
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if(common->kernelX() != common->kernelY()) {
        return false;
    }
    if(common->kernelX() != 3 && common->kernelX() != 5){
        return false;
    }
    
    int ic = input->channel();
    int oc = common->outputCount();
    int ow = output->width();
    int oh =output->height();
    int kh = common->kernelX();
    int wUnit = UP_DIV(ow, UNIT);
    int hUnit = UP_DIV(oh, UNIT);
    int alpha  = kh + UNIT - 1;
    int sourceWidth  = UP_DIV(ic, 4) * 4 * wUnit;
    int sourceHeight = alpha * alpha * hUnit;
    int destWidth  = alpha * alpha * wUnit * 4;
    int destHeight = UP_DIV(ic, 4) * hUnit;

    if(sourceWidth > maxWidth || sourceHeight > maxHeight || destWidth > maxWidth || destHeight > maxHeight){
        return false;
    }
    if(ic >= 32 && oc >= 32){
        return true;
    }
    return ((oc * oh * ow) / (ic * kh) <= 5);
}


ConvWinograd::ConvWinograd(const MNN::Op *op, Backend* backend) : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mResource.reset(new ConvWinoResource);
    auto conv2D  = op->main_as_Convolution2D();
    mResource->mCommon = conv2D->common();
    MNN_ASSERT((3 == mResource->mCommon->kernelY() && 3 == mResource->mCommon->kernelX()) ||
               (5 == mResource->mCommon->kernelX() && 5 == mResource->mCommon->kernelY()));
    MNN_ASSERT(1 == mResource->mCommon->strideX() && 1 == mResource->mCommon->strideY());
    MNN_ASSERT(1 == mResource->mCommon->dilateX() && 1 == mResource->mCommon->dilateY());
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    int ky       = mResource->mCommon->kernelY();
    int kx       = mResource->mCommon->kernelX();

    int weightSize             = 0;
    const float* filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2D->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2D, backend, true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution \n");
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    }

    if (nullptr == filterDataPtr) {
        weightSize    = conv2D->weight()->size();
        filterDataPtr = conv2D->weight()->data();
    }

    int co     = mResource->mCommon->outputCount();
    int ci     = weightSize / co / mResource->mCommon->kernelX() / mResource->mCommon->kernelY();
    auto coC4  = UP_DIV(co, 4);
    auto ciC4  = UP_DIV(ci, 4);
    auto queue = runTime->commandQueue();

    auto imageChannelType = CL_HALF_FLOAT;
    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_High) {
        imageChannelType = CL_FLOAT;
    }
    // Create Image
    {
        mResource->mBias.reset(new cl::Image2D(runTime->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                    UP_DIV(co, 4), 1, 0, nullptr, nullptr));
        
        size_t buffer_size = ALIGN_UP4(co) * sizeof(float);
        std::shared_ptr<cl::Buffer> biasBuffer(
            new cl::Buffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));

        cl_int error;
        auto biasC = queue.enqueueMapBuffer(*biasBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(biasC != nullptr && error == CL_SUCCESS){
            ::memset(biasC, 0, buffer_size);
            ::memcpy(biasC, conv2D->bias()->data(), co * sizeof(float));
        }else{
            MNN_ERROR("Map error biasC == nullptr \n");
        }
        queue.enqueueUnmapMemObject(*biasBuffer, biasC);
        copyBufferToImage(runTime, *biasBuffer, *mResource->mBias, coC4, 1);

        std::shared_ptr<Tensor> sourceWeight(
            Tensor::create<float>(std::vector<int>{co, ci, ky, kx}, (void*)(filterDataPtr), Tensor::CAFFE));

        int unit       = UNIT;
        int kernelSize = kx;
        Math::WinogradGenerater generator(unit, kernelSize, INTERP);
        int alpha       = unit + kernelSize - 1;
        auto weightDest = generator.allocTransformWeight(sourceWeight.get());
        generator.transformWeight(weightDest.get(), sourceWeight.get());
        auto weightDestSize = weightDest->size();
        
        buffer_size = weightDest->elementSize() * sizeof(float);
        cl::Buffer weightBuffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        {
            cl_int error;
            auto weightPtr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
            if(weightPtr != nullptr && error == CL_SUCCESS){
                ::memcpy(weightPtr, weightDest->host<float>(), buffer_size);
            } else{
                MNN_ERROR("Map error weightPtr == nullptr \n");
            }

            queue.enqueueUnmapMemObject(weightBuffer, weightPtr);
        }
        mResource->mWeight.reset(new cl::Image2D(runTime->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                      ciC4 * 4, coC4 * alpha * alpha, 0, nullptr, nullptr));
        copyBufferToImage(runTime, weightBuffer, *mResource->mWeight, ciC4 * 4, coC4 * alpha * alpha);
    }
}

ConvWinograd::ConvWinograd(std::shared_ptr<ConvWinoResource> resource, const MNN::Op* op, Backend *backend) : CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto conv2D  = op->main_as_Convolution2D();
    mResource->mCommon = conv2D->common();
}

bool ConvWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvWinograd(mResource, op, bn);
    return true;
}

ErrorCode ConvWinograd::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    mKernelX    = mResource->mCommon->kernelX();
    mKernelY    = mResource->mCommon->kernelY();
    mStrideX    = mResource->mCommon->strideX();
    mStrideY    = mResource->mCommon->strideY();
    mPadMode    = mResource->mCommon->padMode();
    
    int alpha  = mResource->mCommon->kernelX() + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mResource->mCommon);
    const int padY = pad.second;
    const int padX  = pad.first;
    
    auto runTime = mOpenCLBackend->getOpenCLRuntime();

    auto bn = backend();
    mSource.reset(Tensor::createDevice<float>(
        std::vector<int>{alpha * alpha, input->channel(), hUnit, wUnit}, Tensor::CAFFE_C4));
    mDest.reset(Tensor::createDevice<float>(
        std::vector<int>{UP_DIV(output->channel(), 4), wUnit * 4, hUnit, alpha * alpha}, Tensor::CAFFE_C4));

    bn->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

    auto icC4 = UP_DIV(input->channel(), 4);
    auto ocC4 = UP_DIV(output->channel(), 4);

    uint32_t total_num = input->batch();
    mMaxWGS_S.resize(total_num);
    mMaxWGS_D.resize(total_num);
    mUnits.resize(total_num * 3);
    
    std::set<std::string> basic;
    /*Create Kernel*/
    for(int i = 0; i < input->batch(); i++) {
        char format[20];
        ::memset(format, 0, sizeof(format));
        sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
        auto formatStr = std::string(format);
        mUnits[i * 3].kernel =
            runTime->buildKernel("winogradTransformSource" + formatStr,
                                 "winogradTransformSource", basic);
        mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mUnits[i * 3].kernel));
        {
            std::set<std::string> buildOptions = basic;
            if (mResource->mCommon->relu()) {
                buildOptions.emplace("-DRELU");
            }
            if (mResource->mCommon->relu6()) {
                buildOptions.emplace("-DRELU6");
            }
            mUnits[i * 3 + 2].kernel =
                runTime->buildKernel("winogradTransformDest" + formatStr,
                                     "winogradTransformDest", buildOptions);
            mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mUnits[i * 3 + 2].kernel));
        }
    }
    std::string info = std::to_string(input->channel()) + "_" + std::to_string(output->channel());
    
    mGWS_S.resize(total_num);
    mGWS_D.resize(total_num);
    mGWS_M.resize(total_num);
    mLWS_S.resize(total_num);
    mLWS_D.resize(total_num);
    mLWS_M.resize(total_num);

    for (int b = 0; b < input->batch(); ++b) {
        cl_int ret = CL_SUCCESS;
        ret |= mUnits[b * 3].kernel->get().setArg(0, openCLImage(input));
        ret |= mUnits[b * 3].kernel->get().setArg(1, openCLImage(mSource.get()));
        ret |= mUnits[b * 3].kernel->get().setArg(2, wUnit);
        ret |= mUnits[b * 3].kernel->get().setArg(3, hUnit);
        ret |= mUnits[b * 3].kernel->get().setArg(4, padX);
        ret |= mUnits[b * 3].kernel->get().setArg(5, padY);
        ret |= mUnits[b * 3].kernel->get().setArg(6, input->width());
        ret |= mUnits[b * 3].kernel->get().setArg(7, input->height());
        ret |= mUnits[b * 3].kernel->get().setArg(8, icC4);
        ret |= mUnits[b * 3].kernel->get().setArg(9, b);


        ret |= mUnits[b * 3 + 2].kernel->get().setArg(0, openCLImage(mDest.get()));
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(1, *mResource->mBias);
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(2, openCLImage(output));
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(3, wUnit);
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(4, hUnit);
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(5, output->width());
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(6, output->height());
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(7, ocC4);
        ret |= mUnits[b * 3 + 2].kernel->get().setArg(8, b);
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradExecution");

        /*Source Transform*/
        {
            mGWS_S[b] = {static_cast<uint32_t>(wUnit * hUnit), static_cast<uint32_t>(icC4)};
            std::string kernelName = "winogradTransformSource";
            mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName + info, mUnits[b * 3].kernel).first;
            mOpenCLBackend->recordKernel2d(mUnits[b * 3].kernel, mGWS_S[b], mLWS_S[b]);
            mUnits[b * 3].globalWorkSize = {mGWS_S[b][0], mGWS_S[b][1]};
            mUnits[b * 3].localWorkSize = {mLWS_S[b][0], mLWS_S[b][1]};
        }

        /*MatMul*/
        {
            const int total_kernel                     = 2;
            const std::string kernelName[total_kernel] = {"gemmWinograd", "gemmWinogradW2"};
            int itemW[total_kernel]                    = {4, 8};
            auto gemmHeight = ocC4;
            int actual_kernel = total_kernel;
            
            std::shared_ptr<KernelWrap> kernel[total_kernel];
            std::vector<uint32_t> globalWorkSize[total_kernel];
            std::vector<uint32_t> localWorkSize[total_kernel];
            std::pair<uint32_t, int> min_cost(UINT_MAX, 0); //(min_time, min_index)
            for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
                cl_int ret = CL_SUCCESS;
                kernel[knl_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm", kernelName[knl_idx], basic);
                uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

                globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(wUnit, itemW[knl_idx]) * hUnit), static_cast<uint32_t>(alpha * alpha * ocC4)};
                ret |= kernel[knl_idx]->get().setArg(0, openCLImage(mSource.get()));
                ret |= kernel[knl_idx]->get().setArg(1, *mResource->mWeight);
                ret |= kernel[knl_idx]->get().setArg(2, openCLImage(mDest.get()));
                ret |= kernel[knl_idx]->get().setArg(3, wUnit);
                ret |= kernel[knl_idx]->get().setArg(4, hUnit);
                ret |= kernel[knl_idx]->get().setArg(5, ocC4);
                ret |= kernel[knl_idx]->get().setArg(6, icC4);
                ret |= kernel[knl_idx]->get().setArg(7, alpha*alpha);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradExecution gemm");

                std::pair<std::vector<uint32_t>, uint32_t> retTune;
                retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
                // printf("gemm %d, %d\n", knl_idx, retTune.second);
                if (min_cost.first > retTune.second) {
                    min_cost.first  = retTune.second;
                    min_cost.second = knl_idx;
                    mLWS_M[b]       = {retTune.first[0], retTune.first[1]};
                }
            }
            cl_int ret = CL_SUCCESS;
            int min_index = min_cost.second;
            //printf("gemm min_index = %d  %d\n", min_index, min_cost.first);
            mUnits[b * 3 + 1].kernel = runTime->buildKernel("gemm", kernelName[min_index], basic);
            
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(0, openCLImage(mSource.get()));
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(1, *mResource->mWeight);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(2, openCLImage(mDest.get()));
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(3, wUnit);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(4, hUnit);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(5, ocC4);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(6, icC4);
            ret |= mUnits[b * 3 + 1].kernel->get().setArg(7, alpha*alpha);
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradExecution gemm");
            mGWS_M[b] = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
            mOpenCLBackend->recordKernel2d(mUnits[b * 3 + 1].kernel, mGWS_M[b], mLWS_M[b]);
            mUnits[b * 3 + 1].globalWorkSize = {mGWS_M[b][0], mGWS_M[b][1]};
            mUnits[b * 3 + 1].localWorkSize = {mLWS_M[b][0], mLWS_M[b][1]};
        }

        // Dest Transform
        {
            mGWS_D[b] = {static_cast<uint32_t>(wUnit*hUnit), static_cast<uint32_t>(ocC4)};
            std::string kernelName = "winogradTransformDest";
            mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName + info, mUnits[b * 3 + 2].kernel).first;
            mOpenCLBackend->recordKernel2d(mUnits[b * 3 + 2].kernel, mGWS_D[b], mLWS_D[b]);
            mUnits[b * 3 + 2].globalWorkSize = {mGWS_D[b][0], mGWS_D[b][1]};
            mUnits[b * 3 + 2].localWorkSize = {mLWS_D[b][0], mLWS_D[b][1]};
        }
    }
    
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
