//
//  ConvolutionWinograd.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionWinograd.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/WingoradGenerater.hpp"
#include <MNN/AutoTime.hpp>
#include "common/MemoryFormater.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2
constexpr int FULSE_THRESHHOLD_NUMERATOR = 8;
constexpr int FULSE_THRESHHOLD_DENOMINATOR = 10;

using namespace MNN::Math;

//#define MNN_WINOGRAD_PRINT_REDUCE_RATE
//#define MNN_WINO_TRANFORM_TEST_CLOSE
namespace MNN {
ConvolutionWinograd::ConvolutionWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                         Backend *b, const float *originWeight, size_t originWeightSize,
                                         const float *bias, size_t biasSize, int unit)
    : MNN::CPUConvolution(convOp, b) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;
    mResource.reset(new Resource);
    mResource->backend = b;
    if (!mResource->copyBiasAlign(bias, biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    MNN_ASSERT(mCommon->kernelX() == mCommon->kernelY());

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    auto kernelSize = mCommon->kernelY();
    WinogradGenerater generator(unit, kernelSize, 1, true);

    int ePack, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    int alpha        = unit + kernelSize - 1;
    int alpha2       = alpha * alpha;
    mSourceTransform = core->chooseWinoSourceTransform(alpha, alpha);
    mDestTransform   = core->chooseWinoDestTransform(alpha, unit);
    mSourceTransformPack = core->chooseWinoSourceTransformPack(alpha, alpha, ePack, lPack, pack);
    int srcCount                       = input->channel();
    int outputCount                    = output->channel();
    auto ic4 = UP_DIV(srcCount, pack);
    auto oc4 = UP_DIV(outputCount, pack);
    mTempBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack, ic4 + oc4, pack * alpha2, bytes}));
    // mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, alpha2, pack, bytes}));
    // mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack * UP_DIV(srcCount, lPack) * lPack, bytes}));

    mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, (1 + ic4 * ePack), alpha2, pack, bytes})); // 1 means original small buffer of alpha2 * pack.
    mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, alpha, ePack * UP_DIV(srcCount, pack) * pack, bytes}));


    mA = generator.A();
    mB = generator.B();


    // Transform Kernel
    auto G = generator.G();
    // replace Tensor::createDevice by Tensor::create and allocTransformWeight's alloc=true to avoid malloc by onAcquireBuffer
    std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)originWeight, Tensor::CAFFE));
    auto tempWeight = generator.allocTransformWeight(sourceWeight.get(), lPack, hPack, true);

    auto shape = tempWeight->shape();
    shape.push_back(bytes);
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(shape));
    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    generator.transformWeight(tempWeight.get(), sourceWeight.get(), true);
    if (bytes != 4) {
        core->MNNFp32ToLowp(tempWeight->host<float>(), mResource->mWeight->host<int16_t>(), tempWeight->elementSize());
    } else {
        ::memcpy(mResource->mWeight->host<float>(), tempWeight->host<float>(), tempWeight->size());
    }

    mPostParameters = getPostParameters();
}
ConvolutionWinograd::~ConvolutionWinograd() {
    // Do nothing
}
bool ConvolutionWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvolutionWinograd(mResource, op->main_as_Convolution2D()->common(), bn);
    dstExe->mA = mA;
    dstExe->mB = mB;
    dstExe->mTempBuffer.reset(Tensor::createDevice<uint8_t>(mTempBuffer->shape()));
    dstExe->mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>(mTransformMidBuffer->shape()));
    dstExe->mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>(mGemmMidBuffer->shape()));
    dstExe->mSourceTransform = mSourceTransform;
    dstExe->mDestTransform = mDestTransform;
    dstExe->mSourceTransformPack = mSourceTransformPack;
    dstExe->mPostParameters = mPostParameters;
    *dst = dstExe;
    return true;
}

ErrorCode ConvolutionWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;

    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dstUnit = mA->length(1); // m
    auto srcUnit = mA->length(0); // n
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    auto srcUnit2 = srcUnit * srcUnit;
    auto alphaXStride = srcUnit * ePack * pack;
    auto IC4alpha2Stride = srcUnit2 * ePack * pack;

    int ow   = output->width();
    int oh   = output->height();
    int iw   = input->width();
    int ih   = input->height();
    int ic_4 = UP_DIV(input->channel(), pack);
    int dc_4 = UP_DIV(output->channel(), pack);
    int batch = input->batch();
    // MNN_PRINT("%d, %d\n", srcUnit, dstUnit);

    int padY = mPadY;
    int padX = mPadX;

    auto wUnit = UP_DIV(ow, dstUnit); // ow / m
    auto hUnit = UP_DIV(oh, dstUnit); // oh / m

    auto totalCount   = wUnit * hUnit * batch;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int tileCount    = UP_DIV(totalCount, ePack);
    int eRemain = totalCount % ePack;
    threadNumber     = std::min(threadNumber, tileCount);
    std::vector<size_t> parameters(6);
    parameters[0] = eRemain * bytes;
    parameters[1] = input->channel();
    parameters[2] = output->channel();
    parameters[3] = ePack * pack * bytes;
    parameters[4] = 0;
    parameters[5] = 0;

    std::vector<size_t> parametersRemain = parameters;
    parametersRemain[3] = eRemain * pack * bytes;

    auto inputOrigin = input->host<uint8_t>();
    auto outputOrigin = output->host<uint8_t>();
    auto srcOrigin = inputOrigin;
    auto dstOrigin = outputOrigin;
    auto midBuffer0Bytes = srcUnit2 * pack * bytes;

    bool allow_x86_bf16_winograd = true;
#ifdef MNN_USE_SSE
    allow_x86_bf16_winograd = bytes != 2; // only bf16 has length of 2 byte on x86. fp16 dosnot exist.
#endif

    // using ElementType = int16_t;
    // MNN_PRINT("winograd: this:%p, n:%d, ih:%d, iw:%d, ic:%d, oh:%d, ow:%d, oc:%d, kh:%d, kw:%d, totalCount:%d, srcUnit:%d, dstUnit:%d, ePack:%d, pack:%d, bytes:%d\n",
    //     this, batch, ih, iw, input->channel(), oh, ow, output->channel(), mCommon->kernelX(), mCommon->kernelY(), totalCount, srcUnit, dstUnit, ePack, pack, bytes);
    // MNN_PRINT("origin data matrix:\n");
    // formatMatrix((const ElementType*)srcOrigin, {ic_4, batch*ih, iw, pack});

    auto weight    = mResource->mWeight->host<uint8_t>();
    auto bias      = mResource->mBias->host<uint8_t>();
    auto tFunction = [&](int tId) {
        auto _srcOrigin = mTempBuffer->host<uint8_t>() + tId * mTempBuffer->stride(0);
        auto gemmBuffer = (mGemmMidBuffer->host<uint8_t>() + tId * mGemmMidBuffer->stride(0));
        auto midBuffer0 = mTransformMidBuffer->host<uint8_t>() + tId * mTransformMidBuffer->stride(0);
        auto midBuffer1 = midBuffer0 + midBuffer0Bytes;
        for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
            int xIndex  = (int)tIndex * ePack;
            int xReamin = totalCount - xIndex;
            int xC      = xReamin > ePack ? ePack : xReamin;
            const bool fuseTransformPack = (xC * FULSE_THRESHHOLD_DENOMINATOR > FULSE_THRESHHOLD_NUMERATOR * ePack) && allow_x86_bf16_winograd && nullptr != mSourceTransformPack;
            // const bool fuseTransformPack = false;

            // Timer timer;
            // uint64_t durationSourceTrans1 = 0;
            // uint64_t durationSourceTrans2 = 0;
            // uint64_t durationMul = 0;
            // uint64_t packATime = 0;
            // uint64_t durationDestTrans1 = 0;
            // uint64_t durationDestTrans2 = 0;

            /*Source Transform Begin*/
#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
            {
                int sourceZStep = iw * ih * batch * pack;
                int oyBegin = xIndex / wUnit;
                int oxBegin = xIndex % wUnit;
                int oyEnd = (xIndex + xC-1) / wUnit;
                int remain = xC;
                int destSOffset = 0;
                if (fuseTransformPack) {
                    for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                        int hIndex = hbIndex % hUnit;
                        int bIndex = hbIndex / hUnit;
                        int step = std::min(wUnit - oxBegin, remain);
                        int srcY  = hIndex * dstUnit - padY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY;
                        int sy    = ALIMAX(0, srcY) - srcY;
                        for (int si=0; si<step; ++si) {
                            auto wIndex = si + oxBegin;
                            int srcX  = wIndex * dstUnit - padX;
                            int sx    = ALIMAX(0, srcX) - srcX;
                            int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                            int count = pack * (ex - sx);

                            auto srcStart = srcOrigin + (srcX + srcY * iw + bIndex * iw * ih) * pack * bytes;
                            // MNN_PRINT("\nxIndex:%d, xC:%d, alphaXStride:%d, srcUnit:%d, destUnit:%d, hUnit:%d, wUnit:%d, srcY:%d, hStart:%d ,hEnd:%d, wStart:%d, wEnd:%d, i_oh:%d, i_ow:%d, srcOffset:%ld, destSOffset:%d\n",
                            //     xIndex, xC, alphaXStride, srcUnit, dstUnit, hUnit, wUnit, srcY, sy, ey, sx, ey, hIndex - oyBegin, si, (srcStart - srcOrigin)/bytes, (destSOffset)/bytes);
                            // timer.reset();
                            auto midBuffer1Offset = midBuffer1 + destSOffset;

                            if (ex - sx == srcUnit && ey - sy == srcUnit) {
                                for (int z = 0; z < ic_4; ++z) {
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    // Transform
                                    // MNN_PRINT("z:%d, srcOffset:%ld, destSOffset:%ld, \n", z, ((unsigned const char*)srcZ - srcOrigin)/bytes, ((unsigned const char*)midBuffer1Offset - midBuffer1)/bytes);
                                    // MNN_PRINT("winograd source sub matrix:\n");
                                    // formatMatrix((const float*)srcZ, {srcUnit, 4});
                                    for (int i = 0; i < srcUnit; ++i) { // i_Nh
                                        auto srcFloatPtr = (const float*)(srcZ + i * iw * pack * bytes);
                                        auto dstFloatPtr = (float*)(midBuffer1Offset + i * ePack * pack * bytes);
                                        mSourceTransform(srcFloatPtr, dstFloatPtr, pack, alphaXStride); // tranform srcUnit*4 elements in one time
                                        // MNN_PRINT("z:%d, 1 stage i_Nh:%d th srcOffset:%ld, destOffset:%ld, \n", z, i, ((unsigned const char*)srcFloatPtr - srcOrigin)/bytes, ((unsigned const char*)dstFloatPtr - midBuffer1)/bytes);
                                        // MNN_PRINT("winograd source sub matrix:\n");
                                        // formatMatrix(srcFloatPtr, {srcUnit, 4});
                                    }
                                    midBuffer1Offset += IC4alpha2Stride * bytes;
                                }
                            } else {
                                for (int z = 0; z < ic_4; ++z) {
                                    // Extract
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    ::memset(midBuffer0, 0, midBuffer0Bytes);
                                    if (count > 0) {
                                        for (int yy = sy; yy < ey; ++yy) {
                                            auto dst_yy = midBuffer0 + (yy * srcUnit + sx) * pack * bytes;
                                            auto src_yy = srcZ + (iw * yy + sx) * pack * bytes;
                                            ::memcpy(dst_yy, src_yy, count * bytes);
                                        }
                                    }
                                    // Transform
                                    for (int i = 0; i < srcUnit; ++i) {
                                        auto srcFloatPtr = (const float*)(midBuffer0 + i * srcUnit * pack * bytes);
                                        auto dstFloatPtr = (float*)(midBuffer1Offset + i * ePack * pack * bytes);
                                        mSourceTransform(srcFloatPtr, dstFloatPtr, pack, alphaXStride);
                                    }
                                    midBuffer1Offset += IC4alpha2Stride * bytes;
                                }
                            }
                            // durationSourceTrans1 += timer.durationInUs();

                            destSOffset += pack * bytes;
                        }
                        oxBegin = 0;
                        remain -= step;
                    }
                } else {
                    int dstZStep    = xC * pack;  // hUnit*wUnit * 4
                    int unitStep    = ic_4 * xC * pack; //  C/4 * hUnit*wUnit * 4
                    for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                        int hIndex = hbIndex % hUnit;
                        int bIndex = hbIndex / hUnit;
                        int step = std::min(wUnit - oxBegin, remain);
                        int srcY  = hIndex * dstUnit - padY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY; //h dim pack element length
                        int sy    = ALIMAX(0, srcY) - srcY;  // first y element
                        for (int si=0; si<step; ++si) {
                            auto wIndex = si + oxBegin;
                            int srcX  = wIndex * dstUnit - padX;
                            int sx    = ALIMAX(0, srcX) - srcX;
                            int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                            int count = pack * (ex - sx);

                            auto srcStart = srcOrigin + (srcX + srcY * iw + bIndex * iw * ih) * pack * bytes;
                            // MNN_PRINT("\nxIndex:%d, xC:%d, alphaXStride:%d, srcUnit:%d, destUnit:%d, hUnit:%d, wUnit:%d, srcY:%d, hStart:%d ,hEnd:%d, wStart:%d, wEnd:%d, i_oh:%d, i_ow:%d, srcOffset:%ld, destSOffset:%d\n",
                            //     xIndex, xC, alphaXStride, srcUnit, dstUnit, hUnit, wUnit, srcY, sy, ey, sx, ey, hIndex - oyBegin, si, (srcStart - srcOrigin)/bytes, (destSOffset)/bytes);
                            // timer.reset();
                            auto dst_x = _srcOrigin + destSOffset;
                            if (ex - sx == srcUnit && ey - sy == srcUnit) {
                                for (int z = 0; z < ic_4; ++z) {
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    // Transform

                                    for (int i = 0; i < srcUnit; ++i) {
                                        auto srcFloatPtr = (const float*)(srcZ + i * iw * pack * bytes);
                                        auto dstFloatPtr = (float*)(midBuffer1 + i * pack * bytes);
                                        // MNN_PRINT("z:%d, 1 stage i_Nh:%d th srcOffset:%ld, destOffset:%ld, \n", z, i, ((unsigned const char*)srcFloatPtr - srcOrigin)/bytes, ((unsigned const char*)dstFloatPtr - midBuffer1)/bytes);
                                        // MNN_PRINT("winograd source sub matrix:\n");
                                        // formatMatrix(srcFloatPtr, {srcUnit, pack});
                                        mSourceTransform(srcFloatPtr, dstFloatPtr, pack, pack * srcUnit);

                                    }
                                    auto dstZ = dst_x + z * dstZStep * bytes;
                                    for (int i = 0; i < srcUnit; ++i) {
                                        auto srcFloatPtr = (const float*)(midBuffer1 + i * srcUnit * pack * bytes);
                                        auto dstFloatPtr = (float*)(dstZ + i * unitStep * bytes);
                                        mSourceTransform(srcFloatPtr, dstFloatPtr, pack,
                                                         unitStep * srcUnit);
                                    }
                                }
                            } else {
                                for (int z = 0; z < ic_4; ++z) {
                                    // Extract
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    ::memset(midBuffer0, 0, mTransformMidBuffer->stride(1));
                                    if (count > 0) {
                                        for (int yy = sy; yy < ey; ++yy) {
                                            auto dst_yy = midBuffer0 + (yy * srcUnit + sx) * pack * bytes;
                                            auto src_yy = srcZ + (iw * yy + sx) * pack * bytes;
                                            ::memcpy(dst_yy, src_yy, count * bytes);
                                        }
                                    }

                                    // Transform
                                    for (int i = 0; i < srcUnit; ++i) {
                                        auto srcFloatPtr = (const float*)(midBuffer0 + i * srcUnit * pack * bytes);
                                        auto dstFloatPtr = (float*)(midBuffer1 + i * pack * bytes);
                                        mSourceTransform(srcFloatPtr, dstFloatPtr, pack, pack * srcUnit);
                                    }
                                    auto dstZ = dst_x + z * dstZStep * bytes;
                                    for (int i = 0; i < srcUnit; ++i) {
                                        auto srcFloatPtr = (const float*)(midBuffer1 + i * srcUnit * pack * bytes);
                                        auto dstFloatPtr = (float*)(dstZ + i * unitStep * bytes);
                                        mSourceTransform(srcFloatPtr, dstFloatPtr, pack, unitStep * srcUnit);
                                    }
                                }
                            }
                            // durationSourceTrans1 += timer.durationInUs();
                            destSOffset += pack * bytes;
                        }
                        oxBegin = 0;
                        remain -= step;
                    }
                }
            }

#endif
            auto* _dstOrigin = _srcOrigin;
            if (fuseTransformPack) {
                _dstOrigin += ePack * srcUnit2 * ic_4 * pack * bytes;
                if (xC != ePack) {
                    auto midTransformPtr = midBuffer1 + xC * pack * bytes;
                    for (int i = 0; i < ic_4 * srcUnit2; ++i) {
                        memset(midTransformPtr, 0, (ePack - xC) * pack * bytes);
                        midTransformPtr += ePack * pack * bytes;
                    }
                }
                // MNN_PRINT("winograd source matrix transform 1 D*B:\n");
                // formatMatrix((const ElementType*)midBuffer1, {ic_4, srcUnit, srcUnit, ePack, pack});
                for (int iNw = 0; iNw < srcUnit; ++iNw) { // i_Nw
                    // timer.reset();
                    auto midTransformPtr = midBuffer1 + iNw * alphaXStride * bytes;
                    auto unitsGemmbuffer = gemmBuffer;
                    for (int z = 0; z < ic_4; ++z) { // ic_4
                        mSourceTransformPack((float*)midTransformPtr, (float*)unitsGemmbuffer, ePack * pack * ic_4);
                        unitsGemmbuffer += ePack * pack * bytes;
                        midTransformPtr += IC4alpha2Stride * bytes;
                    }

                    // durationSourceTrans2 += timer.durationInUs();
                    // timer.reset();
                    // MNN_PRINT("winograd source matrix transform 2 BT*D*B, iNw:%d\n", iNw);
                    // formatMatrix((const ElementType*)gemmBuffer, {srcUnit, ic_4 * pack, ePack});

                    // Previous tranform requires xC aligned with EPack, xC should be Epack;
                    for (int iNh = 0; iNh < srcUnit; ++iNh) { // i_Nh, gemm
                        auto unitsGemmbuffer = gemmBuffer + iNh * ic_4 * pack * ePack * bytes;
                        auto _dstFloatPtr = (float*)(_dstOrigin + (iNh * srcUnit + iNw) * dc_4 * pack * ePack * bytes);
                        auto _weightFloatPtr = (const float*)(weight + (iNh * srcUnit + iNw) * mResource->mWeight->stride(0));
                        core->MNNPackedMatMul(_dstFloatPtr, (float*)unitsGemmbuffer, _weightFloatPtr, parameters.data(), nullptr, nullptr);
                        // MNN_PRINT("winograd MatMul result, iNh%d, iNw:%d\n", iNh, iNw);
                        // formatMatrix((const ElementType*)_dstFloatPtr, { dc_4, ePack, pack});
                    }
                    // durationMul += timer.durationInUs();
                }
            } else {
                // MNN_PRINT("winograd source matrix after b*d*b:\n");
                // formatMatrix((const ElementType*)_srcOrigin, {srcUnit, srcUnit, ic_4, hUnit, wUnit, pack});
                /*Source Transform End*/
                // // Multi
                _dstOrigin += xC * srcUnit2 * ic_4 * pack * bytes;

                int32_t info[4];
                info[0] = 1;
                info[1] = xC;
                info[2] = xC;
                info[3] = 1;
                int32_t el[4];
                el[0] = xC;
                el[1] = parameters[1];
                el[2] = 0;
                el[3] = 0;
                if (xC == ePack) {
                    for (int i = 0; i < srcUnit2; ++i) {
                        // timer.reset();

                        auto srcTemp = (const float*)(_srcOrigin + i * ic_4 * pack * xC * bytes);
                        auto _dstFloatPtr = (float*)(_dstOrigin + i * dc_4 * pack * xC * bytes);
                        auto _weightFloatPtr = (const float*)(weight + i * mResource->mWeight->stride(0));
                        // MNN_PRINT("winograd i_n:%d, xC:%d, ePack:%d, before packA:\n", i, xC, ePack);
                        // formatMatrix((const ElementType*)srcTemp, {ic_4, xC, pack});

                        core->MNNPackC4ForMatMul_A((float*)gemmBuffer, &srcTemp, info, el);

                        // packATime += timer.durationInUs();
                        // timer.reset();
                        // MNN_PRINT("winograd i_n:%d, after packA:\n", i);
                        // formatMatrix((const ElementType*)gemmBuffer, {1, ic_4 * pack, xC});
                        core->MNNPackedMatMul(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, parameters.data(), nullptr, nullptr);
                        // MNN_PRINT("winograd MatMul result, iNh:%d, iNw:%d\n", i/srcUnit, i % srcUnit);
                        // formatMatrix((const ElementType*)_dstFloatPtr, { dc_4, xC, pack});
                        // durationMul += timer.durationInUs();
                    }
                } else {
                    for (int i = 0; i < srcUnit2; ++i) {
                        // timer.reset();
                        auto srcTemp = (const float*)(_srcOrigin + i * ic_4 * pack * xC * bytes);
                        auto _dstFloatPtr = (float*)(_dstOrigin + i * dc_4 * pack * xC * bytes);
                        auto _weightFloatPtr = (const float*)(weight + i * mResource->mWeight->stride(0));
                        // MNN_PRINT("winograd i_n:%d, xC:%d, ePack:%d, before packA:\n", i, xC, ePack);
                        // formatMatrix((const ElementType*)srcTemp, {ic_4, xC, pack});

                        core->MNNPackC4ForMatMul_A((float*)gemmBuffer, &srcTemp, info, el);
                        // packATime += timer.durationInUs();
                        // timer.reset();
                        // MNN_PRINT("winograd i_n:%d, after packA:\n", i);
                        // formatMatrix((const ElementType*)gemmBuffer, {1, ic_4 * pack, xC});
                        core->MNNPackedMatMulRemain(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, xC, parametersRemain.data(), nullptr, nullptr);
                        // MNN_PRINT("winograd MatMul result, iNh:%d, iNw:%d\n", i/srcUnit, i % srcUnit);
                        // formatMatrix((const ElementType*)_dstFloatPtr, { dc_4, xC, pack});
                        // durationMul += timer.durationInUs();
                    }
                }
            }

#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
            /* Dest Transform And Post Treat Begin */
            {

                int srcZStep = (fuseTransformPack ? ePack : xC) * pack;
                int unitStep = (fuseTransformPack ? ePack : xC) * dc_4 * pack;
                int dstZStep = ow * oh * pack * batch;
                int oyBegin = xIndex / wUnit;
                int oxBegin = xIndex % wUnit;
                int oyEnd = (xIndex + xC-1) / wUnit;
                int remain = xC;
                auto dstS = _dstOrigin;
                for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                    int hIndex = hbIndex % hUnit;
                    int bIndex = hbIndex / hUnit;
                    int step = std::min(wUnit - oxBegin, remain);
                    int dstY = hIndex * dstUnit;
                    int ey = ALIMIN(dstY + dstUnit, oh) - dstY;
                    for (int si=0; si<step; ++si) {
                        auto wIndex = si + oxBegin;
                        auto srcXi = dstS + pack * si * bytes;
                        int dstX = wIndex * dstUnit;
                        auto dstStart = dstOrigin + (dstX + dstY * ow + bIndex * ow * oh) * pack * bytes;
                        int ex = ALIMIN(dstX + dstUnit, ow) - dstX;

                        int count = ex * pack;
                        if (ex == dstUnit) {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep * bytes;
                                auto srcZ     = srcXi + z * srcZStep * bytes;
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    auto srcFloatPtr = (const float*)(srcZ + i * unitStep * bytes);
                                    auto dstFloatPtr = (float*)(midBuffer0 + i * dstUnit * pack * bytes);
                                    mDestTransform(srcFloatPtr, dstFloatPtr, srcUnit * unitStep, pack);
                                }
                                for (int i = 0; i < ey; ++i) {
                                    auto srcFloatPtr = (const float*)(midBuffer0 + i * pack * bytes);
                                    auto dstFloatPtr = (float*)(dstZAddr + i * pack * ow * bytes);
                                    mDestTransform(srcFloatPtr, dstFloatPtr, pack * dstUnit, pack);
                                }
                            }
                        } else {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep * bytes;
                                auto srcZ     = srcXi + z * srcZStep * bytes;
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    auto srcFloatPtr = (const float*)(srcZ + i * unitStep * bytes);
                                    auto dstFloatPtr = (float*)(midBuffer0 + i * dstUnit * pack * bytes);
                                    mDestTransform(srcFloatPtr, dstFloatPtr, srcUnit * unitStep, pack);
                                }
                                for (int i = 0; i < ey; ++i) {
                                    auto srcFloatPtr = (const float*)(midBuffer0 + i * pack * bytes);
                                    auto dstFloatPtr = (float*)(midBuffer1 + i * dstUnit * pack * bytes);
                                    mDestTransform(srcFloatPtr, dstFloatPtr, pack * dstUnit, pack);
                                }
                                for (int yy = 0; yy < ey; ++yy) {
                                    auto dstYAddr = dstZAddr + yy * pack * ow * bytes;
                                    auto srcYAddr = midBuffer1 + yy * pack * dstUnit * bytes;
                                    ::memcpy(dstYAddr, srcYAddr, count * bytes);
                                }
                            }
                        }
                    }
                    oxBegin = 0;
                    remain -= step;
                    dstS += pack * step * bytes;
                }
            }
#endif
            /*Dest Transform And Post Treat End*/
            // if (fuseTransformPack) {
            //     MNN_PRINT(
            //         "\n relayout fused:\n\tdurationSourceTrans1: %lu us \n\tdurationSourceTrans2: %lu us \n\tdurationMul: %lu us\n\ttotal: %lu us\n",
            //         durationSourceTrans1, durationSourceTrans2, durationMul, durationSourceTrans1 + durationSourceTrans2 + durationMul);
            // } else {
            //     MNN_PRINT(
            //         "\n origin:\n\tdurationSourceTrans1+2: %lu us \n\t packA:%lu us \n\t durationMul:%lu us\n\ttotal: %lu us\n",
            //         durationSourceTrans1, packATime, durationMul, durationSourceTrans1 + durationSourceTrans2 + durationMul + packATime);
            // }
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        tFunction((int)tId);
    }
    MNN_CONCURRENCY_END();

    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        for (int dy=(int)tId; dy < dc_4; dy += threadNumber) {
            auto dataFloatPtr = (float*)(dstOrigin + ow * oh * batch * dy * pack * bytes);
            auto biasFloatPtr = (const float*)(bias + pack * dy * bytes);
            core->MNNAxByClampBroadcastUnit(dataFloatPtr, dataFloatPtr, biasFloatPtr, ow * oh * batch, 0, 0, 1,  mPostParameters.data());
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

int ConvolutionWinograd::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
    auto core = static_cast<CPUBackend*>(b)->functions();
    int ow      = outputTensor->width();
    int oh      = outputTensor->height();
    int oc      = outputTensor->channel();
    int ePack, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    int unit2   = UP_DIV(ow * oh, ePack * threadNumber);
    int maxUnit = (int)::sqrtf((float)unit2);
    maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
    maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);

    int ic           = inputTensor->channel();
    auto kernelSize  = common->kernelY();
    int unit         = 0;
    float maxRate    = 0.0f;
    float originCost = (float)ow * oh * (float)ic * oc * kernelSize * kernelSize;
    std::set<int> supportSu{4, 6, 8};
    for (int u = CONVOLUTION_WINOGRAD_MIN_UNIT; u <= maxUnit; ++u) {
        auto sui = u + kernelSize - 1;
        auto su = (float)sui;
        if (supportSu.find(sui) == supportSu.end()) {
            continue;
        }
        if (nullptr == core->chooseWinoDestTransform((int)su, u)) {
            continue;
        }
        /*Let F(6,3) be choosed when it can speed up from F(2,3) than 0.6*/
        float penalty = (su * su) / (float)(kernelSize * kernelSize) * 0.12f;
        float winogradCost =
            (2 * su * su * ic + su * su * ic * oc + (su + u) * su * oc) * (UP_DIV(ow, u) * UP_DIV(oh, u));
        float reduceRate = originCost / winogradCost - penalty;
        // MNN_PRINT("ow=%d, oh=%d, %f, %f, winograd unit:%d\n", ow, oh, winogradCost, reduceRate, u);
        if (reduceRate > maxRate) {
            maxRate = reduceRate;
            unit    = u;
        }
    }
    if (maxRate < 1.0f) {
        return 0;
    }
    return unit;
}

bool ConvolutionWinograd::canUseWinograd(const Convolution2DCommon *common) {
    if (common->kernelY() != common->kernelX() || common->kernelY() <= 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    return true;
}

ErrorCode ConvolutionWinograd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    // FUNC_PRINT(mA->length(1));
    bool success = backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    success      = success && (backend()->onAcquireBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC));
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}
} // namespace MNN
