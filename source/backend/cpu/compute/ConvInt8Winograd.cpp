#include "ConvInt8Winograd.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "ConvOpt.h"
#include "Int8FunctionsOpt.h"
#include "CommonOptFunction.h"
#include "WinogradOptFunctionInt8.hpp"
#include "MNN/AutoTime.hpp"
#include "math/Vec.hpp"
#include <map>
#include <numeric>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define CONV_WINOGRAD_ALPHA 4

namespace MNN {

std::shared_ptr<ConvInt8Winograd::WinoResource> ConvInt8Winograd::makeWinoResource(const int8_t* originWeight, Backend* backend, int oc, int ic, int alpha2, int step) {
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int oc4 = UP_DIV(oc, UNIT), ic4 = UP_DIV(ic, SRC_UNIT);
    
    std::shared_ptr<Tensor> weight, offsets;
    weight.reset(Tensor::createDevice<int8_t>({alpha2, oc4, ic4, UNIT, SRC_UNIT}));
    offsets.reset(Tensor::createDevice<int32_t>({alpha2, oc4, UNIT}));
    bool success = backend->onAcquireBuffer(weight.get(), Backend::STATIC);
    success &= backend->onAcquireBuffer(offsets.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Memory not enough\n");
        return nullptr;
    }
    ::memset(weight->host<int8_t>(), 0, weight->size());
    ::memset(offsets->host<int32_t>(), 0, offsets->size());
    for (int i = 0; i < alpha2; ++i) {
        for (int oz = 0; oz < oc; ++oz) {
            auto srcZ = originWeight + oz * ic * step + i;
            auto dstZ = weight->host<int8_t>() + ((i * oc4 + oz / UNIT) * ic4 * UNIT + oz % UNIT) * SRC_UNIT;
            for (int sz = 0; sz < ic; ++sz) {
                dstZ[(sz / SRC_UNIT) * UNIT * SRC_UNIT + sz % SRC_UNIT] = srcZ[sz * step];
            }
#ifdef MNN_USE_SSE
            int offset = 0;
            for (int sz = 0; sz < ic; ++sz) {
                offset += srcZ[sz * step];
            }
            offsets->host<int32_t>()[i * oc4 * UNIT + oz] = offset * (-128);
#endif
        }
    }
    std::shared_ptr<WinoResource> resource(new WinoResource);
    resource->weight = weight;
    resource->offsets = offsets;
    resource->backend = backend;
    return resource;
}

ConvInt8Winograd::ConvInt8Winograd(Backend *b, const Convolution2D *convOp, std::shared_ptr<ResourceInt8> res) : CPUConvolution(convOp->common(), b), mResource(res) {
    int oc = mCommon->outputCount(), ic = mCommon->inputCount();
    auto core = static_cast<CPUBackend*>(b)->int8Functions();
    
    int size = convOp->symmetricQuan()->winogradAttr()->size();
    if (size <= 4 || (size - 4) % 4 != 0) {
        MNN_ERROR("wrong winogradAttr proto");
        mValid = false;
        return;
    }
    int unitNum = (size - 4) / 4, step = res->mWeightInt8->elementSize() / oc / ic;
    auto weightData = res->mWeightInt8->host<int8_t>();
    for (int i = 0; i < unitNum; ++i) {
        auto attr = convOp->symmetricQuan()->winogradAttr()->data() + i * 4 + 4;
        int kyStart = attr[0], kxStart = attr[1], unitY = attr[2], unitX = attr[3];
        int alphaY = (unitY == 1 ? 1 : CONV_WINOGRAD_ALPHA), alphaX = (unitX == 1 ? 1 : CONV_WINOGRAD_ALPHA), alpha2 = alphaY * alphaX;
        int kySize = alphaY - unitY + 1, kxSize = alphaX - unitX + 1;
        std::shared_ptr<Tensor> tempInput, tempOutput;
        auto winoRes = makeWinoResource(weightData, b, oc, ic, alpha2, step);
        std::shared_ptr<WinoExecution> exe(new WinoExecution(winoRes, kySize, kxSize, unitY, unitX, oc, ic));
        mUnits.push_back({kyStart, kxStart, tempInput, tempOutput, exe});
        weightData += alpha2;
    }
#ifdef MNN_USE_SSE
    for (int i = 0; i < mResource->offsets.size(); ++i) {
        mResource->mBiasInt32->host<int32_t>()[i] -= mResource->offsets[i];
    }
    mResource->offsets.clear();
#endif
    mResource->mWeightInt8.reset((Tensor*)nullptr);
}
ConvInt8Winograd::ConvInt8Winograd(Backend* backend, const Convolution2DCommon* common, const ConvInt8Winograd& exe)
    : CPUConvolution(common, backend) {
    for (const auto& unit : exe.mUnits) {
        std::shared_ptr<Tensor> tempInput, tempOutput;
        std::shared_ptr<WinoExecution> runner(new WinoExecution(backend, *unit.runner.get()));
        mUnits.push_back({unit.kyStart, unit.kxStart, tempInput, tempOutput, runner});
    }
    mResource = exe.mResource;
}
ConvInt8Winograd::~ConvInt8Winograd() {
    // Do nothing
}
bool ConvInt8Winograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvInt8Winograd(bn, op->main_as_Convolution2D()->common(), *this);
    if (!dstExe->valid()) {
        return false;
    }
    *dst = dstExe;
    return false;
}
ErrorCode ConvInt8Winograd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
    CPUConvolution::onResize(inputs, outputs);
    
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    auto input = inputs[0], output = outputs[0];
    int batch = input->batch(), ic = input->channel(), oc = output->channel();
    int ih = input->height(), iw = input->width();
    for (auto& unit : mUnits) {
        unit.output.reset(Tensor::createDevice<float>(output->shape(), Tensor::CAFFE_C4));
        mValid = backend()->onAcquireBuffer(unit.output.get(), Backend::DYNAMIC);
        if (!mValid) {
            return OUT_OF_MEMORY;
        }
    }
    for (auto& unit : mUnits) {
        int sy = ALIMAX(unit.kyStart - mPadY, 0), sx = ALIMAX(unit.kxStart - mPadX, 0);
        auto srcData = input->host<int8_t>() + (sy * iw + sx) * UNIT;
        unit.input.reset(Tensor::create<int8_t>({batch, ic, ih - sy, iw - sx}, srcData, Tensor::CAFFE_C4));
        TensorUtils::getDescribe(unit.input.get())->quantAttr = TensorUtils::getDescribe(input)->quantAttr;
        for (int i = 0; i < input->dimensions(); ++i) {
            unit.input->setStride(i, input->stride(i));
        }
        unit.runner->mPadY = ALIMAX(mPadY - unit.kyStart, 0);
        unit.runner->mPadX = ALIMAX(mPadX - unit.kxStart, 0);
        auto res = unit.runner->onResize({unit.input.get()}, {unit.output.get()});
        if (res != NO_ERROR) {
            mValid = false;
            return res;
        }
    }
    for (auto& unit : mUnits) {
        backend()->onReleaseBuffer(unit.output.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

#if defined(MNN_USE_NEON)
// Assumption UNIT == 4
static void mergeAddBiasScaleQuantize(const std::vector<Tensor*>& inputs, Tensor* output, const QuanPostTreatParameters* quanParam) {
    constexpr int UNIT = 4;
    using VecType = MNN::Math::Vec<float, UNIT>;
    int countC4 = UP_DIV(output->channel(), UNIT), plane = output->height() * output->width();
    std::vector<float*> srcDatas(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        srcDatas[i] = inputs[i]->host<float>();
    }
    auto minVec = vdupq_n_s32(quanParam->minValue), maxVec = vdupq_n_s32(quanParam->maxValue);
#ifndef __aarch64__
    auto zeroVec = vdupq_n_f32(0), roundValuePosVec = vdupq_n_f32(0.5), roundValueNegVec = vdupq_n_f32(-0.5);
#endif // nodef __aarch64__
    for (int c = 0; c < countC4; ++c) {
        auto biasVec = VecType::load(quanParam->bias + c * UNIT); // load int32_t vec from mem then convert to float vec
        auto dstZ = output->host<int8_t>() + c * plane * UNIT;
        // save one time fp32 read/write compared to invoke MNNFloat2Int8 directly
        auto scaleVec = VecType::load(quanParam->scale + c * UNIT);
        int p = 0;
        for (; p < plane - 3; p += 4) {
            VecType result[4] = {biasVec, biasVec, biasVec, biasVec};
            for (int j = 0; j < inputs.size(); ++j) {
                for (int k = 0; k < 4; ++k) {
                    result[k] = result[k] + VecType::load(srcDatas[j] + (p + k) * UNIT);
                }
            }
#ifdef __aarch64__
#define ROUND_CVT_CLIP(i) auto tmp##i = vmaxq_s32(vminq_s32(vcvtaq_s32_f32((result[i] * scaleVec).value), maxVec), minVec);
#define I32_to_I16(v1, v2) vqmovn_high_s32(vqmovn_s32(v1), v2)
#define I16_to_I8(v1, v2) vqmovn_high_s16(vqmovn_s16(v1), v2)
#else
#define ROUND_CVT_CLIP(i) \
auto ftmp##i = (result[i] * scaleVec).value; \
auto tmp##i = vmaxq_s32(vminq_s32(vcvtq_s32_f32(vaddq_f32(ftmp##i, vbslq_f32(vcgtq_f32(ftmp##i, zeroVec), roundValuePosVec, roundValueNegVec))), maxVec), minVec);
#define I32_to_I16(v1, v2) vcombine_s16(vqmovn_s32(v1), vqmovn_s32(v2))
#define I16_to_I8(v1, v2) vcombine_s8(vqmovn_s16(v1), vqmovn_s16(v2))
#endif
            ROUND_CVT_CLIP(0)
            ROUND_CVT_CLIP(1)
            ROUND_CVT_CLIP(2)
            ROUND_CVT_CLIP(3)
            auto i8result = I16_to_I8(I32_to_I16(tmp0, tmp1), I32_to_I16(tmp2, tmp3));
            vst1q_s8(dstZ + p * UNIT, i8result);
        }
        for (; p < plane; ++p) {
            for (int k = 0; k < UNIT; ++k) {
                float result = quanParam->bias[c * UNIT + k];
                for (int j = 0; j < inputs.size(); ++j) {
                    result += srcDatas[j][p * UNIT + k];
                }
                auto i32result = ALIMAX(ALIMIN(roundf(result * quanParam->scale[c * UNIT + k]), quanParam->maxValue), quanParam->minValue);
                dstZ[p * UNIT + k] = (int8_t)(i32result);
            }
        }
        for (int i = 0; i < inputs.size(); ++i) {
            srcDatas[i] += plane * UNIT;
        }
    }
}
#else
template<int UNIT>
static void mergeAddBiasScaleQuantize(const std::vector<Tensor*>& inputs, Tensor* output, const QuanPostTreatParameters* quanParam, Backend* backend) {
    auto MNNFloat2Int8 = static_cast<CPUBackend*>(backend)->int8Functions()->MNNFloat2Int8;
    // Vec8/Vec16 use AVX2/AVX512 has different ABI (vector argument pass convention), so can't use directly here. We use many Vec4(SSE) do VecN work
    using VecType = MNN::Math::Vec<float, 4>;
    constexpr int N = UNIT / 4;
    int countUnit = UP_DIV(output->channel(), UNIT), plane = output->height() * output->width();
    std::vector<float*> srcDatas(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        srcDatas[i] = inputs[i]->host<float>();
    }
    for (int c = 0; c < countUnit; ++c) {
        std::array<VecType, N> biasVec; // load int32_t vec from mem then convert to float vec
        for (int n = 0; n < N; ++n) {
            biasVec[n] = VecType::load(quanParam->bias + c * UNIT + n * 4);
        }
        auto dstZ = output->host<int8_t>() + c * plane * UNIT;
        for (int p = 0; p < plane; ++p) {
            auto result = biasVec;
            for (int j = 0; j < inputs.size(); ++j) {
                for (int n = 0; n < N; ++n) {
                    result[n] = result[n] + VecType::load(srcDatas[j] + p * UNIT + n * 4);
                }
            }
            for (int n = 0; n < N; ++n) {
                VecType::save(srcDatas[0] + p * UNIT + n * 4, result[n]);
            }
        }
        MNNFloat2Int8(srcDatas[0], dstZ, plane, quanParam->scale + c * UNIT, quanParam->minValue, quanParam->maxValue, 0);
        for (int i = 0; i < inputs.size(); ++i) {
            srcDatas[i] += plane * UNIT;
        }
    }
}
#endif

ErrorCode ConvInt8Winograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<Tensor*> tmp_outputs;
    for (auto& unit : mUnits) {
        auto ret = unit.runner->onExecute({unit.input.get()}, {unit.output.get()});
        if (ret != NO_ERROR) {
            return ret;
        }
        tmp_outputs.push_back(unit.output.get());
    }
    QuanPostTreatParameters quanParam;
    quanParam.scale = mResource->mScaleFloat->host<float>();
    quanParam.bias = mResource->mBiasInt32->host<int32_t>();
    quanParam.maxValue = mResource->mClampMax;
    if (mResource->mRelu) {
        quanParam.minValue = mResource->mOutputZeroPoint;
    } else {
        quanParam.minValue = mResource->mClampMin;
    }
#if defined(MNN_USE_NEON)
    mergeAddBiasScaleQuantize(tmp_outputs, outputs[0], &quanParam);
#else
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    if (UNIT == 4) {
        mergeAddBiasScaleQuantize<4>(tmp_outputs, outputs[0], &quanParam, backend()); // SSE and other
    } else if (UNIT == 8) {
        mergeAddBiasScaleQuantize<8>(tmp_outputs, outputs[0], &quanParam, backend()); // AVX2
    } else if (UNIT == 16) {
        mergeAddBiasScaleQuantize<16>(tmp_outputs, outputs[0], &quanParam, backend()); // AVX512
    } else {
        return NOT_SUPPORT;
    }
#endif
    return NO_ERROR;
};

ConvInt8Winograd::WinoExecution::WinoExecution(std::shared_ptr<WinoResource> res, int kernelY, int kernelX, int unitY, int unitX, int outputCount, int inputCount)
: Execution(res->backend), mWinoResource(res), mUnitY(unitY), mUnitX(unitX), mKernelY(kernelY), mKernelX(kernelX) {
    auto core = static_cast<CPUBackend*>(res->backend)->int8Functions();
    
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    int alphaY = mUnitY + mKernelY - 1, alphaX = mUnitX + mKernelX - 1, alpha2 = alphaY * alphaX;
    int ic4 = UP_DIV(inputCount, SRC_UNIT), oc4 = UP_DIV(outputCount, UNIT);
    mTempInputBuffer.reset(Tensor::createDevice<int8_t>({threadNumber, alpha2, ic4, DST_XUNIT * SRC_UNIT}));
    mTempOutputBuffer.reset(Tensor::createDevice<float>({threadNumber, alpha2, oc4, DST_XUNIT * UNIT}));
    // ROUND_UP(... , sizeof(float)) make midSize align up with float mem
    int midSize = ALIMAX(ROUND_UP(ic4 * 16 * alpha2 * DST_XUNIT, sizeof(float)), oc4 * UNIT * alpha2 * DST_XUNIT * sizeof(float));
    mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, midSize}));
}
ConvInt8Winograd::WinoExecution::WinoExecution(Backend* bn, const WinoExecution& exe)
    : Execution(bn), mWinoResource(exe.mWinoResource),
    mUnitY(exe.mUnitY), mUnitX(exe.mUnitX), mKernelY(exe.mKernelY), mKernelX(exe.mKernelX),
    mPadY(exe.mPadY), mPadX(exe.mPadX) {
    
    mTempInputBuffer.reset(Tensor::createDevice<int8_t>(exe.mTempInputBuffer->shape()));
    mTempOutputBuffer.reset(Tensor::createDevice<float>(exe.mTempOutputBuffer->shape()));
    mTransformMidBuffer.reset(Tensor::createDevice<int8_t>(exe.mTransformMidBuffer->shape()));
}
ErrorCode ConvInt8Winograd::WinoExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    bool success = backend()->onAcquireBuffer(mTempInputBuffer.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mTempOutputBuffer.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempInputBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempOutputBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}
ErrorCode ConvInt8Winograd::WinoExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto input = inputs[0], output = outputs[0];
    
    int alphaY = mKernelY + mUnitY - 1, alphaX = mKernelX + mUnitX - 1, alpha2 = alphaY * alphaX;
    bool conv1d = (alphaY == 1 || alphaX == 1);
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    auto gemmFunc = core->Int8GemmKernel;
    
    WinogradFunctionInt8::SrcTransXFunc srcTransXFunc;
    WinogradFunctionInt8::SrcTransYFunc srcTransYFunc;
    WinogradFunctionInt8::SrcTrans2Func srcTrans2Func;
    if (mKernelY != 1 && mKernelX != 1) {
        if (alphaY != alphaX) {
            return NOT_SUPPORT;
        }
        srcTrans2Func = WinogradFunctionInt8::chooseSourceTransform2(alphaX, UNIT, SRC_UNIT);
    } else if (mKernelY == 1) {
        srcTransXFunc = WinogradFunctionInt8::chooseSourceTransformX(alphaX, UNIT, SRC_UNIT);
    } else {
        srcTransYFunc = WinogradFunctionInt8::chooseSourceTransformY(alphaY, UNIT, SRC_UNIT);
    }
    CoreFunctions::WinoTransFunc dstTransXFunc, dstTransYFunc;
    auto chooseDest = static_cast<CPUBackend*>(backend())->functions()->chooseWinoDestTransform;
    if (mKernelY != 1) {
        dstTransYFunc = chooseDest(alphaY, mUnitY);
    }
    if (mKernelX != 1) {
        dstTransXFunc = chooseDest(alphaX, mUnitX);
    }
    
    int32_t inputZeroPoint = TensorUtils::getQuantInfo(input)[1];
#ifdef MNN_USE_SSE
    inputZeroPoint += 128;
#endif
    
    int ow = output->width(), oh = output->height();
    int iw = input->width(), ih = input->height();
    int ic = input->channel(), ic_4 = UP_DIV(ic, UNIT);
    int dc_4 = UP_DIV(output->channel(), UNIT);

    int padY = mPadY, padX = mPadX;
    auto wUnit = UP_DIV(ow, mUnitX), hUnit = UP_DIV(oh, mUnitY);

    auto totalCount   = wUnit * hUnit;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int tileCount    = UP_DIV(totalCount, DST_XUNIT);
    threadNumber     = std::min(threadNumber, tileCount);
    
    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = input->host<int8_t>() + batchIndex * input->stride(0);
        auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);

        auto weight    = mWinoResource->weight->host<int8_t>();
        auto tFunction = [&](int tId) {
            auto _srcOrigin = mTempInputBuffer->host<int8_t>() + tId * mTempInputBuffer->stride(0);
            auto _dstOrigin = mTempOutputBuffer->host<float>() + tId * mTempOutputBuffer->stride(0);
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex  = (int)tIndex * DST_XUNIT;
                int xReamin = totalCount - xIndex;
                int xC      = xReamin > DST_XUNIT ? DST_XUNIT : xReamin;

#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
                {
                    auto midBuffer0 = mTransformMidBuffer->host<int8_t>() + tId * mTransformMidBuffer->stride(0);
                    int dstZStep = mTempInputBuffer->stride(2);
                    int unitStep = mTempInputBuffer->stride(1);
                    int oyBegin = xIndex / wUnit;
                    int oxBegin = xIndex % wUnit;
                    int oyEnd = (xIndex + xC-1) / wUnit;
                    int remain = xC;
                    auto dstS = _srcOrigin;
                    for (int hIndex=oyBegin; hIndex <= oyEnd; ++hIndex) {
                        int step = std::min(wUnit - oxBegin, remain);
                        int srcY  = hIndex * mUnitY - padY;
                        int ey    = ALIMIN(srcY + alphaY, ih) - srcY;
                        int sy    = ALIMAX(0, srcY) - srcY;
                        
                        int sBegin = step, sEnd = step;
                        if (ey - sy == alphaY) {
                            for (int si = 0; si < step; ++si) {
                                auto wIndex = si + oxBegin;
                                int srcX  = wIndex * mUnitX - padX;
                                int sx    = ALIMAX(0, srcX) - srcX;
                                int ex    = ALIMIN(srcX + alphaX, iw) - srcX;
                                if (sBegin == step && ex - sx == alphaX) {
                                    sBegin = si;
                                } else if (sBegin < step && ex - sx != alphaX) {
                                    sEnd = si;
                                    break;
                                }
                            }
                        }
                        for (int si=0; si<step;) {
                            int sStep = (si == sBegin ? sEnd - sBegin : 1);
                            auto wIndex = si + oxBegin;
                            int srcX  = wIndex * mUnitX - padX;
                            int sx    = ALIMAX(0, srcX) - srcX;
                            int ex    = ALIMIN(srcX + alphaX, iw) - srcX;
                            auto dst_x = dstS + si * SRC_UNIT;
                            
                            int sourceZStep = input->stride(1) * UNIT, sourceYStep = input->stride(2) * UNIT;
                            auto srcStart = srcOrigin + srcY * sourceYStep + srcX * UNIT;
                            // when input window exceed limit (so need pad value), copy from src to midbuffer0
                            if (ex - sx != alphaX || ey - sy != alphaY) {
                                ::memset(midBuffer0, inputZeroPoint, mTransformMidBuffer->stride(1));
                                int count = UNIT * (ex - sx);
                                for (int z = 0; count > 0 && z < ic_4; ++z) {
                                    for (int yy = sy; yy < ey; ++yy) {
                                        auto dst_yy = midBuffer0 + ((z * alphaY + yy) * alphaX + sx) * UNIT;
                                        auto src_yy = srcStart + z * sourceZStep + yy * sourceYStep + sx * UNIT;
                                        ::memcpy(dst_yy, src_yy, count);
                                    }
                                }
                                srcStart = midBuffer0;
                                sourceZStep = alpha2 * UNIT;
                                sourceYStep = alphaX * UNIT;
                            }
                            if (!conv1d) {
                                srcTrans2Func(srcStart, dst_x, sourceYStep, sourceZStep, unitStep, dstZStep,
                                              ic_4, sStep, mUnitX);
                            } else if (alphaX == 1) {
                                srcTransYFunc(srcStart, dst_x, sourceYStep, sourceZStep, unitStep, dstZStep, ic_4, sStep);
                            } else {
                                srcTransXFunc(srcStart, dst_x, sourceZStep, unitStep, dstZStep, ic_4, sStep, mUnitX);
                            }
                            si += sStep;
                        }
                        oxBegin = 0;
                        remain -= step;
                        dstS += SRC_UNIT * step;
                    }
                }
#endif
                // Multi
                for (int i = 0; i < alpha2; ++i) {
                    auto _srcInt8Ptr = _srcOrigin + i * mTempInputBuffer->stride(1);
                    auto _dstFloatPtr = _dstOrigin + i * dc_4 * xC * UNIT;
                    auto _weightInt8Ptr = weight + i * mWinoResource->weight->stride(0);
                    QuanPostTreatParameters quanParam;
                    quanParam.bias = mWinoResource->offsets->host<int32_t>() + i * mWinoResource->offsets->stride(0);
                    quanParam.scale = nullptr;
                    gemmFunc((int8_t*)_dstFloatPtr, _srcInt8Ptr, _weightInt8Ptr, mTempInputBuffer->length(2), xC * UNIT * sizeof(float), dc_4, &quanParam, xC);
                }
#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
                {
                    auto midBuffer0 = (float*)(mTransformMidBuffer->host<int8_t>() + tId * mTransformMidBuffer->stride(0));
                    auto midBuffer1 = (float*)((int8_t*)midBuffer0 + mTransformMidBuffer->stride(1));
                    int srcZStep = xC * UNIT;
                    int unitStep = dc_4 * xC * UNIT;
                    int oyBegin = xIndex / wUnit;
                    int oxBegin = xIndex % wUnit;
                    int oyEnd = (xIndex + xC-1) / wUnit;
                    int remain = xC;
                    auto dstS = _dstOrigin;
                    for (int hIndex=oyBegin; hIndex <= oyEnd; ++hIndex) {
                        int step = std::min(wUnit - oxBegin, remain);
                        int dstY = hIndex * mUnitY;
                        int ey = ALIMIN(dstY + mUnitY, oh) - dstY;
                        
                        int sBegin = step, sEnd = step;
                        if (alphaX != 1 || ey == mUnitY) {
                            for (int si = 0; si < step; ++si) {
                                auto wIndex = si + oxBegin;
                                int dstX = wIndex * mUnitX;
                                int ex = ALIMIN(dstX + mUnitX, ow) - dstX;
                                if (sBegin == step && ex == mUnitX) {
                                    sBegin = si;
                                } else if (sBegin < step && ex != mUnitX) {
                                    sEnd = si;
                                    break;
                                }
                            }
                        }
                        for (int si=0; si<step;) {
                            int sStep = (si == sBegin ? sEnd - sBegin : 1);
                            auto wIndex = si + oxBegin;
                            auto srcXi = dstS + UNIT * si;
                            int dstX = wIndex * mUnitX;
                            auto dstStart = dstOrigin + (dstX + dstY * ow) * UNIT;
                            int ex = ALIMIN(dstX + mUnitX, ow) - dstX;
                            int count = ex * UNIT;
                            
                            auto _dstStart = dstStart;
                            int dstZStep = oh * ow * UNIT, dstYStep = ow * UNIT;
                            if (ex != mUnitX || (alphaX == 1 && ey != mUnitY)) {
                                dstZStep = mUnitY * mUnitX * UNIT;
                                dstYStep = mUnitX * UNIT;
                                _dstStart = midBuffer1;
                            }
                            for (int z = 0; z < dc_4; ++z) {
                                for (int x = 0; x < sStep; ++x) {
                                    auto srcXiZ = srcXi + z * srcZStep + x * UNIT;
                                    auto _dstStartZ = _dstStart + z * dstZStep + x * mUnitX * UNIT;
                                    if (!conv1d) {
                                        for (int i = 0; i < 4; ++i) {
                                            dstTransYFunc(srcXiZ + i * unitStep, midBuffer0 + i * UNIT, alphaX * unitStep, alphaX * UNIT);
                                        }
                                        for (int i = 0; i < ey; ++i) {
                                            dstTransXFunc(midBuffer0 + i * alphaX * UNIT, _dstStartZ + i * dstYStep, UNIT, UNIT);
                                        }
                                    } else if (alphaX == 1) {
                                        dstTransYFunc(srcXiZ, _dstStartZ, unitStep, dstYStep);
                                    } else {
                                        for (int i = 0; i < ey; ++i) {
                                            dstTransXFunc(srcXiZ + i * alphaX * unitStep, _dstStartZ + i * dstYStep, unitStep, UNIT);
                                        }
                                    }
                                }
                            }
                            if (ex != mUnitX || (alphaX == 1 && ey != mUnitY)) {
                                for (int z = 0; z < dc_4; ++z) {
                                    for (int yy = 0; yy < ey; ++yy) {
                                        auto srcYAddr = _dstStart + (z * mUnitY + yy) * mUnitX * UNIT;
                                        auto dstYAddr = dstStart + (z * oh + yy) * ow * UNIT;
                                        ::memcpy(dstYAddr, srcYAddr, count * sizeof(float));
                                    }
                                }
                            }
                            si += sStep;
                        }
                        oxBegin = 0;
                        remain -= step;
                        dstS += UNIT * step;
                    }
                }
#endif
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            tFunction((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

bool ConvInt8Winograd::mustUse(const Convolution2D *convOp) {
    auto quan = convOp->symmetricQuan();
    if (quan == nullptr || quan->winogradAttr() == nullptr) {
        return false;
    }
    return true;
}

} /* MNN */
