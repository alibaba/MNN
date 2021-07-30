#include "ConvInt8Winograd.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "ConvOpt.h"
#include "Int8FunctionsOpt.h"
#include "CommonOptFunction.h"
#include "WinogradInt8Helper.hpp"
#include "ConvInt8TiledExecutor.hpp"
#include "MNN/AutoTime.hpp"
#include "math/Vec.hpp"
#include <map>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#ifdef MNN_USE_SSE
extern "C" {
void MNNInt8ToUInt8(void* ptr, int count);
}
#endif

namespace MNN {

bool ConvInt8Winograd::chooseTransformFuncs(int kernelY, int kernelX, int unitY, int unitX, ConvInt8Winograd::WinoExecution* exe, Backend* bn) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    int alphaY = kernelY + unitY - 1, alphaX = kernelX + unitX - 1;
    WinoSrcTransFunc srcFuncY = nullptr, srcFuncX = nullptr;
    WinoDstTransFunc dstFuncY = nullptr, dstFuncX = nullptr;
    if (kernelY != 1 && kernelX != 1) {
        srcFuncX = WinogradInt8Helper::chooseSourceTransform(alphaX, UNIT, 16);
        srcFuncY = WinogradInt8Helper::chooseSourceTransform(alphaY, 16, SRC_UNIT);
    } else if (kernelY == 1 && kernelX != 1) {
        srcFuncX = WinogradInt8Helper::chooseSourceTransform(alphaX, UNIT, SRC_UNIT);
    } else if (kernelY != 1 && kernelX == 1) {
        srcFuncY = WinogradInt8Helper::chooseSourceTransform(alphaY, UNIT, SRC_UNIT);
    }
    if (kernelY != 1) {
        dstFuncY = WinogradInt8Helper::chooseDestTransform(alphaY, unitY);
    }
    if (kernelX != 1) {
        dstFuncX = WinogradInt8Helper::chooseDestTransform(alphaX, unitX);
    }
    if (kernelY != 1 && (srcFuncY == nullptr || dstFuncY == nullptr)) {
        return false;
    }
    if (kernelX != 1 && (srcFuncX == nullptr || dstFuncX == nullptr)) {
        return false;
    }
    if (exe != nullptr) {
        exe->mSourceTransformY = srcFuncY;
        exe->mSourceTransformX = srcFuncX;
        exe->mDestTransformY = dstFuncY;
        exe->mDestTransformX = dstFuncX;
    }
    return true;
}

// create a copy of commonOrigin, then update kernels or pads if needed
using CommonPair = std::pair<const Convolution2DCommon*, unsigned char*>;
static std::shared_ptr<CommonPair> createCommon(const Convolution2DCommon* commonOrigin, std::vector<int> kernels, std::vector<int> pads) {
    using namespace flatbuffers;
    std::shared_ptr<Convolution2DCommonT> temp(commonOrigin->UnPack());
    if (kernels.size() > 0) {
        temp->kernelY = kernels[0];
        temp->kernelX = kernels[1];
    }
    if (pads.size() > 0) {
        temp->padY = pads[0];
        temp->padX = pads[1];
        temp->padMode = PadMode_CAFFE; // use specified pad value
    }
    FlatBufferBuilder builder;
    builder.Finish(Convolution2DCommon::Pack(builder, temp.get()));
    int bufSize = builder.GetSize();
    std::shared_ptr<CommonPair> res(new CommonPair {nullptr, new unsigned char[bufSize]},
                                    [](CommonPair* ptr) { delete[] ptr->second; });
    memcpy(res->second, builder.GetBufferPointer(), bufSize);
    res->first = GetRoot<Convolution2DCommon>(res->second);
    return res;
}

static ErrorCode extractWeight(const Tensor* weightOrigin, std::shared_ptr<Tensor>& weight, const Convolution2DCommon* common, ConvInt8Winograd::UnitAttr unitAttr, Backend* bn) {
    int oc = common->outputCount(), ic = common->inputCount(), kernelY = common->kernelY(), kernelX = common->kernelX();
    weight.reset(Tensor::createDevice<int8_t>({oc, ic, unitAttr.kySize, unitAttr.kxSize}), [=](Tensor* ptr) {
        bn->onReleaseBuffer(ptr, Backend::STATIC);
        delete ptr;
    });
    if (!bn->onAcquireBuffer(weight.get(), Backend::STATIC)) {
        MNN_ERROR("Memory not enough\n");
        return OUT_OF_MEMORY;
    }
    for (int i = 0; i < oc * ic; ++i) {
        auto srcZ = weightOrigin->host<int8_t>() + i * kernelY * kernelX;
        auto dstZ = weight->host<int8_t>() + i * weight->stride(1);
        for (int j = 0; j < unitAttr.kySize; ++j) {
            auto src = srcZ + (j + unitAttr.kyStart) * kernelX + unitAttr.kxStart;
            auto dst = dstZ + j * weight->stride(2);
            ::memcpy(dst, src, unitAttr.kxSize * sizeof(int8_t));
        }
    }
    return NO_ERROR;
}

ConvInt8Winograd::ConvInt8Winograd(Backend *b, const Convolution2D *convOp, std::shared_ptr<ResourceInt8> res,
                                   std::vector<ConvInt8Winograd::UnitAttr>& unitAttrs) : CPUConvolution(convOp->common(), b), mResource(res) {
    int oc = mCommon->outputCount(), ic = mCommon->inputCount(), kernelY = mCommon->kernelY(), kernelX = mCommon->kernelX();
    auto core = static_cast<CPUBackend*>(b)->int8Functions();
    for (const auto& unitAttr : unitAttrs) {
        std::shared_ptr<Tensor> tempInput, tempOutput, weight;
        auto subCommon = createCommon(mCommon, {unitAttr.kySize, unitAttr.kxSize}, {});
        mValid = (extractWeight(mResource->mWeightInt8.get(), weight, mCommon, unitAttr, b) == NO_ERROR);
        if (!mValid) {
            return;
        }
        std::shared_ptr<Execution> exe;
        if (unitAttr.unitY == 1 && unitAttr.unitX == 1) {
#ifdef MNN_USE_SSE
            bool fastgemm = (convOp->symmetricQuan()->nbits() <= 7);
#else
            bool fastgemm = (convOp->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE);
#endif
            exe.reset(new ConvInt8TiledExecutor(b, subCommon->first, weight, fastgemm));
        } else {
            bool fastgemm = false;
#ifdef MNN_USE_SSE
            fastgemm = true;
#endif
            exe.reset(new WinoExecution(b, subCommon->first, weight.get(), unitAttr.unitY, unitAttr.unitX, fastgemm));
        }
        mUnits.push_back({unitAttr, subCommon, tempInput, tempOutput, exe});
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
        Execution* exe = nullptr;
        unit.runner->onClone(backend, nullptr, &exe);
        mUnits.push_back({unit.attr, unit.common, tempInput, tempOutput, std::shared_ptr<Execution>(exe)});
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
        int sy = ALIMAX(unit.attr.kyStart - mPadY, 0), sx = ALIMAX(unit.attr.kxStart - mPadX, 0);
        auto srcData = input->host<int8_t>() + (sy * iw + sx) * UNIT;
        unit.input.reset(Tensor::create<int8_t>({batch, ic, ih - sy, iw - sx}, srcData, Tensor::CAFFE_C4));
        for (int i = 0; i < input->dimensions(); ++i) {
            unit.input->setStride(i, input->stride(i));
        }
        unit.common = createCommon(unit.common->first, {}, {ALIMAX(mPadY - unit.attr.kyStart, 0), ALIMAX(mPadX - unit.attr.kxStart, 0)});
        if (unit.attr.unitY == 1 && unit.attr.unitX == 1) {
            static_cast<ConvInt8TiledExecutor*>(unit.runner.get())->mCommon = unit.common->first;
        } else {
            static_cast<WinoExecution*>(unit.runner.get())->mCommon = unit.common->first;
        }
        
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

// Assumption UNIT == 4
static void mergeAddBiasScaleQuantize(const std::vector<Tensor*>& inputs, Tensor* output, const QuanPostTreatParameters* quanParam) {
    constexpr int UNIT = 4;
    using VecType = MNN::Math::Vec<float, UNIT>;
    int countC4 = UP_DIV(output->channel(), UNIT), plane = output->height() * output->width();
    std::vector<float*> srcDatas(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        srcDatas[i] = inputs[i]->host<float>();
    }
    for (int c = 0; c < countC4; ++c) {
        auto biasVec = VecType::load(quanParam->bias + c * UNIT); // load int32_t vec from mem then convert to float vec
        for (int p = 0; p < plane; ++p) {
            VecType result = biasVec;
            for (int j = 0; j < inputs.size(); ++j) {
                result = result + VecType::load(srcDatas[j] + p * UNIT);
            }
            VecType::save(srcDatas[0] + p * UNIT, result);
        }
        auto dstZ = output->host<int8_t>() + c * plane * UNIT;
        // TODO: use mOutputZeroPoint instead of 0
        MNNFloat2Int8(srcDatas[0], dstZ, plane, quanParam->scale + c * UNIT, quanParam->minValue, quanParam->maxValue, 0);
        for (int i = 0; i < inputs.size(); ++i) {
            srcDatas[i] += plane * UNIT;
        }
    }
}
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
    quanParam.minValue = mResource->mClampMin;
    mergeAddBiasScaleQuantize(tmp_outputs, outputs[0], &quanParam);
    return NO_ERROR;
};

ConvInt8Winograd::WinoExecution::WinoExecution(Backend *bn, const Convolution2DCommon * common, Tensor* weight, int unitY, int unitX, bool fastgemm)
: CPUConvolution(common, bn), mUnitY(unitY), mUnitX(unitX), mKernelY(common->kernelY()), mKernelX(common->kernelX()) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    mGemmKernel = core->Int8GemmKernel;
    if (fastgemm) {
        mGemmKernel = core->Int8GemmKernelFast;
    }
    
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    int alpha2 = (mUnitY + mKernelY - 1) * (mUnitX + mKernelX - 1);
    int ic4 = UP_DIV(common->inputCount(), SRC_UNIT), oc4 = UP_DIV(common->outputCount(), UNIT);
    mTempInputBuffer.reset(Tensor::createDevice<int8_t>({threadNumber, alpha2, ic4, DST_XUNIT * SRC_UNIT}));
    mTempOutputBuffer.reset(Tensor::createDevice<float>({threadNumber, alpha2, oc4, DST_XUNIT * UNIT}));
    // ROUND_UP(... , sizeof(float)) make midSize align up with float mem
    int midSize = ALIMAX(ROUND_UP(ic4 * SRC_UNIT * alpha2, sizeof(float)), oc4 * UNIT * alpha2 * sizeof(float));
    mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, midSize}));

    chooseTransformFuncs(mKernelY, mKernelX, mUnitY, mUnitX, this, bn);
    WinogradInt8Helper helper(mUnitY, mUnitX, common, core);
    
    mWeight = helper.allocTransformWeight(weight);
    mOffsets.reset(Tensor::createDevice<int32_t>({alpha2, oc4 * UNIT}));
    mValid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    mValid &= backend()->onAcquireBuffer(mOffsets.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    helper.transformWeight(weight, mWeight.get());
    ::memset(mOffsets->host<int32_t>(), 0, mOffsets->size());
#ifdef MNN_USE_SSE
    for (int i = 0; i < alpha2; ++i) {
        for (int oz = 0; oz < common->outputCount(); ++oz) {
            int32_t offset = 0;
            int ozUnit = oz / UNIT, ozRemain = oz % UNIT;
            auto srcZ = mWeight->host<int8_t>() + ((i * oc4 + ozUnit) * ic4 * UNIT + ozRemain) * SRC_UNIT;
            for (int sz = 0; sz < common->inputCount(); ++sz) {
                int szUnit = sz / SRC_UNIT, szRemain = sz % SRC_UNIT;
                offset += srcZ[szUnit * UNIT * SRC_UNIT + szRemain];
            }
            mOffsets->host<int32_t>()[i * oc4 * UNIT + oz] = offset * (-128);
        }
    }
#endif
}
ConvInt8Winograd::WinoExecution::WinoExecution(Backend* bn, const Convolution2DCommon* common, const WinoExecution& exe)
    : CPUConvolution(common, bn), mWeight(exe.mWeight),
    mSourceTransformY(exe.mSourceTransformY), mSourceTransformX(exe.mSourceTransformX),
    mDestTransformY(exe.mDestTransformY), mDestTransformX(exe.mDestTransformX),
    mUnitY(exe.mUnitY), mUnitX(exe.mUnitX), mKernelY(exe.mKernelY), mKernelX(exe.mKernelX),
    mGemmKernel(exe.mGemmKernel), mInputZeroPoint(exe.mInputZeroPoint) {
    
    mTempInputBuffer.reset(Tensor::createDevice<int8_t>(exe.mTempInputBuffer->shape()));
    mTempOutputBuffer.reset(Tensor::createDevice<float>(exe.mTempOutputBuffer->shape()));
    mTransformMidBuffer.reset(Tensor::createDevice<int8_t>(exe.mTransformMidBuffer->shape()));
}
ConvInt8Winograd::WinoExecution::~WinoExecution() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mOffsets.get(), Backend::STATIC);
}
bool ConvInt8Winograd::WinoExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new WinoExecution(bn, mCommon, *this);
    if (!dstExe->valid()) {
        return false;
    }
    *dst = dstExe;
    return true;
}
ErrorCode ConvInt8Winograd::WinoExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    mInputZeroPoint = 0;
    bool success = backend()->onAcquireBuffer(mTempInputBuffer.get(), Backend::DYNAMIC);
    success &= (backend()->onAcquireBuffer(mTempOutputBuffer.get(), Backend::DYNAMIC));
    success &= (backend()->onAcquireBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC));
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
    
    int ow = output->width(), oh = output->height();
    int iw = input->width(), ih = input->height();
    int ic = input->channel(), ic_4 = UP_DIV(ic, UNIT);
    int dc_4 = UP_DIV(output->channel(), UNIT);

    int padY = mPadY, padX = mPadX;
    auto wUnit = UP_DIV(ow, mUnitX), hUnit = UP_DIV(oh, mUnitY);

    auto totalCount   = wUnit * hUnit * input->batch();
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int tileCount    = UP_DIV(totalCount, DST_XUNIT);
    threadNumber     = std::min(threadNumber, tileCount);
    
    auto srcOrigin = input->host<int8_t>();
    auto dstOrigin = output->host<float>();

    auto weight    = mWeight->host<int8_t>();
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
                auto midBuffer1 = midBuffer0 + mTransformMidBuffer->stride(1);
                int dstZStep = mTempInputBuffer->stride(2);
                int unitStep = mTempInputBuffer->stride(1);
                int oyBegin = xIndex / wUnit;
                int oxBegin = xIndex % wUnit;
                int oyEnd = (xIndex + xC-1) / wUnit;
                int remain = xC;
                auto dstS = _srcOrigin;
                for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                    int hIndex = hbIndex % hUnit;
                    int bIndex = hbIndex / hUnit;
                    int step = std::min(wUnit - oxBegin, remain);
                    int srcY  = hIndex * mUnitY - padY;
                    int ey    = ALIMIN(srcY + alphaY, ih) - srcY;
                    int sy    = ALIMAX(0, srcY) - srcY;
                    for (int si=0; si<step; ++si) {
                        auto wIndex = si + oxBegin;
                        int srcX  = wIndex * mUnitX - padX;
                        int sx    = ALIMAX(0, srcX) - srcX;
                        int ex    = ALIMIN(srcX + alphaX, iw) - srcX;
                        int count = UNIT * (ex - sx);
                        auto dst_x = dstS + si * SRC_UNIT;
                        
                        int sourceZStep = iw * ih * input->batch() * UNIT;
                        int sourceYStep = iw * UNIT;
                        auto srcStart = srcOrigin + srcY * sourceYStep + srcX * UNIT + bIndex * iw * ih * UNIT;
                        // when input window exceed limit (so need pad value), copy from src to midbuffer0
                        if (alphaX == 1 || ex - sx != alphaX || ey - sy != alphaY) {
                            ::memset(midBuffer0, mInputZeroPoint, mTransformMidBuffer->stride(1));
                            for (int z = 0; count > 0 && z < ic_4; ++z) {
                                for (int yy = sy; yy < ey; ++yy) {
                                    if (alphaX == 1) {
                                        auto dst_yy = midBuffer0 + (z * alpha2 + yy) * UNIT;
                                        auto src_yy = srcStart + z * sourceZStep + yy * sourceYStep + sx * UNIT;
                                        ::memcpy(dst_yy, src_yy, UNIT);
                                    } else {
                                        auto dst_yy = midBuffer0 + ((z * alphaY + yy) * alphaX + sx) * UNIT;
                                        auto src_yy = srcStart + z * sourceZStep + yy * sourceYStep + sx * UNIT;
                                        ::memcpy(dst_yy, src_yy, count);
                                    }
                                }
                            }
                            srcStart = midBuffer0;
                            sourceZStep = alpha2 * UNIT;
                            sourceYStep = alphaX * UNIT;
                        }
                        
                        if (!conv1d) {
                            for (int i = 0; i < alphaY; ++i) {
                                mSourceTransformX(srcStart + i * sourceYStep, midBuffer1 + i * SRC_UNIT,
                                                  sourceZStep, SRC_UNIT * alphaX, alpha2 * SRC_UNIT, ic_4);
                            }
                            for (int i = 0; i < alphaX; ++i) {
                                mSourceTransformY(midBuffer1 + i * alphaY * SRC_UNIT, dst_x + i * unitStep,
                                                  alpha2 * SRC_UNIT, alphaX * unitStep, dstZStep, UP_DIV(ic, SRC_UNIT));
                            }
                        } else if (alphaX == 1) {
                            mSourceTransformY(srcStart, dst_x, sourceZStep, unitStep, dstZStep, ic_4);
                        } else {
                            mSourceTransformX(srcStart, dst_x, sourceZStep, unitStep, dstZStep, ic_4);
                        }
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
                auto _weightInt8Ptr = weight + i * mWeight->stride(1);
#ifdef MNN_USE_SSE
                MNNInt8ToUInt8(_srcInt8Ptr, mTempInputBuffer->stride(1));
#endif
                QuanPostTreatParameters quanParam;
                quanParam.bias = mOffsets->host<int32_t>() + i * mOffsets->stride(0);
                quanParam.scale = nullptr;
                mGemmKernel((int8_t*)_dstFloatPtr, _srcInt8Ptr, _weightInt8Ptr, mTempInputBuffer->length(2), xC * UNIT * sizeof(float), dc_4, &quanParam, xC);
            }
#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
            {
                auto midBuffer0 = (float*)(mTransformMidBuffer->host<int8_t>() + tId * mTransformMidBuffer->stride(0));
                auto midBuffer1 = (float*)(midBuffer0 + mTransformMidBuffer->stride(1));
                int srcZStep = xC * UNIT;
                int unitStep = dc_4 * xC * UNIT;
                int oyBegin = xIndex / wUnit;
                int oxBegin = xIndex % wUnit;
                int oyEnd = (xIndex + xC-1) / wUnit;
                int remain = xC;
                auto dstS = _dstOrigin;
                for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                    int hIndex = hbIndex % hUnit;
                    int bIndex = hbIndex / hUnit;
                    int step = std::min(wUnit - oxBegin, remain);
                    int dstY = hIndex * mUnitY;
                    int ey = ALIMIN(dstY + mUnitY, oh) - dstY;
                    for (int si=0; si<step; ++si) {
                        auto wIndex = si + oxBegin;
                        auto srcXi = dstS + UNIT * si;
                        int dstX = wIndex * mUnitX;
                        auto dstStart = dstOrigin + (dstX + dstY * ow + bIndex * ow * oh) * UNIT;
                        int ex = ALIMIN(dstX + mUnitX, ow) - dstX;
                        int count = ex * UNIT;
                        
                        auto _dstStart = dstStart;
                        int dstZStep = oh * ow * output->batch() * UNIT, dstYStep = ow * UNIT;
                        if (ex != mUnitX || (alphaX == 1 && ey != mUnitY)) {
                            dstZStep = mUnitY * mUnitX * UNIT;
                            dstYStep = mUnitX * UNIT;
                            _dstStart = midBuffer1;
                        }
                        if (!conv1d) {
                            // transform read along Y direction, then write along Y direction
                            for (int i = 0; i < alphaX; ++i) {
                                mDestTransformY(srcXi + i * unitStep, midBuffer0 + i * UNIT,
                                                alphaX * unitStep, srcZStep, alphaX * UNIT, alphaX * mUnitY * UNIT, dc_4);
                            }
                            // transform read along X direction, then write along X direction
                            for (int i = 0; i < ey; ++i) {
                                mDestTransformX(midBuffer0 + i * alphaX * UNIT, _dstStart + i * dstYStep,
                                                UNIT, alphaX * mUnitY * UNIT, UNIT, dstZStep, dc_4);
                            }
                        } else if (alphaX == 1) {
                            mDestTransformY(srcXi, _dstStart, unitStep, srcZStep, dstYStep, dstZStep, dc_4);
                        } else {
                            for (int i = 0; i < ey; ++i) {
                                mDestTransformX(srcXi + i * alphaX * unitStep, _dstStart + i * dstYStep,
                                                unitStep, srcZStep, UNIT, dstZStep, dc_4);
                            }
                        }
                        if (ex != mUnitX || (alphaX == 1 && ey != mUnitY)) {
                            for (int z = 0; z < dc_4; ++z) {
                                for (int yy = 0; yy < ey; ++yy) {
                                    auto srcYAddr = _dstStart + (z * mUnitY + yy) * mUnitX * UNIT;
                                    auto dstYAddr = dstStart + (z * oh * output->batch() + yy) * ow * UNIT;
                                    ::memcpy(dstYAddr, srcYAddr, count * sizeof(float));
                                }
                            }
                        }
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

    return NO_ERROR;
}

bool ConvInt8Winograd::bestWinogradUnit(const Convolution2D *convOp, const Tensor *input, const Tensor* weightSrc, const Tensor *output, Backend* bn, std::vector<UnitAttr>& unitAttrs) {
    auto quantAttr = TensorUtils::getDescribe(input)->quantAttr;
    if (quantAttr == nullptr) {
        return false;
    }
    //MNN_PRINT("%f %f\n", quantAttr->min, quantAttr->max);
    auto common = convOp->common();
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    int kernelY = common->kernelY(), kernelX = common->kernelX();
    int oh = output->height(), ow = output->width(), oc = common->outputCount(), ic = common->inputCount();
    
    const int CONV_WINOGRAD_MAX_KERNEL = 3, CONV_WINOGRAD_ALPHA = 4;
    using Vec = std::vector<std::pair<int, int>>;
    auto partitionKernelFunc = [=](int kernel, bool range = false) -> Vec {
        Vec partition;
        for (int i = 0, count; i < kernel; i += count) {
            count = (range ? 1 : ALIMIN(kernel - i, CONV_WINOGRAD_MAX_KERNEL));
            partition.emplace_back(i, count);
        }
        return partition;
    };
    typedef std::tuple<bool, float, std::vector<UnitAttr>> RES;
    std::function<RES(const Vec&, const Vec&)> partitionFunc = [=](const Vec& yAttrs, const Vec& xAttrs) {
        auto core = static_cast<CPUBackend*>(bn)->int8Functions();
        bool support = true;
        float optMAC = 0;
        std::vector<UnitAttr> unitAttrs;
        for (auto& yAttr : yAttrs) {
            for (auto& xAttr : xAttrs) {
                int ky = yAttr.second, uy = (ky == 1 ? 1 : (CONV_WINOGRAD_ALPHA - ky + 1)), alphaY = uy + ky - 1;
                int kx = xAttr.second, ux = (kx == 1 ? 1 : (CONV_WINOGRAD_ALPHA - kx + 1)), alphaX = ux + kx - 1;
                UnitAttr unitAttr {yAttr.first, ky, xAttr.first, kx, uy, ux};
                if (uy != 1 || ux != 1) {
                    std::shared_ptr<Tensor> weight;
                    support &= (chooseTransformFuncs(ky, kx, uy, ux, nullptr, bn));
                    support &= (!WinogradInt8Helper::featureOverflow(input, alphaY, alphaX));
                    support &= (extractWeight(weightSrc, weight, common, unitAttr, bn) == NO_ERROR);
                    auto subCommon = createCommon(common, {unitAttr.kySize, unitAttr.kxSize}, {});
                    support &= (!WinogradInt8Helper::weightOverflow(weight.get(), uy, ux, subCommon->first, core));
                    if (!support) {
                        break;
                    }
                    optMAC += (float)oc * ic * alphaY * alphaX * UP_DIV(oh, uy) * UP_DIV(ow, ux);
                } else {
                    optMAC += (float)oc * ic * ky * kx * oh * ow;
                }
                unitAttrs.push_back(unitAttr);
            }
            if (!support) {
                break;
            }
        }
        return std::make_tuple(support, optMAC, unitAttrs);
    };
    unitAttrs.clear();
    float bestMAC = 1.0f * oc * ic * kernelY * kernelX * oh * ow;
    auto yAttrs = partitionKernelFunc(kernelY), yRangeAttrs = partitionKernelFunc(kernelY, true);
    auto xAttrs = partitionKernelFunc(kernelX), xRangeAttrs = partitionKernelFunc(kernelX, true);
    // partition to 2D + 1D
    auto res = partitionFunc(yAttrs, xAttrs);
    if (std::get<0>(res) && std::get<1>(res) < bestMAC) {
        bestMAC = std::get<1>(res);
        unitAttrs = std::get<2>(res);
    }
    // only partition to 1xN
    res = partitionFunc(yRangeAttrs, xAttrs);
    if (std::get<0>(res) && std::get<1>(res) < bestMAC) {
        bestMAC = std::get<1>(res);
        unitAttrs = std::get<2>(res);
    }
    // only partition to Nx1
    res = partitionFunc(yAttrs, xRangeAttrs);
    if (std::get<0>(res) && std::get<1>(res) < bestMAC) {
        bestMAC = std::get<1>(res);
        unitAttrs = std::get<2>(res);
    }
    return unitAttrs.size() > 0;
}

} /* MNN */
