#include "ConvInt8Winograd.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "ConvOpt.h"
#include "Int8FunctionsOpt.h"
#include "CommonOptFunction.h"
#include "MNN/AutoTime.hpp"
#include "math/Vec.hpp"
#include "math/WingoradGenerater.hpp"
#include <map>
#include <numeric>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

std::shared_ptr<ConvInt8Winograd::WinoResource> ConvInt8Winograd::makeWinoResource(const int8_t* originWeight, std::shared_ptr<Tensor> scaleFloat, const int32_t* attr, Backend* backend, int oc, int ic, int kernelY, int kernelX) {
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int oc4 = UP_DIV(oc, UNIT), ic4 = UP_DIV(ic, SRC_UNIT);
    int kySize = attr[2], kxSize = attr[3], unitY = attr[4], unitX = attr[5]; attr += 6;
    int alphaY = kySize + unitY - 1, alphaX = kxSize + unitX - 1, alpha2 = alphaY * alphaX;
    
    std::shared_ptr<Tensor> weight, offsets, scales, inputScales;
    weight.reset(Tensor::createDevice<int8_t>({alpha2, oc4, ic4, UNIT, SRC_UNIT}));
    offsets.reset(Tensor::createDevice<float>({alpha2, oc4, UNIT}));
    scales.reset(Tensor::createDevice<float>({alpha2, oc4 * UNIT}));
    inputScales.reset(Tensor::createDevice<float>({alpha2, UNIT}));
    
    auto allocTensors = [=](std::vector<std::shared_ptr<Tensor>> tensors) -> bool {
        bool success = true;
        for (const auto& t : tensors) {
            success &= backend->onAcquireBuffer(t.get(), Backend::STATIC);
        }
        return success;
    };
    if (!allocTensors({weight, offsets, scales, inputScales})) {
        MNN_ERROR("Memory not enough\n");
        return nullptr;
    }
    ::memset(weight->host<int8_t>(), 0, weight->size());
    ::memset(offsets->host<float>(), 0, offsets->size());
    ::memset(scales->host<float>(), 0, scales->size());
    auto inputScaleData = (const float*)attr; attr += alpha2;
    auto inputPointData = (const int32_t*)attr; attr += alpha2;
    auto weightScaleData = (const float*)attr; attr += alpha2 * oc;
    for (int i = 0; i < alpha2; ++i) {
        auto scale = 1.0f / inputScaleData[i];
        for (int u = 0; u < UNIT; ++u) {
            inputScales->host<float>()[i * UNIT + u] = scale;
        }
    }
    
    std::shared_ptr<Tensor> originWeightFloat, weightFloat;
    originWeightFloat.reset(Tensor::createDevice<float>({oc, ic, kySize, kxSize}));
    weightFloat.reset(Tensor::createDevice<float>({alpha2, oc, ic, 1, 1}));
    if (!allocTensors({originWeightFloat, weightFloat})) {
        MNN_ERROR("Memory not enough\n");
        return nullptr;
    }
    for (int c = 0; c < oc * ic; ++c) {
        for (int h = 0; h < kySize; ++h) {
            for (int w = 0; w < kxSize; ++w) {
                auto srcInt8 = originWeight[(c * kernelY + h) * kernelX + w];
                auto scale = scaleFloat->host<float>()[c / ic];
                originWeightFloat->host<float>()[(c * kySize + h) * kxSize + w] = srcInt8 * scale;
            }
        }
    }
    Math::WinogradGenerater generator({unitY, unitX}, {kySize, kxSize}, 1, true);
    generator.transformWeight(weightFloat.get(), originWeightFloat.get(), true);
    
    for (int a = 0; a < alpha2; ++a) {
        for (int oz = 0; oz < oc; ++oz) {
            int oz4 = oz / UNIT, ozRemain = oz % UNIT;
            int offset_int32 = 0;
            float offset = 0.f;
            float scale = weightScaleData[a * oc + oz];
            for (int sz = 0; sz < ic; ++sz) {
                int sz4 = sz / SRC_UNIT, szRemain = sz % SRC_UNIT;
                int index = (((a * oc4 + oz4) * ic4 + sz4) * UNIT + ozRemain) * SRC_UNIT + szRemain;
                float srcData = weightFloat->host<float>()[(a * oc + oz) * ic + sz];
                // -ffast-math may cause inexact input then wrong rounded result, add eps to avoid this
                float eps = ((srcData/scale) > 0 ? 1 : -1) * 1e-6;
                auto quanData = (int8_t)ALIMIN(ALIMAX(roundf(srcData / scale + eps), -127), 127);
                weight->host<int8_t>()[index] = quanData;
                offset += quanData * (-inputPointData[a]);
#ifdef MNN_USE_SSE
                offset += quanData * (-128);
#endif
            }
            offsets->host<float>()[a * oc4 * UNIT + oz] = offset * scale * inputScaleData[a];
            scales->host<float>()[a * oc4 * UNIT + oz] = scale * inputScaleData[a];
        }
    }
    backend->onReleaseBuffer(originWeightFloat.get(), Backend::STATIC);
    backend->onReleaseBuffer(weightFloat.get(), Backend::STATIC);
    
    std::shared_ptr<WinoResource> resource(new WinoResource);
    resource->weight = weight;
    resource->offsets = offsets;
    resource->scales = scales;
    resource->transInputScales = inputScales;
    std::vector<int32_t> inputZeroPoints(inputPointData, inputPointData + alpha2);
    resource->transInputZeroPoints = inputZeroPoints;
    resource->backend = backend;
    return resource;
}

ConvInt8Winograd::ConvInt8Winograd(Backend *b, const Convolution2D *convOp, std::shared_ptr<ResourceInt8> res) : CPUConvolution(convOp->common(), b), mResource(res) {
    int oc = mCommon->outputCount(), ic = mCommon->inputCount();
    int kernelY = mCommon->kernelY(), kernelX = mCommon->kernelX();
    auto core = static_cast<CPUBackend*>(b)->int8Functions();
    
    auto attr = convOp->symmetricQuan()->winogradAttr()->data();
    int version = *(attr++), unitNum = *(attr++);
    if (version != 0) {
        MNN_ERROR("ConvInt8 winograd attr proto version must be 1\n");
        mValid = false;
        return;
    }
    //FUNC_PRINT(convOp->symmetricQuan()->winogradAttr()->size());
    auto weightData = res->mWeightInt8->host<int8_t>();
    for (int i = 0; i < unitNum; ++i) {
        int unitSize = *(attr++);
        int kyStart = attr[0], kxStart = attr[1], kySize = attr[2], kxSize = attr[3], unitY = attr[4], unitX = attr[5];
        int alphaY = kySize + unitY - 1, alphaX = kxSize + unitX - 1;
        // TODO: support alphaY != alphaX
        if (alphaY != alphaX) {
            MNN_ERROR("ConvInt8 winograd only support ky==kx && unitY==unitX\n");
            mValid = false;
            return;
        }
        std::shared_ptr<Tensor> tempInput, tempOutput;
        auto winoRes = makeWinoResource(weightData + kyStart * kernelY + kxStart, mResource->mOriginScale, attr, b, oc, ic, kernelY, kernelX);
        attr += unitSize;
        std::shared_ptr<WinoExecution> exe(new WinoExecution(winoRes, kySize, kxSize, unitY, unitX, oc, ic));
        mUnits.push_back({kyStart, kxStart, tempInput, tempOutput, exe});
    }
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
    CPUConvolution::onResize(inputs, outputs);
    
    mInputFloat.reset(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE_C4));
    mValid = backend()->onAcquireBuffer(mInputFloat.get(), Backend::DYNAMIC);
    if (!mValid) {
        return OUT_OF_MEMORY;
    }
    
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    UNIT = gcore->pack;
    
    auto input = mInputFloat.get(), output = outputs[0];
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
        auto srcChunk = TensorUtils::getDescribeOrigin(input)->mem->chunk() + (sy * iw + sx) * UNIT;
        unit.input.reset(Tensor::createDevice<float>({batch, ic, ih - sy, iw - sx}, Tensor::CAFFE_C4));
        TensorUtils::getDescribeOrigin(unit.input.get())->mem = (new CPUMemObj(nullptr, srcChunk, 0));
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
    backend()->onReleaseBuffer(mInputFloat.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

static void mergeAddBiasScaleQuantize(const std::vector<Tensor*>& inputs, Tensor* output, const QuanPostTreatParameters* quanParam, CPUBackend* cpuBn, int zeroPoint) {
    auto core = cpuBn->functions();
    auto coreInt8 = cpuBn->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    coreInt8->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    UNIT = core->pack;
    
    int countC4 = UP_DIV(output->channel(), UNIT), plane = output->height() * output->width() * output->batch();
    auto mergeFloat = inputs[0]->host<float>();
    for (int i = 1; i < inputs.size(); ++i) {
        core->MNNMatrixAdd(mergeFloat, mergeFloat, inputs[i]->host<float>(), plane * countC4, 0, 0, 0, 1);
    }
    std::vector<float> fakeScale(countC4 * UNIT, 1);
    core->MNNScaleAndAddBias(mergeFloat, mergeFloat, quanParam->biasFloat, fakeScale.data(), plane, countC4);
    coreInt8->MNNFloat2Int8(mergeFloat, output->host<int8_t>(), plane * countC4, quanParam->scale, quanParam->minValue, quanParam->maxValue, zeroPoint);
}

// AVX: 8 -> 16, arm32/64: 4 -> 16, AVX512: 16 -> 16, arm82: 4 -> 4
static void _reorderCommon(float* dst, const float* src, size_t area, size_t depth, int* areaOffset, int uFrom, int uTo) {
    if (uFrom == 1 && uTo == 4) {
        MNNPackC4((float*)dst, (const float*)src, area, depth, areaOffset);
        return;
    }
    size_t srcOffset = areaOffset[0], dstOffset = areaOffset[1];
    int z = 0;
    if (uFrom == 2 && uTo == 4) {
        for (; z + 3 < depth; z += 4) {
            auto srcZ = src + z * srcOffset;
            auto dstZ = dst + z * dstOffset;
            for (int i = 0; i < area; ++i) {
                dstZ[i * 4] = srcZ[i * 2];
                dstZ[i * 4 + 1] = srcZ[i * 2 + 1];
                dstZ[i * 4 + 2] = srcZ[srcOffset * 2 + i * 2];
                dstZ[i * 4 + 3] = srcZ[srcOffset * 2 + i * 2 + 1];
            }
        }
    }
    // Other UNIT != SRC_UNIT case if exist, and remain
    for (; z < depth; ++z) {
        auto dstZ = dst + (z / uTo) * dstOffset * uTo + (z % uTo);
        auto srcZ = src + (z / uFrom) * srcOffset * uFrom + (z % uFrom);
        for (int i = 0; i < area; ++i) {
            dstZ[i * uTo] = srcZ[i * uFrom];
        }
    }
    int depthLast = depth % uTo;
    if (depthLast != 0) {
        int zero = 0;
#ifdef MNN_USE_SSE
        zero = 128;
#endif
        auto dstZ = dst + (depth / uTo) * dstOffset + depthLast;
        for (int i = 0; i < area; ++i) {
            ::memset(dstZ + i * uTo, zero, (uTo - depthLast) * sizeof(float));
        }
    }
}

ErrorCode ConvInt8Winograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto bn = static_cast<CPUBackend*>(backend());
    auto core = bn->int8Functions();
    auto gcore = bn->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    UNIT = gcore->pack;
    // scale, zero, min, max
    auto inputQuant = TensorUtils::getQuantInfo(inputs[0]);
    auto outputQuant = TensorUtils::getQuantInfo(outputs[0]);
    if (TensorUtils::getDescribe(inputs[0])->quantAttr.get() == nullptr) {
        inputQuant = {(float)mResource->mInputScale,
            (float)mResource->mInputZeroPoint,
            (float)mResource->mClampMin,
            (float)mResource->mClampMax,
        };
        outputQuant = {(float)mResource->mOutputScale,
            (float)mResource->mOutputZeroPoint,
            (float)mResource->mClampMin,
            (float)mResource->mClampMax,
        };
    }

    std::vector<float> scale(UNIT, inputQuant[0]);
    int size = bn->getTensorSize(mInputFloat.get());
    core->MNNInt8ScaleToFloat(mInputFloat->host<float>(), inputs[0]->host<int8_t>(), scale.data(), size / UNIT, inputQuant[1]);
    std::vector<Tensor*> tmp_outputs;
    for (auto& unit : mUnits) {
        unit.input->buffer().host = TensorUtils::getDescribeOrigin(unit.input.get())->mem->chunk().ptr();
        auto ret = unit.runner->onExecute({unit.input.get()}, {unit.output.get()});
        if (ret != NO_ERROR) {
            return ret;
        }
        tmp_outputs.push_back(unit.output.get());
    }
    QuanPostTreatParameters quanParam;
    scale.assign(UNIT, 1.0 / outputQuant[0]);
    quanParam.scale = scale.data();
    // For winograd Int8, will not treat origin bias to int32, use float directly
    quanParam.biasFloat = mResource->mOriginBias->host<float>();
    quanParam.maxValue = outputQuant[3];
    if (mResource->mRelu) {
        quanParam.minValue = outputQuant[1];
    } else {
        quanParam.minValue = outputQuant[2];
    }
    mergeAddBiasScaleQuantize(tmp_outputs, outputs[0], &quanParam, bn, outputQuant[1]);
    return NO_ERROR;
};

ConvInt8Winograd::WinoExecution::WinoExecution(std::shared_ptr<WinoResource> res, int kernelY, int kernelX, int unitY, int unitX, int outputCount, int inputCount)
: Execution(res->backend), mWinoResource(res), mUnitY(unitY), mUnitX(unitX), mKernelY(kernelY), mKernelX(kernelX) {
    auto core = static_cast<CPUBackend*>(res->backend)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(res->backend)->functions();
    
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    UNIT = gcore->pack;

    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    int alphaY = mUnitY + mKernelY - 1, alphaX = mUnitX + mKernelX - 1, alpha2 = alphaY * alphaX;
    int ic4 = UP_DIV(inputCount, SRC_UNIT), oc4 = UP_DIV(outputCount, UNIT);
    mTempInputBuffer.reset(Tensor::createDevice<int8_t>({threadNumber, alpha2, ic4, DST_XUNIT * SRC_UNIT}));
    mTempOutputBuffer.reset(Tensor::createDevice<float>({threadNumber, alpha2, oc4, DST_XUNIT * UNIT}));
    int midSize = alpha2 * DST_XUNIT * ALIMAX(ROUND_UP(inputCount, UNIT), oc4 * UNIT);
    mTransformMidBuffer.reset(Tensor::createDevice<float>({threadNumber, 3, midSize}));
}
ConvInt8Winograd::WinoExecution::WinoExecution(Backend* bn, const WinoExecution& exe)
    : Execution(bn), mWinoResource(exe.mWinoResource),
    mUnitY(exe.mUnitY), mUnitX(exe.mUnitX), mKernelY(exe.mKernelY), mKernelX(exe.mKernelX),
    mPadY(exe.mPadY), mPadX(exe.mPadX) {
    
    mTempInputBuffer.reset(Tensor::createDevice<int8_t>(exe.mTempInputBuffer->shape()));
    mTempOutputBuffer.reset(Tensor::createDevice<float>(exe.mTempOutputBuffer->shape()));
    mTransformMidBuffer.reset(Tensor::createDevice<float>(exe.mTransformMidBuffer->shape()));
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
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto coreInt8 = static_cast<CPUBackend*>(backend())->int8Functions();
    auto input = inputs[0], output = outputs[0];
    
    int alphaY = mKernelY + mUnitY - 1, alphaX = mKernelX + mUnitX - 1, alpha2 = alphaY * alphaX;
    bool conv1d = (alphaY == 1 || alphaX == 1);
    int UNIT, SRC_UNIT, DST_XUNIT;
    coreInt8->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    UNIT = core->pack;
    
    auto gemmFunc = coreInt8->Int8GemmKernel;
    CoreFunctions::WinoUnrollTransFunc srcTransXFunc = nullptr, srcTransYFunc = nullptr;
    if (mKernelX != 1) {
        srcTransXFunc = core->chooseWinoSourceUnrollTransform(alphaX, alphaX);
    }
    if (mKernelY != 1) {
        srcTransYFunc = core->chooseWinoSourceUnrollTransform(alphaY, alphaY);
    }
    
#define MAX_UNIT 8
    CoreFunctions::WinoUnrollDestTransFunc dstTransXFunc[MAX_UNIT + 1], dstTransYFunc[MAX_UNIT + 1];
    if (mKernelX != 1) {
        core->chooseWinoDestUnrollTransform(dstTransXFunc, MAX_UNIT + 1, alphaX, mUnitX);
    }
    if (mKernelY != 1) {
        core->chooseWinoDestUnrollTransform(dstTransYFunc, MAX_UNIT + 1, alphaY, mUnitY);
    }
    
    int ow = output->width(), oh = output->height();
    int iw = input->width(), ih = input->height();
    int ic = input->channel(), ic_4 = UP_DIV(ic, UNIT);
    int dc_4 = UP_DIV(output->channel(), UNIT);

    int padY = mPadY, padX = mPadX;
    auto wUnit = UP_DIV(ow, mUnitX), hUnit = UP_DIV(oh, mUnitY);
    int batch = output->batch();

    auto totalCount   = wUnit * hUnit * batch;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int tileCount    = UP_DIV(totalCount, DST_XUNIT);
    threadNumber     = std::min(threadNumber, tileCount);

    auto src_trans_func = [&](float* dstOrigin, const float* srcOrigin, float* buffer, int xIndex, int xC) {
        int bufSize = mTransformMidBuffer->stride(1);
        auto midBuffer0 = buffer, midBuffer1 = midBuffer0 + bufSize;
        int oybBegin = xIndex / wUnit;
        int oxBegin = xIndex % wUnit;
        int oybEnd = (xIndex + xC-1) / wUnit;
        int remain = xC;
        for (int hbIndex=oybBegin; hbIndex <= oybEnd; ++hbIndex) {
            auto hIndex = hbIndex % hUnit;
            auto bIndex = hbIndex / hUnit;
            auto bOffset = iw * ih * UNIT * bIndex;
            auto srcBatch = srcOrigin + bOffset;
            int dstZStep = DST_XUNIT * UNIT, unitStep = dstZStep * ic_4;
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
                auto dst_x = dstOrigin + si * UNIT;
                
                int sourceZStep = iw * ih * UNIT * batch, sourceYStep = iw * UNIT;
                auto srcStart = srcBatch + srcY * sourceYStep + srcX * UNIT;
                // when input window exceed limit (so need pad value), copy from src to midbuffer0
                if (ex - sx != alphaX || ey - sy != alphaY) {
                    ::memset(midBuffer0, 0, alpha2 * ic_4 * UNIT * sizeof(float));
                    int count = UNIT * (ex - sx);
                    for (int z = 0; count > 0 && z < ic_4; ++z) {
                        for (int yy = sy; yy < ey; ++yy) {
                            auto dst_yy = midBuffer0 + ((z * alphaY + yy) * alphaX + sx) * UNIT;
                            auto src_yy = srcStart + z * sourceZStep + yy * sourceYStep + sx * UNIT;
                            ::memcpy(dst_yy, src_yy, count * sizeof(float));
                        }
                    }
                    srcStart = midBuffer0;
                    sourceZStep = alpha2 * UNIT;
                    sourceYStep = alphaX * UNIT;
                }
                for (int sz = 0; sz < ic_4; ++sz) {
                    for (int s = 0; s < sStep; ++s) {
                        auto dst = dst_x + sz * dstZStep + s * UNIT;
                        auto src = srcStart + sz * sourceZStep + s * mUnitX * UNIT;
                        srcTransXFunc(src, midBuffer1, sourceYStep, alphaX * UNIT, UNIT, UNIT);
                        srcTransYFunc(midBuffer1, dst, UNIT, unitStep, alphaX * UNIT, alphaX * unitStep);
                    }
                }
                si += sStep;
            }
            oxBegin = 0;
            remain -= step;
            dstOrigin += UNIT * step;
        }
        
    };
    
        auto srcOrigin = input->host<float>();
        auto dstOrigin = output->host<float>();

        auto weight    = mWinoResource->weight->host<int8_t>();
        std::vector<float> xkernelSum(DST_XUNIT, 0);
        std::vector<float> wKernelSum(dc_4 * UNIT, 0);
        std::vector<float> reluThred = {-std::numeric_limits<float>().max(), std::numeric_limits<float>().max()};
        
        auto tFunction = [&](int tId) {
            auto _srcOrigin = mTempInputBuffer->host<int8_t>() + tId * mTempInputBuffer->stride(0);
            auto _dstOrigin = mTempOutputBuffer->host<float>() + tId * mTempOutputBuffer->stride(0);
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex  = (int)tIndex * DST_XUNIT;
                int xReamin = totalCount - xIndex;
                int xC      = xReamin > DST_XUNIT ? DST_XUNIT : xReamin;

                int bufSize = mTransformMidBuffer->stride(1);
                auto buffer0 = mTransformMidBuffer->host<float>() + tId * mTransformMidBuffer->stride(0);
                auto buffer1 = buffer0 + bufSize, buffer2 = buffer1 + bufSize;
    #ifndef MNN_WINO_TRANFORM_TEST_CLOSE
                src_trans_func(buffer2, srcOrigin, buffer0, xIndex, xC);
    #endif
                ::memset(buffer1, 0, dc_4 * UNIT * sizeof(float));
                // Multi
                for (int i = 0; i < alpha2; ++i) {
                    auto _srcInt8Ptr = _srcOrigin + i * mTempInputBuffer->stride(1);
                    
                    auto scaleVec = mWinoResource->transInputScales->host<float>() + i * UNIT;
                    int zeroPoint = mWinoResource->transInputZeroPoints[i];
                    coreInt8->MNNFloat2Int8(buffer2 + i * DST_XUNIT * ic_4 * UNIT, (UNIT == SRC_UNIT ? _srcInt8Ptr: (int8_t*)buffer0), ic_4 * DST_XUNIT, scaleVec, -127, 127, zeroPoint);
                    if (UNIT != SRC_UNIT) {
                        int areaOffset[] = {DST_XUNIT, DST_XUNIT}, byte = sizeof(float);
                        _reorderCommon((float*)_srcInt8Ptr, buffer0, DST_XUNIT, UP_DIV(ic, byte), areaOffset, UNIT / byte, SRC_UNIT / byte);
                    }
                    
                    auto _dstFloatPtr = _dstOrigin + i * dc_4 * xC * UNIT;
                    auto _weightInt8Ptr = weight + i * mWinoResource->weight->stride(0);
                    QuanPostTreatParameters quanParam;
                    quanParam.biasFloat = (mWinoResource->offsets->host<float>() + i * mWinoResource->offsets->stride(0));
                    quanParam.useInt8 = 0;
                    quanParam.srcKernelSum = xkernelSum.data();
                    quanParam.weightQuanBias = wKernelSum.data();
                    quanParam.fp32minmax = reluThred.data();
                    quanParam.scale = mWinoResource->scales->host<float>() + i * dc_4 * UNIT;
                    quanParam.extraScale = nullptr;
                    gemmFunc((int8_t*)_dstFloatPtr, _srcInt8Ptr, _weightInt8Ptr, mTempInputBuffer->length(2), xC * UNIT * sizeof(float), dc_4, &quanParam, xC);
                }
    #ifndef MNN_WINO_TRANFORM_TEST_CLOSE
                {
                    auto midBuffer0 = buffer0;
                    auto midBuffer1 = (float*)((int8_t*)midBuffer0 + mTransformMidBuffer->stride(1));
                    int srcZStep = xC * UNIT;
                    int unitStep = dc_4 * xC * UNIT;
                    int oybBegin = xIndex / wUnit;
                    int oxBegin = xIndex % wUnit;
                    int oybEnd = (xIndex + xC-1) / wUnit;
                    int remain = xC;
                    auto dstS = _dstOrigin;
                    for (int hbIndex=oybBegin; hbIndex <= oybEnd; ++hbIndex) {
                        int hIndex = hbIndex % hUnit;
                        int bIndex = hbIndex / hUnit;
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
                            auto dstStart = dstOrigin + (dstX + dstY * ow + bIndex * ow * oh) * UNIT;
                            int ex = ALIMIN(dstX + mUnitX, ow) - dstX;
                            int count = ex * UNIT;
                            
                            auto _dstStart = dstStart;
                            int dstZStep = oh * ow * batch * UNIT, dstYStep = ow * UNIT;
                            if (ex != mUnitX || (alphaX == 1 && ey != mUnitY)) {
                                dstZStep = mUnitY * mUnitX * UNIT;
                                dstYStep = mUnitX * UNIT;
                                _dstStart = midBuffer1;
                            }
                            for (int z = 0; z < dc_4; ++z) {
                                for (int x = 0; x < sStep; ++x) {
                                    auto srcXiZ = srcXi + z * srcZStep + x * UNIT;
                                    auto _dstStartZ = _dstStart + z * dstZStep + x * mUnitX * UNIT;
                                    dstTransYFunc[alphaX](srcXiZ, midBuffer0, nullptr, nullptr, unitStep, UNIT, alphaX * unitStep, alphaX * UNIT);
                                    dstTransXFunc[ey](midBuffer0, _dstStartZ, nullptr, nullptr, alphaX * UNIT, dstYStep, UNIT, UNIT);
                                }
                            }
                            if (ex != mUnitX || (alphaX == 1 && ey != mUnitY)) {
                                for (int z = 0; z < dc_4; ++z) {
                                    for (int yy = 0; yy < ey; ++yy) {
                                        auto srcYAddr = _dstStart + (z * mUnitY + yy) * mUnitX * UNIT;
                                        auto dstYAddr = dstStart + z * ow * oh * batch * UNIT + yy * ow * UNIT;
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
