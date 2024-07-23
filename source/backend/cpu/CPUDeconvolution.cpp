//
//  CPUDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDeconvolution.hpp"
#include "core/BufferAllocator.hpp"
#include "CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "core/AutoStorage.h"
#include "math/Matrix.hpp"
#include "core/TensorUtils.hpp"
#include "core/ConvolutionCommon.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "compute/DeconvolutionWithStride.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {

CPUDeconvolutionBasic::CPUDeconvolutionBasic(const Tensor* input, const Op* convOp, Backend* b)
    : CPUConvolution(convOp->main_as_Convolution2D()->common(), b) {
    mSrcCount = input->channel();
    mPostParameters = getPostParameters();
}

ErrorCode CPUDeconvolutionBasic::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mCommon);
    mPadY = pad.second;
    mPadX = pad.first;
    return NO_ERROR;
}

CPUDeconvolutionCommon::CPUDeconvolutionCommon(const Tensor* input, const Op* convOp, Backend* b, bool dynamicWeight)
    : CPUDeconvolutionBasic(input, convOp, b) {
    auto conv2D     = convOp->main_as_Convolution2D();
    int outputCount = mCommon->outputCount();
    auto core = static_cast<CPUBackend*>(b)->functions();
    mDynamicWeight = dynamicWeight;
    mBias.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, core->pack) * core->pack}));
    if (dynamicWeight) {
        return;
    }
    bool success = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->length(0) * core->bytes);
    if (core->bytes == 4) {
        ::memcpy(mBias->host<float>(), conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));
    } else {
        core->MNNFp32ToLowp(conv2D->bias()->data(), mBias->host<int16_t>(), conv2D->bias()->size());
    }
}

CPUDeconvolutionCommon::~CPUDeconvolutionCommon() {
    // Do nothing
}

// Float Weight.
static void _transformWeight(const uint8_t* tempWeight, uint8_t* dest, int outputCount, int srcCount, int fh, int fw,
                             uint8_t* cache, const CoreFunctions* core) {
    auto outputC4 = UP_DIV(outputCount, core->pack);
    int offset[] = {
        (int)(fw * fh),
        (int)(fw * fh),
    };
    // c, n, h, w-> c, n/4 * 4, h, w
    for (int c=0; c<srcCount; ++c) {
        auto dst = cache + c * outputC4 * fw * fh * core->pack * core->bytes;
        auto src = tempWeight + c * outputCount * fw * fh * core->bytes;
        core->MNNPackCUnit((float*)dst, (const float*)src, fw*fh, outputCount, offset);
    }
    //printf("%d - %d - %d - %d\n", outputCount, srcCount, fh, fw);
    core->MNNPackForMatMul_B((float*)dest, (const float*)cache, outputC4 * fw * fh * core->pack, srcCount, false);
}
// Int8 Weight.
static void _reorderWeightInt8(Backend* bn, const Convolution2DCommon* common, const int8_t* srcPtr,
                               std::shared_ptr<Tensor>& weight) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    auto gcore =  static_cast<CPUBackend*>(bn)->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    UNIT = gcore->pack;

    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    std::vector<int> shape = {UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount, UNIT, SRC_UNIT};

    weight.reset(Tensor::createDevice<int8_t>(shape));
    bool succ = bn->onAcquireBuffer(weight.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return;
    }
    auto dstPtr = weight->host<int8_t>();
    ::memset(dstPtr, 0, weight->size());

    int icDiv = UP_DIV(ic, SRC_UNIT);
     for (int k = 0; k < kernelCount; ++k) {
        auto srcK = srcPtr + k;
        auto dstK = dstPtr + k * SRC_UNIT * UNIT * icDiv;
        for (int x = 0; x < oc; ++x) {
            int xout = x / UNIT;
            int xin = x % UNIT;
            auto srcY = srcK + x * kernelCount;
            auto dstY = dstK + xout * SRC_UNIT * UNIT * icDiv * kernelCount + xin * SRC_UNIT;
            for (int y = 0; y < ic; ++y) {
                int yout = y / SRC_UNIT;
                int yin = y % SRC_UNIT;

                const int dstIndex = yout * SRC_UNIT * UNIT + yin;
                const int srcIndex = y * oc * kernelCount;
                dstY[dstIndex] = srcY[srcIndex];
            }
        }
    }
}
CPUDeconvolution::CPUDeconvolution(const Tensor* input, const Op* convOp, Backend* backend, bool dynamicWeight)
    : MNN::CPUDeconvolutionCommon(input, convOp, backend, dynamicWeight) {
    auto core               = static_cast<CPUBackend*>(backend)->functions();
    auto coreInt8           = static_cast<CPUBackend*>(backend)->int8Functions();
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    int UNIT, SRC_UNIT, DST_XUNIT;
    coreInt8->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    bool ModeInt8        =  false;

    if (CPUBackend::getDataType(input) == DataType_DT_INT8 || input->getType().bytes() == 1) {
        eP = DST_XUNIT;
        lP = SRC_UNIT;
        hP = UNIT;
        ModeInt8 = true;
    }
    auto conv2d                  = convOp->main_as_Convolution2D();
    auto layer                   = conv2d->common();
    int outputCount              = layer->outputCount();
    const auto outputChannleUp4  = UP_DIV(outputCount, hP) * hP;
    int fw                  = layer->kernelX();
    int fh                  = layer->kernelY();
    int srcCount            = mSrcCount;
    mParam.fh = fh;
    mParam.fw = fw;
    mParam.srcCount = srcCount;
    mParam.outputCount = outputCount;
    auto outputAlign = UP_DIV(layer->outputCount(), core->pack) * core->pack * fw * fh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
    std::shared_ptr<Tensor> cache(Tensor::createDevice<float>({outputAlign * srcCount}));
    if (dynamicWeight) {
        mOrigin.reset(new CPUDeconvolutionOrigin(input, mWeight.get(), convOp, backend, ModeInt8));
        mWeightTransformCache = cache;
        return;
    }

    const float* tempWeight      = nullptr;
    const int8_t* quanWeightInt8 = nullptr;

    int tempWeightSize   = 0;
    std::unique_ptr<Tensor> externalWeightTensor;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    std::vector<int32_t> _bias(outputChannleUp4, 0);
    std::vector<float> _scale(outputChannleUp4, 0);
    std::vector<int32_t> _beta(outputChannleUp4, 0);
    auto biasPtr = _bias.data();
    auto scalePtr = _scale.data();
    auto betaPtr = _beta.data();
    
    if (ModeInt8) {
        ConvolutionCommon::getConvInt8Parameters(conv2d, quanCommon, backend, quanWeightInt8, tempWeightSize, scalePtr, biasPtr, betaPtr);
    } else {
        ConvolutionCommon::getConvParameters(&quanCommon, backend, conv2d, &tempWeight, &tempWeightSize);
    }
    
    bool success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC) &&
                   backend->onAcquireBuffer(cache.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    AutoStorage<uint8_t> lowpWeight;
    if (core->bytes < 4) {
        lowpWeight.reset(outputCount * srcCount * fh * fw * core->bytes);
        if (lowpWeight.get() == nullptr) {
            mValid = false;
            return;
        }
        core->MNNFp32ToLowp(tempWeight, (int16_t*)lowpWeight.get(), outputCount * srcCount * fh * fw);
        tempWeight = (float*)lowpWeight.get();
    }
    if (!ModeInt8) {
        mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
        success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC);
        if (!success) {
            mValid = false;
            return;
        }
        auto dest = mWeight->host<uint8_t>();
        _transformWeight((uint8_t*)tempWeight, dest, outputCount, srcCount, fh, fw, cache->host<uint8_t>(), core);
    } else {
        mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
        success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC);
        if (!success) {
            mValid = false;
            return;
        }
        _reorderWeightInt8(backend, layer, quanWeightInt8, mWeight);
    }
    backend->onReleaseBuffer(cache.get(), Backend::STATIC);
    mOrigin.reset(new CPUDeconvolutionOrigin(input, mWeight.get(), convOp, backend, ModeInt8));
}

CPUDeconvolution::~CPUDeconvolution() {
    // Do nothing
}
ErrorCode CPUDeconvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mDynamicWeight) {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        _transformWeight(inputs[1]->host<uint8_t>(), mWeight->host<uint8_t>(), mParam.outputCount, mParam.srcCount, mParam.fh, mParam.fw, mWeightTransformCache->host<uint8_t>(), core);
        ::memset(mBias->host<uint8_t>(), 0, mBias->length(0) * core->bytes);
        if (inputs.size() >= 3) {
            ::memcpy(mBias->host<uint8_t>(), inputs[2]->host<uint8_t>(), TensorUtils::getRawSize(inputs[2]) * core->bytes);
        }
    }
    return mOrigin->onExecute(mTempInputs, outputs);
}
ErrorCode CPUDeconvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mDynamicWeight) {
        bool res = backend()->onAcquireBuffer(mWeight.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        res = backend()->onAcquireBuffer(mWeightTransformCache.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        res = backend()->onAcquireBuffer(mBias.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
    }
    mTempInputs = {inputs[0], mWeight.get(), mBias.get()};
    auto code = mOrigin->onResize(mTempInputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    if (mDynamicWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mWeightTransformCache.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mBias.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}


ErrorCode CPUDeconvolutionOrigin::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionBasic::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto gcore = static_cast<CPUBackend*>(backend())->int8Functions();
    int bytes = core->bytes;
    auto input  = inputs[0];
    auto output = outputs[0];
    auto oc     = output->channel();
    int UNIT, SRC_UNIT, DST_XUNIT;
    gcore->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    if (UP_DIV(oc, core->pack) * core->pack != inputs[2]->length(0)) {
        return INPUT_DATA_ERROR;
    }

    auto ocC4       = UP_DIV(output->channel(), core->pack);
    auto icC4       = UP_DIV(input->channel(), core->pack);
    auto kw         = mCommon->kernelX();
    auto kh         = mCommon->kernelY();
    auto dilateX    = mCommon->dilateX();
    auto dilateY    = mCommon->dilateY();
    auto strideX    = mCommon->strideX();
    auto strideY    = mCommon->strideY();
    auto padX       = mPadX;
    auto padY       = mPadY;
    auto width      = input->width();
    auto height     = input->height();
    auto src_height = output->height();
    auto src_width  = output->width();
    auto batch      = output->batch();

    auto kernelCount = ocC4 * mCommon->kernelX() * mCommon->kernelY();
    mPostFunctions.clear();
    auto plane         = width * height * batch;
    const int maxDepth = 5;
    auto allocator = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    //int zeroPoint = 0;

    auto biasTensor = inputs[2];
    
    // prepare for float2int8 if necessary.
    auto outputQuant = TensorUtils::getQuantInfo(outputs[0]);
    float scale = outputQuant[0];
    scale = (scale == 0.f ? 0.f : 1.f / scale);
    auto maxValue = outputQuant[3];
    auto minValue = outputQuant[2];
    auto zeroPoint = outputQuant[1];

    AutoRelease<Tensor> tempInput(Tensor::createDevice<float>({icC4, plane, core->pack}));
    bool needReleaseTempInput = true;
    int outi8 = 0;
    if (CPUBackend::getDataType(output) == DataType_DT_INT8 || output->getType().bytes() == 1) {
        outi8 = 1;
    }
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mTempOutput.reset(Tensor::createDevice<float>({batch, height, width, ocC4 * kw * kh * core->pack}));
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mDeconvInt8Exe->onResize({input}, {mTempOutput.get()});
        if (mResource->mRelu) {
            minValue = outputQuant[1];
        }
    }
    else {
        mTempOutput.reset(Tensor::createDevice<float>({kernelCount, plane, core->pack}));
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mMatMul.reset(new StrassenMatrixComputor(backend(), true, maxDepth));
        // tempInput->buffer().host = (uint8_t*)inputPtr;
        
        needReleaseTempInput = false;
        TensorUtils::getDescribeOrigin(tempInput.get())->mem = new CPUMemObj(nullptr, TensorUtils::getDescribeOrigin(input)->mem->chunk(), 0);
        mMatMul->onEncode({tempInput.get(), inputs[1]}, {mTempOutput.get()});
    }
    auto threadNumber = ((CPUBackend*)backend())->threadNumber();
    std::vector<float> scales(core->pack * src_height * src_width * batch, scale);
    auto outputFp32Ptr = allocator->alloc(batch * src_height * src_width * ocC4 * core->pack * bytes);
    if (outputFp32Ptr.invalid()) {
        return OUT_OF_MEMORY;
    }

    mPostFunctions.emplace_back(std::make_pair([ocC4, width, height, kh, kw, padY, padX, dilateY, dilateX, strideY,
                       strideX, threadNumber, src_width, src_height, plane, input, biasTensor, this, core, gcore, batch, outi8, scales,
                       minValue, maxValue, zeroPoint, outputFp32Ptr](uint8_t* outputPtr, int tId) {
        auto colBufferPtr = mTempOutput->host<uint8_t>();
        auto biasPtr      = biasTensor->host<float>();
        auto inputPtr  = input->host<float>();
        auto unitBytes = core->pack * core->bytes;
        auto tempOutPtr = outputPtr;
        auto float2Int8_step = src_height * src_width * batch;
        if (outi8) {
            tempOutPtr = outputFp32Ptr.ptr();
        }
        for (int z = (tId); z < ocC4; z += threadNumber) {
            auto dstZ = tempOutPtr + z * src_height * src_width * batch * unitBytes;
            auto srcZ = colBufferPtr + kw * kh * plane * z * unitBytes;
            ::memset(dstZ, 0, src_width * src_height * batch * unitBytes);
            for (int b = 0; b < batch; ++b) {
                auto dstB = dstZ + b * src_width  * src_height * unitBytes;
                auto srcB = srcZ + b * width * height * unitBytes;
                for (int oy = 0; oy < height; ++oy) {
                    for (int ox = 0; ox < width; ++ox) {
                        int srcStartX = ox * strideX - padX;
                        int srcStartY = oy * strideY - padY;

                        int sfy = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
                        int efy = ALIMIN(kh, UP_DIV(src_height - srcStartY, dilateY));

                        int sfx = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                        int efx = ALIMIN(kw, UP_DIV(src_width - srcStartX, dilateX));

                        auto dstStart = dstB + srcStartX * unitBytes + srcStartY * src_width * unitBytes;
                        auto srcStart = srcB + unitBytes * (ox + oy * width);
                        if (sfy >= efy || sfx >= efx) {
                            continue;
                        }

                        for (int fy = sfy; fy < efy; ++fy) {
                            auto dstY = dstStart + fy * unitBytes * dilateY * src_width;
                            auto srcY = srcStart + fy * kw * plane * unitBytes;
                            core->MNNAddC4WithStride((const float*)(srcY + sfx * plane * unitBytes), (float*)(dstY + sfx * dilateX * unitBytes), plane * core->pack, dilateX * core->pack, efx - sfx);
                        }
                    }
                }
            }
            core->MNNAxByClampBroadcastUnit((float*)dstZ, (float*)dstZ, (const float*)((uint8_t*)biasPtr +  unitBytes * z), src_height * src_width * batch, 0, 0, 1, mPostParameters.data());
            if (outi8) {
                gcore->MNNFloat2Int8((float*)dstZ, (int8_t*)(outputPtr + z * float2Int8_step * core->pack), float2Int8_step, scales.data(), minValue, maxValue, zeroPoint);
            }
        }
    }, threadNumber));
    /*
    if (TensorUtils::getDescribe(tempInput.get())->mem->chunk().offset() != TensorUtils::getDescribe(input)->mem->chunk().offset()) {
        backend()->onReleaseBuffer(tempInput.get(), Backend::DYNAMIC);
    }
     if (tempInput->host<float>() != inputPtr) {
         backend()->onReleaseBuffer(tempInput.get(), Backend::DYNAMIC);
     }
    */
    allocator->free(outputFp32Ptr);
    if (needReleaseTempInput) {
        backend()->onReleaseBuffer(tempInput.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUDeconvolutionOrigin::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputPtr = inputs[0]->host<uint8_t>();
    auto outputPtr = outputs[0]->host<uint8_t>();
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mDeconvInt8Exe->onExecute({inputs[0], inputs[1]}, {mTempOutput.get()});
    }
    else {
        mMatMul->onExecute();
    }
    for (auto& unit : mPostFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, unit.second) {
            unit.first(outputPtr, (int)tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
class CPUDeconvolutionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto convOp = op->main_as_Convolution2D();
        auto common = convOp->common();
        if (backend->type() == MNN_FORWARD_CPU && inputs.size() == 1) {
            if (common->strideY() > 1 || common->strideX() > 1) {
                if (common->dilateX() == 1 && common->dilateY() == 1) {
                    if (common->kernelX() / common->strideX() > 2 || common->kernelY() / common->strideY() > 2) {
                        return new DeconvolutionWithStride(inputs[0], op, backend);
                    }
                }
            }
        }
        return new CPUDeconvolution(inputs[0], op, backend, inputs.size() > 1);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionCreator, OpType_Deconvolution);
} // namespace MNN
