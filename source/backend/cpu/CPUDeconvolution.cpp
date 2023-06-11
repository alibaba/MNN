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

CPUDeconvolutionCommon::CPUDeconvolutionCommon(const Tensor* input, const Op* convOp, Backend* b)
    : CPUDeconvolutionBasic(input, convOp, b) {
    auto conv2D     = convOp->main_as_Convolution2D();
    int outputCount = mCommon->outputCount();
    auto core = static_cast<CPUBackend*>(b)->functions();
    mBias.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, core->pack) * core->pack}));
    bool success = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->length(0) * core->bytes);
    if (USE_EXTERNAL_DATA(conv2D)) {
        auto external = conv2D->external();
        auto offset = external->Get(0) + external->Get(1);
        auto bytes = external->Get(2);
        if (core->bytes == 4) {
            OpCommonUtils::loadExternalData(backend(), mBias->host<char>(), offset, bytes);
        } else {
            int biasSize = static_cast<int>(bytes / sizeof(float));
            std::unique_ptr<Tensor> externalBiasTensor(Tensor::createDevice<float>({biasSize}));
            auto status = backend()->onAcquireBuffer(externalBiasTensor.get(), Backend::STATIC);
            if (!status) {
                MNN_ERROR("Out of memory when externalBiasTensor is acquired in CPUDeconvolutionCommon.\n");
                return;
            }
            OpCommonUtils::loadExternalData(backend(), externalBiasTensor->host<char>(), offset, bytes);
            core->MNNFp32ToLowp(externalBiasTensor->host<float>(), mBias->host<int16_t>(), biasSize);
        }
    } else {
        if (core->bytes == 4) {
            ::memcpy(mBias->host<float>(), conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));
        } else {
            core->MNNFp32ToLowp(conv2D->bias()->data(), mBias->host<int16_t>(), conv2D->bias()->size());
        }
    }
}

CPUDeconvolutionCommon::~CPUDeconvolutionCommon() {
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
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
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    std::vector<int> shape = {UP_DIV(oc, UNIT) * kernelCount, UP_DIV(UP_DIV(ic, UNIT), SRC_UNIT / UNIT), UNIT, SRC_UNIT};

    weight.reset(Tensor::createDevice<int8_t>(shape));
    bool succ = bn->onAcquireBuffer(weight.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return;
    }
    auto dstPtr = weight->host<int8_t>();

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
CPUDeconvolution::CPUDeconvolution(const Tensor* input, const Op* convOp, Backend* backend)
    : MNN::CPUDeconvolutionCommon(input, convOp, backend) {
    auto core               = static_cast<CPUBackend*>(backend)->functions();
    auto coreInt8           = static_cast<CPUBackend*>(backend)->int8Functions();
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    int UNIT, SRC_UNIT, DST_XUNIT;
    coreInt8->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    if (CPUBackend::getDataType(input) == DataType_DT_INT8 || input->getType().bytes() == 1) {
        eP = DST_XUNIT;
        lP = SRC_UNIT;
        hP = UNIT;
    }
    auto conv2d                  = convOp->main_as_Convolution2D();
    auto layer                   = conv2d->common();
    int outputCount              = layer->outputCount();
    const auto outputChannleUp4  = UP_DIV(outputCount, hP) * hP;
    const float* tempWeight      = nullptr;
    const int8_t* quanWeightInt8 = nullptr;

    bool ModeInt8        =  false;
    int tempWeightSize   = 0;
    std::unique_ptr<Tensor> externalWeightTensor;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    std::vector<int32_t> _bias(outputChannleUp4, 0);
    std::vector<float> _scale(outputChannleUp4, 0);
    auto biasPtr = _bias.data();
    auto scalePtr = _scale.data();
    
    if (USE_EXTERNAL_DATA(conv2d)) {
        auto bytes = conv2d->external()->Get(1);
        tempWeightSize = static_cast<int>(bytes / sizeof(float));
        externalWeightTensor.reset(Tensor::createDevice<float>({tempWeightSize}));
        auto status = backend->onAcquireBuffer(externalWeightTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when externalWeightTensor is acquired in CPUDeconvolution.\n");
            return;
        }
        OpCommonUtils::loadExternalData(backend, externalWeightTensor->host<char>(), conv2d->external()->Get(0), bytes);
        tempWeight = externalWeightTensor->host<float>();
    } else {
        if (CPUBackend::getDataType(input) == DataType_DT_INT8 || input->getType().bytes() == 1) {
            ConvolutionCommon::getConvInt8Parameters(conv2d, quanCommon, quanWeightInt8, tempWeightSize, scalePtr, biasPtr);
            ModeInt8 = true;
        } else {
            ConvolutionCommon::getConvParameters(&quanCommon, conv2d, &tempWeight, &tempWeightSize);
        }
    }

    int fw                  = layer->kernelX();
    int fh                  = layer->kernelY();
    int srcCount            = mSrcCount;
    
    auto outputAlign = UP_DIV(layer->outputCount(), core->pack) * core->pack * fw * fh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
    std::shared_ptr<Tensor> cache(Tensor::createDevice<float>({outputAlign * srcCount}));
    bool success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC) &&
                   backend->onAcquireBuffer(cache.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    auto dest = mWeight->host<uint8_t>();
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
        _transformWeight((uint8_t*)tempWeight, dest, outputCount, srcCount, fh, fw, cache->host<uint8_t>(), core);
    } else {
        _reorderWeightInt8(backend, layer, quanWeightInt8, mWeight);
    }
    backend->onReleaseBuffer(cache.get(), Backend::STATIC);
    mOrigin.reset(new CPUDeconvolutionOrigin(input, mWeight.get(), convOp, backend, ModeInt8));
}

CPUDeconvolution::~CPUDeconvolution() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}


ErrorCode CPUDeconvolutionOrigin::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionBasic::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto gcore = static_cast<CPUBackend*>(backend())->int8Functions();
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
    //int zeroPoint = 0;

    auto biasPtr      = inputs[2]->host<float>();
    auto inputPtr  = input->host<float>();
    
    // prepare for float2int8 if necessary.
    auto outputQuant = TensorUtils::getQuantInfo(outputs[0]);
    float scale = outputQuant[0];
    scale = (scale == 0.f ? 0.f : 1.f / scale);
    auto maxValue = outputQuant[3];
    auto minValue = outputQuant[2];
    auto zeroPoint = outputQuant[1];

    AutoRelease<Tensor> tempInput(Tensor::createDevice<float>({icC4, plane, core->pack}));
    int outi8 = 0;
    if (CPUBackend::getDataType(output) == DataType_DT_INT8 || output->getType().bytes() == 1) {
        outi8 = 1;
    }
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mTempOutput.reset(Tensor::createDevice<uint8_t>({batch, ocC4 * kw * kh * core->pack, height, width, core->bytes}, Tensor::CAFFE_C4));
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
        tempInput->buffer().host = (uint8_t*)inputPtr;
        mMatMul->onEncode({tempInput.get(), inputs[1]}, {mTempOutput.get()});
    }
    auto colBufferPtr = mTempOutput->host<uint8_t>();
    auto threadNumber = ((CPUBackend*)backend())->threadNumber();
    std::vector<float> scales(core->pack * src_height * src_width * batch, scale);
    
    std::shared_ptr<Tensor> OutputFloat(Tensor::createDevice<float>(output->shape()));
    auto res = backend()->onAcquireBuffer(OutputFloat.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto outputFp32Ptr = OutputFloat->host<uint8_t>();

    mPostFunctions.emplace_back(std::make_pair([colBufferPtr, ocC4, width, height, kh, kw, padY, padX, dilateY, dilateX, strideY,
                       strideX, threadNumber, src_width, src_height, plane, biasPtr, this, core, gcore, batch, outi8, scales,
                       minValue, maxValue, zeroPoint, outputFp32Ptr](uint8_t* outputPtr, int tId) {
        auto unitBytes = core->pack * core->bytes;
        auto tempOutPtr = outputPtr;
        auto float2Int8_step = src_height * src_width * batch;
        if (outi8) {
            tempOutPtr = outputFp32Ptr;
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
    if (tempInput->host<float>() != inputPtr) {
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
        if (backend->type() == MNN_FORWARD_CPU) {
            if (common->strideY() > 1 || common->strideX() > 1) {
                if (common->dilateX() == 1 && common->dilateY() == 1) {
                    if (common->kernelX() / common->strideX() > 2 || common->kernelY() / common->strideY() > 2) {
                        return new DeconvolutionWithStride(inputs[0], op, backend);
                    }
                }
            }
        }
        return new CPUDeconvolution(inputs[0], op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionCreator, OpType_Deconvolution);
} // namespace MNN
