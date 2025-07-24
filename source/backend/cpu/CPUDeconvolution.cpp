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
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {

CPUDeconvolutionBasic::CPUDeconvolutionBasic(int inputChannel, const Op* convOp, Backend* b)
    : CPUConvolution(convOp->main_as_Convolution2D()->common(), b) {
    mSrcCount = inputChannel;
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
    core->MNNPackForMatMul_B((float*)dest, (const float*)cache, outputC4 * fw * fh * core->pack, 1, srcCount, false);
}
std::shared_ptr<DeconvolutionResource> CPUDeconvolution::makeResource(int srcCount, const Op *convOp, Backend* backend, bool dynamic) {
    auto core = static_cast<CPUBackend*>(backend)->functions();
    auto coreInt8 = static_cast<CPUBackend*>(backend)->int8Functions();
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto conv2d = convOp->main_as_Convolution2D();
    auto layer = conv2d->common();
    int outputCount = layer->outputCount();
    const auto outputChannleUp4  = UP_DIV(outputCount, hP) * hP;
    int fw = layer->kernelX();
    int fh = layer->kernelY();
    std::shared_ptr<DeconvolutionResource> res(new DeconvolutionResource);
    res->mParam.fh = fh;
    res->mParam.fw = fw;
    res->mParam.srcCount = srcCount;
    res->mParam.outputCount = outputCount;
    if (dynamic) {
        return res;
    }
    auto outputAlign = UP_DIV(layer->outputCount(), core->pack) * core->pack * fw * fh;
    const float* tempWeight = nullptr;
    int tempWeightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, convOp, &tempWeight, &tempWeightSize);
    AutoStorage<uint8_t> lowpWeight;
    if (core->bytes < 4) {
        lowpWeight.reset(outputCount * srcCount * fh * fw * core->bytes);
        if (lowpWeight.get() == nullptr) {
            return nullptr;
        }
        core->MNNFp32ToLowp(tempWeight, (int16_t*)lowpWeight.get(), outputCount * srcCount * fh * fw);
        tempWeight = (float*)lowpWeight.get();
        quanCommon.reset();
    }
    res->mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
    res->mBias.reset(Tensor::createDevice<float>({UP_DIV(outputCount, core->pack) * core->pack}));
    bool success = backend->onAcquireBuffer(res->mWeight.get(), Backend::STATIC) && backend->onAcquireBuffer(res->mBias.get(), Backend::STATIC);
    AutoStorage<float> cache(outputAlign * srcCount);
    if (!success || cache.get() == nullptr) {
        MNN_ERROR("Alloc memory error for deconvolution\n");
        return nullptr;
    }
    CPUConvolution::Resource::copyBias(res->mBias->host<float>(), convOp->main_as_Convolution2D()->bias()->data(), outputCount, backend);
    _transformWeight((uint8_t*)tempWeight, res->mWeight->host<uint8_t>(), outputCount, srcCount, fh, fw, (uint8_t*)cache.get(), core);
    return res;
}

bool CPUDeconvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (mDynamicWeight) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto exe = new CPUDeconvolution(mSrcCount, op, bn, mDynamicWeight, mResource);
    *dst = exe;
    return true;
}

CPUDeconvolution::CPUDeconvolution(int srcCount, const Op* convOp, Backend* backend, bool dynamicWeight, std::shared_ptr<DeconvolutionResource> resource) : MNN::CPUDeconvolutionBasic(srcCount, convOp, backend) {
    mDynamicWeight = dynamicWeight;
    mResource = resource;
    if (dynamicWeight) {
        auto core = static_cast<CPUBackend*>(backend)->functions();
        auto coreInt8 = static_cast<CPUBackend*>(backend)->int8Functions();
        int eP, lP, hP;
        core->MNNGetMatMulPackMode(&eP, &lP, &hP);
        auto conv2d                  = convOp->main_as_Convolution2D();
        auto layer                   = conv2d->common();
        int outputCount              = layer->outputCount();
        const auto outputChannleUp4  = UP_DIV(outputCount, hP) * hP;
        int fw                  = layer->kernelX();
        int fh                  = layer->kernelY();
        auto outputAlign = UP_DIV(layer->outputCount(), core->pack) * core->pack * fw * fh;
        mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputAlign, hP), UP_DIV(srcCount, lP) * lP, hP}));
        mBias.reset(Tensor::createDevice<float>({UP_DIV(outputCount, core->pack) * core->pack}));
        mOrigin.reset(new CPUDeconvolutionOrigin(srcCount, convOp, backend));
        mWeightTransformCache.reset(Tensor::createDevice<float>({outputAlign * srcCount}));
        return;
    } else {
        mWeight = mResource->mWeight;
        mBias = mResource->mBias;
    }
    mOrigin.reset(new CPUDeconvolutionOrigin(srcCount, convOp, backend));
}

CPUDeconvolution::~CPUDeconvolution() {
    // Do nothing
}
ErrorCode CPUDeconvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mDynamicWeight) {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        _transformWeight(inputs[1]->host<uint8_t>(), mWeight->host<uint8_t>(), mResource->mParam.outputCount, mResource->mParam.srcCount, mResource->mParam.fh, mResource->mParam.fw, mWeightTransformCache->host<uint8_t>(), core);
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

CPUDeconvolutionOrigin::CPUDeconvolutionOrigin(int inputChannel, const Op *convOp, Backend *b) : CPUDeconvolutionBasic(inputChannel, convOp, b) {
    // Do nothing
}

ErrorCode CPUDeconvolutionOrigin::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionBasic::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes = core->bytes;
    auto input  = inputs[0];
    auto output = outputs[0];
    auto oc     = output->channel();
    if (UP_DIV(oc, core->pack) * core->pack != inputs[2]->length(0)) {
        return INPUT_DATA_ERROR;
    }
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);

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
    auto weightTensor = inputs[1];
    auto biasTensor = inputs[2];

    auto kernelCount = ocC4 * mCommon->kernelX() * mCommon->kernelY();
    auto plane = width * height * batch;
    auto allocator = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    auto tileCount = UP_DIV(plane, eP);
    threadNumber = ALIMIN(tileCount, threadNumber);
    auto memMode = static_cast<CPUBackend*>(backend())->memoryMode();
    if (memMode != BackendConfig::Memory_High) {
        // Limit threadNumber to avoid too large memory
        threadNumber = ALIMIN(threadNumber, 4);
    }
    auto im2colOutputStride = input->channel() * eP * core->bytes;
    mGemmInput = allocator->alloc(threadNumber * im2colOutputStride);
    auto gemmOutputStride = kernelCount * core->pack * eP * core->bytes;
    mGemmOutput = allocator->alloc(threadNumber * gemmOutputStride);
    auto outputSize = batch*src_width*src_height*ocC4*core->pack*core->bytes;
    if (threadNumber > 1) {
        mExtraOutput = allocator->alloc((threadNumber-1)*outputSize);
    }
    allocator->free(mGemmInput);
    allocator->free(mGemmOutput);
    if (threadNumber > 1) {
        allocator->free(mExtraOutput);
    }
    auto first = std::make_pair([=](uint8_t* outputPtr, int tId) {
        auto gemmInputBufferPtr = mGemmInput.ptr() + tId * im2colOutputStride;
        auto colBufferPtr = mGemmOutput.ptr() + tId * gemmOutputStride;
        auto inputPtr  = input->host<uint8_t>();
        auto unitBytes = core->pack * core->bytes;
        auto tempOutPtr = outputPtr;
        if (tId > 0) {
            tempOutPtr = mExtraOutput.ptr() + (tId-1) * outputSize;
        }
        ::memset(tempOutPtr, 0, outputSize);

        int l = ROUND_UP(mSrcCount, lP);
        int h = kernelCount * core->pack;
        auto weightPtr = weightTensor->host<uint8_t>();
        for (int index=tId; index < tileCount; index+=threadNumber) {
            int xStart = index * eP;
            int xEnd = ALIMIN(xStart + eP, plane);
            int xCount = xEnd-xStart;
            if (xCount <= 0) {
                continue;
            }
            size_t parameters[7];
            parameters[0] = xCount * core->bytes;
            parameters[1] = l;
            parameters[2] = h;
            parameters[3] = xCount * core->bytes * core->pack;
            parameters[4] = 0;
            parameters[5] = 0;
            parameters[6] = 0;
            const float* postParametersPtr = nullptr;
            int32_t info[4];
            int32_t stride[4];
            stride[0] = xCount;
            stride[1] = (int32_t)parameters[1];
            stride[2] = 0;
            stride[3] = 0;
            info[0] = 1;
            info[1] = plane;
            info[2] = xCount;
            info[3] = 1;
            auto aStart = inputPtr + xStart * unitBytes;
            core->MNNPackC4ForMatMul_A((float*)(gemmInputBufferPtr), (const float**)(&aStart), info, stride);
            if (xCount == eP) {
                core->MNNPackedMatMul((float*)(colBufferPtr), (float*)gemmInputBufferPtr, (float*)weightPtr, parameters, postParametersPtr, nullptr, nullptr, nullptr);
            } else {
                core->MNNPackedMatMulRemain((float*)(colBufferPtr), (float*)gemmInputBufferPtr, (float*)weightPtr, xCount, parameters, postParametersPtr, nullptr, nullptr, nullptr);
            }
            // Col2Im
            for (int z = 0; z < ocC4; ++z) {
                auto dstZ = tempOutPtr + z * src_height * src_width * batch * unitBytes;
                auto srcZ = colBufferPtr + kw * kh * xCount * z * unitBytes;
                for (int x=0; x<xCount; ++x) {
                    auto index = xStart + x;
                    int b = index / (width * height);
                    index = index % (width * height);
                    int oy = index / width;
                    int ox = index % width;
                    int srcStartX = ox * strideX - padX;
                    int srcStartY = oy * strideY - padY;
                    
                    int sfy = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
                    int efy = ALIMIN(kh, UP_DIV(src_height - srcStartY, dilateY));
                    
                    int sfx = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                    int efx = ALIMIN(kw, UP_DIV(src_width - srcStartX, dilateX));
                    
                    auto dstStart = dstZ + b * src_width * src_height * unitBytes + srcStartX * unitBytes + srcStartY * src_width * unitBytes;
                    auto srcStart = srcZ + x * unitBytes;
                    if (sfy >= efy || sfx >= efx) {
                        continue;
                    }
                    
                    for (int fy = sfy; fy < efy; ++fy) {
                        auto dstY = dstStart + fy * unitBytes * dilateY * src_width;
                        auto srcY = srcStart + fy * kw * xCount * unitBytes;
                        core->MNNAddC4WithStride((const float*)(srcY + sfx * xCount * unitBytes), (float*)(dstY + sfx * dilateX * unitBytes), xCount * core->pack, dilateX * core->pack, efx - sfx);
                    }
                }
            }
        }
    }, threadNumber);
    auto second = std::make_pair([ocC4, src_height, src_width, threadNumber, batch, biasTensor, this, outputSize, core](uint8_t* outputPtr, int tId) {
        auto unitBytes = core->pack * core->bytes;
        auto biasPtr = biasTensor->host<uint8_t>();
        for (int z = tId; z < ocC4; z+=threadNumber) {
            auto dstZ = outputPtr + z * src_height * src_width * batch * unitBytes;
            if (threadNumber > 1) {
                for (int index=0; index<threadNumber-1; ++index) {
                    auto src = mExtraOutput.ptr() + index * outputSize + z * src_height * src_width * batch * unitBytes;
                    core->MNNMatrixAdd((float*)(dstZ), (float*)(src), (float*)(dstZ), src_height * src_width * batch, 0, 0, 0, 1);
                }
            }
            core->MNNAxByClampBroadcastUnit((float*)dstZ, (float*)dstZ, (const float*)((uint8_t*)biasPtr +  unitBytes * z), src_height * src_width * batch, 0, 0, 1, mPostParameters.data());
        }

    }, threadNumber);
    mExecuteFuntion = {first, second};
    return NO_ERROR;
}

ErrorCode CPUDeconvolutionOrigin::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputPtr = inputs[0]->host<uint8_t>();
    auto outputPtr = outputs[0]->host<uint8_t>();
    for (auto& unit : mExecuteFuntion) {
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
        auto res = CPUDeconvolution::makeResource(inputs[0]->channel(), op, backend, inputs.size() > 1);
        if (nullptr == res) {
            MNN_ERROR("CPUDeconvolution makeResource error\n");
            return nullptr;
        }
        return new CPUDeconvolution(inputs[0]->channel(), op, backend, inputs.size() > 1, res);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionCreator, OpType_Deconvolution);
} // namespace MNN
