//
//  ConvInt8_1xN.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN/AutoTime.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "ConvInt8_1xN.hpp"
#include "Int8FunctionsOpt.h"
#include "WinogradHelper.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include <string>

static const int BLOCK_UNIT = MNN::WinogradHelper::L2K3::blockUnit();
static const int SRC_UNIT = MNN::WinogradHelper::L2K3::srcUnit();
static const int DST_UNIT = MNN::WinogradHelper::L2K3::dstUnit();
static const int GEMM_TILE_UNIT = DST_XUNIT;

using namespace MNN::WinogradHelper::L2K3;

// dim: 4-element, sizeDW, sizeDH, strideSW, strideDH
// C4 will get best performance in future, because it will be apply to C4 data eventually
static void MNNTranspose8Bit(int8_t* dstO, const int8_t* srcO, int* dim, int unit) {
    const int w = dim[0], h = dim[1], srcStride = dim[2], dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i * 4;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j * 4;
            for (int z=0; z<unit; ++z) {
                dj[z] = sj[z];
            }
            
        }
    }
}

namespace MNN {

ConvInt8_1xN::ConvInt8_1xN(Backend *backend, const MNN::Convolution2D *convParam, float inputScale, float outputScale) : CPUConvolution(convParam->common(), backend) {
    const auto convCommon      = convParam->common();
    const auto kx = convCommon->kernelX(), ky = convCommon->kernelY();
    const auto outputCount = convCommon->outputCount(), srcCount = convCommon->inputCount();
    
    if (kx == 1 && ky != 1) {
        mTranspose = true;
    }
    mKernelSize = ALIMAX(kx, ky);
    mActBits = convParam->symmetricQuan()->nbits();
    
    const int unitI = 8;
    const auto outputCountUnit = UP_DIV(outputCount, 4), srcCountUnit = UP_DIV(srcCount, unitI);
    std::shared_ptr<Tensor> weightInt8(Tensor::createDevice<int8_t>({outputCountUnit, srcCountUnit, mKernelSize, unitI * 4}));
    mWeight.reset(Tensor::createDevice<int8_t>({UP_DIV(mKernelSize, 3), SRC_UNIT, outputCountUnit, srcCountUnit, unitI * 4}));
    bool res = backend->onAcquireBuffer(weightInt8.get(), Backend::STATIC);
    res = res && backend->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!res) {
        mValid = false;
        return;
    }
    const int outputChannleUp4 = ALIGN_UP4(outputCount);
    mBiasFloat.reset(Tensor::createDevice<float>({outputChannleUp4}));
    res = backend->onAcquireBuffer(mBiasFloat.get(), Backend::STATIC);
    if (!res) {
        mValid = false;
        return;
    }
    mScaleFloat.reset(Tensor::createDevice<float>({outputChannleUp4}));
    res = backend->onAcquireBuffer(mScaleFloat.get(), Backend::STATIC);
    if (!res) {
        mValid = false;
        return;
    }
    auto biasPtr = mBiasFloat->host<int32_t>();
    memset(biasPtr, 0, outputChannleUp4 * sizeof(int32_t));
    auto scalePtr = mScaleFloat->host<float>();
    memset(scalePtr, 0, outputChannleUp4 * sizeof(float));
    const int8_t *weightSrc = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (!ConvolutionCommon::getConvInt8Parameters(convParam, quanCommon, weightSrc, scalePtr, biasPtr, inputScale, outputScale)) {
        return;
    }

    auto weightDst = weightInt8->host<int8_t>();
    memset(weightDst, 0, weightInt8->size());
    CPUConvolution::reorderWeightSlow<int8_t>(weightDst, weightSrc, srcCount, outputCount, mKernelSize, unitI, 4, true);
    const auto weightOffset = mWeight->stride(1);
    for (int z = 0; z < srcCountUnit * outputCountUnit; ++z) {
        auto src = weightInt8->host<int8_t>() + mKernelSize * z * unitI * 4;
        auto dst = mWeight->host<int8_t>() + z * unitI * 4;
        
        for (int i = 0; i < mKernelSize / 3; ++i) {
            weightTransform1D<int8_t, 16>(src + i * 3 * unitI * 4, dst + i * mWeight->stride(0),
                                          unitI * 4, weightOffset, (unitI * 4) / 16);
        }
        if (mKernelSize % 3 != 0) {
            src = src + (mKernelSize / 3) * 3 * unitI * 4;
            dst = dst + (mKernelSize / 3) * mWeight->stride(0);
            for (int i = 0; i < mKernelSize % 3; ++i) {
                ::memcpy(dst + i * weightOffset, src + i * unitI * 4, unitI * 4 * sizeof(int8_t));
            }
        }
    }
    
    backend->onReleaseBuffer(weightInt8.get(), Backend::STATIC);

    mRelu    = convCommon->relu() || convCommon->relu6();
}

ConvInt8_1xN::~ConvInt8_1xN() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mBiasFloat.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mScaleFloat.get(), Backend::STATIC);
}

ErrorCode ConvInt8_1xN::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input = inputs[0];
    const int srcCount = input->channel(), outputCountUnit = UP_DIV(outputs[0]->channel(), 4);
    auto threadNums = ((CPUBackend*)backend())->threadNumber();
    int unitI = 8, srcCountUnit = UP_DIV(srcCount, unitI);
    
    int ih = input->height(), iw = input->width(), batch = input->batch();
    if (mTranspose) {
        int oh = outputs[0]->height(), ow = outputs[0]->width();
        if ((ih != 1 && iw != 1) || (oh != 1 && ow != 1)) {
            mTransBuffer.reset(Tensor::createDevice<int8_t>({ALIMAX(input->stride(0), outputs[0]->stride(0))}));
        }
        std::swap(ih, iw);
    }
    mTempInput.reset(Tensor::createDevice<int8_t>({batch, srcCountUnit, ih, iw, unitI}));
    mTempSrcBuffer.reset(Tensor::createDevice<int8_t>({threadNums, BLOCK_UNIT, srcCountUnit, GEMM_TILE_UNIT * unitI}));
    mTempDstBuffer.reset(Tensor::createDevice<float>({threadNums, BLOCK_UNIT, outputCountUnit, GEMM_TILE_UNIT * 4}));
    mTempOutBuffer.reset(Tensor::createDevice<float>({threadNums, 2, outputCountUnit, DST_UNIT * GEMM_TILE_UNIT * 4}));
    // threadNum * [2 * SRC_UNIT * 4 or GEMM_TILE_UNIT * DST_UNIT * 4]
    mTempTransformBuffer.reset(Tensor::createDevice<float>({threadNums, 2, ALIMAX(SRC_UNIT * 4, GEMM_TILE_UNIT * DST_UNIT * 2)}));
    
    std::vector<Tensor*> dynamicAllocTensors = {
        mTempSrcBuffer.get(), mTempDstBuffer.get(), mTempOutBuffer.get(),
        mTempTransformBuffer.get(), mTempInput.get()
    };
    if (mTransBuffer.get() != nullptr) {
        dynamicAllocTensors.push_back(mTransBuffer.get());
    }
    bool res = true;
    for (int i = 0; i < dynamicAllocTensors.size(); ++i) {
        res = res && backend()->onAcquireBuffer(dynamicAllocTensors[i], Backend::DYNAMIC);
    }
    if (!res) {
        return OUT_OF_MEMORY;
    }
    for (int i = 0; i < dynamicAllocTensors.size(); ++i) {
        backend()->onReleaseBuffer(dynamicAllocTensors[i], Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode ConvInt8_1xN::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    int ow = output->width(), oh = output->height(), iw = input->width(), ih = input->height(), padY = mPadY, padX = mPadX;
    const int kernelSize = mKernelSize, dc_4 = UP_DIV(output->channel(), 4);
    const int unitI = 8, ic_unit = UP_DIV(input->channel(), unitI), weightOffset = ic_unit * dc_4 * unitI * 4;
    
    for (int b = 0; b < input->batch(); ++b) {
        auto src = input->host<int8_t>() + b * input->stride(0);
        auto dst = mTempInput->host<int8_t>() + b * mTempInput->stride(0);
        const int threadNumber = ((CPUBackend*)backend())->threadNumber(), ic4 = UP_DIV(input->channel(), 4);
        // Clip into [minVal, maxVal]
        const int minVal = -(1<<(mActBits-1)), maxVal = (1<<(mActBits-1))-1, size = input->elementSize();
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int step = UP_DIV(size, 8 * threadNumber), start = (int)tId * step, num = ALIMIN(start + step, size) - start;
            if (num > 0) {
                MNNInt8ClipInplace(dst + start, num, minVal, maxVal);
            }
        } MNN_CONCURRENCY_END();

        // C4 to C8, may do real transpose
        int8_t* transOrigin = nullptr;
        if (mTranspose && iw != 1 && ih != 1) {
            transOrigin = mTransBuffer->host<int8_t>();
        }
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int step = UP_DIV(ic_unit, threadNumber) * 2; // C4 step must be even so we can do C4 -> C8
            int start = (int)tId * step, num = ALIMIN(start + step, ic4) - start; // based on C4
            if (num > 0) {
                int8_t* _src = (transOrigin != nullptr ? transOrigin : src);
                if (transOrigin != nullptr) {
                    int dims[] = {ih, iw, iw * 4, ih * 4};
                    for (int z = start; z < start + num; ++z) {
                        MNNTranspose8Bit(_src + z * ih * iw * 4, src + z * ih * iw * 4, &dims[0], 4); // C4
                    }
                }
                MNNInt8C4ToC8(dst + start * iw * ih * 4, _src + start * iw * ih * 4, iw * ih, num);
            }
        } MNN_CONCURRENCY_END();
    }
    if (mTranspose) {
        std::swap(ih, iw); std::swap(oh, ow); std::swap(padX, padY);
    }
    const int wUnit = UP_DIV(ow, DST_UNIT), hUnit = oh, totalCount = hUnit * wUnit;

    int tileCount     = UP_DIV(totalCount, GEMM_TILE_UNIT);
    const int threadNumber = std::min(((CPUBackend*)backend())->threadNumber(), tileCount);
    
    auto sourceTransformFunc = [=](int xIndex, int xC, int offset, const int8_t* srcOrigin,
                                   int8_t* srcBlock, int8_t* dstOrigin) {
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto dstUnit = dstOrigin + unitI * xi;

            int wIndex = index % wUnit;
            int hIndex = index / wUnit;

            int srcX = wIndex * DST_UNIT - padX + offset;
            int srcY = hIndex - padY;
            int sy   = ALIMAX(0, srcY) - srcY;
            int ey   = ALIMIN(srcY + 1, ih) - srcY;
            int sx   = ALIMAX(0, srcX) - srcX;
            int ex   = ALIMIN(srcX + SRC_UNIT, iw) - srcX;
            auto xL = ex - sx;

            auto srcStart = srcOrigin + (srcX + srcY * iw) * unitI;
            for (int z = 0; z < ic_unit; ++z) {
                ::memset(srcBlock, 0, unitI * SRC_UNIT * sizeof(int8_t));
                auto _dstStart = dstUnit + z * unitI * xC;
                auto src_z = srcStart + z * unitI * iw * ih;
                // Extract One Block
                if (xL > 0) {
                    for (int yy = sy; yy < ey; ++yy) {
                        auto dst_yy = srcBlock + yy * unitI * SRC_UNIT;
                        auto src_yy = src_z + unitI * iw * yy;
                        ::memcpy(dst_yy + sx * unitI, src_yy + sx * unitI , xL * unitI * sizeof(int8_t));
                    }
                }
                // Source Transform
                sourceTransformUnit1D<int8_t, 8>(srcBlock, _dstStart, unitI, unitI * xC * ic_unit, 1);
            }
        }
    }; // [thread, channelC4, GEMM_UNIT * DST_UNIT * 4]
        
    auto destTransformFunc = [=](int xC, const float* srcOrigin, float* dstOrigin) {
        // Dest Transform
        for (int xi = 0; xi < xC; ++xi) {
            auto srcUnit = srcOrigin + 4 * xi;
            auto dstStart = dstOrigin + 4 * xi * DST_UNIT;
            for (int z = 0; z < dc_4; ++z) {
                auto srcZ = srcUnit + z * xC * 4;
                auto dstZ = dstStart + z * xC * DST_UNIT * 4;
                destTransform1D<WinogradHelper::FractionsInA>(srcZ, dstZ, dc_4 * 4 * xC, 4, 1);
            }
        }
    };
        
    auto gemmFunc = [=](int xC, int start, int end, int dstStride, const int8_t* srcOrigin, const int8_t* weight, float* dstOrigin) {
        if (xC == GEMM_TILE_UNIT) {
            for (int i = start; i < end; ++i) {
                MNNGemmInt8toFloat32_8x4_Unit(dstOrigin + i * dc_4 * 4 * xC, srcOrigin + i * ic_unit * unitI * xC,
                                              weight + i * weightOffset, ic_unit, dstStride, dc_4);
            }
        } else {
            for (int i = start; i < end; ++i) {
                MNNGemmInt8toFloat32_8x4_Common(dstOrigin + i * dc_4 * 4 * xC, srcOrigin + i * ic_unit * unitI * xC,
                                                weight + i * weightOffset, ic_unit, xC, dstStride, dc_4);
            }
        }
    };
    auto gemmConcurrencyFunc = [=, &gemmFunc](int xC, int dstStride, const int8_t* srcOrigin, const int8_t* weight, float* dstOrigin) {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            const int step = UP_DIV(SRC_UNIT, threadNumber);
            gemmFunc(xC, tId * step, ALIMIN((tId + 1) * step, SRC_UNIT), dstStride, srcOrigin, weight, dstOrigin);
        }
        MNN_CONCURRENCY_END()
    };
        
    auto sourceFetchFunc = [=](const int8_t* srcOrigin, int8_t* dstOriginInt8,
                               int& index, int& offset, int kOffset, int fetchNum) {
        int remain = fetchNum;
        while (remain > 0) {
            bool fillZero = true;
            int wIndex = index % wUnit, hIndex = index / wUnit, num;
            int srcX = wIndex * DST_UNIT - padX + kOffset + offset, srcY = hIndex - padY;
            if (srcY < 0 || srcY >= ih) {
                num = (srcX < iw ? iw - srcX : DST_UNIT - offset);
            } else if (srcX < 0 || srcX >= iw) {
                num = (srcX < 0 ? -srcX : DST_UNIT - offset);
            } else {
                num = iw - srcX;
                fillZero = false;
            }
            num = ALIMIN(num, remain);
            if (fillZero) {
                for (int i = 0; i < ic_unit; ++i) {
                    ::memset(dstOriginInt8 + i * fetchNum * unitI, 0, num * sizeof(int8_t) * unitI);
                }
            } else {
                auto _srcOrigin = srcOrigin + (iw * srcY + srcX) * unitI;
                for (int i = 0; i < ic_unit; ++i) {
                    ::memcpy(dstOriginInt8 + i * fetchNum * unitI, _srcOrigin + i * iw * ih * unitI, num * sizeof(int8_t) * unitI);
                }
            }
            remain -= num;
            dstOriginInt8 += num * unitI;
            if (num >= DST_UNIT - offset) {
                num = num - (DST_UNIT - offset);
                index += 1;
                offset = 0;
            }
            index += num / DST_UNIT;
            offset += num % DST_UNIT;
        }
    };
    
    auto outAddBiasQuantizeStore = [=](int xIndex, int xC, const float* bias, const float* scale, const float* srcOrigin,
                                       int8_t* dstOrigin, float* tempBuffer, size_t numBit, bool uint) {
        const int indexEnd = xIndex + xC;
        int index = xIndex;
        
        ssize_t minValue, maxValue;
        if (uint) {
            minValue = 0;
            maxValue = (1 << numBit) - 1;
        } else {
            minValue = -(1 << (numBit - 1));
            maxValue = (1 << (numBit - 1)) - 1;
        }
        
        while(index < indexEnd) {
            int wIndex = index % wUnit, hIndex = index / wUnit;
            int dstX = wIndex * DST_UNIT, dstY = hIndex;
            int num = ALIMIN(ow - dstX, (indexEnd - index) * DST_UNIT);
            
            auto _srcOrigin = srcOrigin + (index - xIndex) * DST_UNIT * 4;
            auto _dstOrigin = dstOrigin + (dstY * ow + dstX) * 4;
            for (int i = 0; i < dc_4; ++i) {
                auto src = _srcOrigin + i * xC * DST_UNIT * 4;
                auto dst = _dstOrigin + i * ow * oh * 4;
                
#ifdef MNN_USE_NEON
                auto biasV = vld1q_f32(bias + i * 4);
                for (int j = 0; j < num; ++j) {
                    vst1q_f32(tempBuffer + j * 4, vld1q_f32(src + j * 4) + biasV);
                }
#else
                for (int j = 0; j < num; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        tempBuffer[k + j * 4] = src[k + j * 4] + bias[k + i * 4];
                    }
                }
#endif
                MNNFloat2Int8(tempBuffer, dst, num, scale + i * 4, minValue, maxValue, 0);
            }
            index += UP_DIV(num, DST_UNIT);
        }
    };
    
    auto tFunction = [&](const int tId, const int tileStart, const int tileStep, const int tileEnd, const int8_t* srcOrigin, int8_t* dstOrigin) {
        auto srcBlock = (int8_t*)(mTempTransformBuffer->host<int32_t>() + mTempTransformBuffer->stride(0) * tId);
        auto _srcOrigin = mTempSrcBuffer->host<int8_t>() + mTempSrcBuffer->stride(0) * tId;
        auto _dstOriginTemp = mTempDstBuffer->host<float>() + mTempDstBuffer->stride(0) * tId;
        float* _dstOrigin[2] = {mTempOutBuffer->host<float>() + mTempOutBuffer->stride(0) * tId, nullptr};
        _dstOrigin[1] = _dstOrigin[0] + mTempOutBuffer->stride(1);
        
        for (int tIndex = tileStart; tIndex < tileEnd; tIndex += tileStep) {
            int xIndex  = (int)tIndex * GEMM_TILE_UNIT;
            int xReamin = totalCount - xIndex;
            int xC      = xReamin > GEMM_TILE_UNIT ? GEMM_TILE_UNIT : xReamin;
        
            // Source Transform
            const int stride = DST_UNIT * xC * 4;
            for (int kIndex = 0; kIndex * 3 < kernelSize - 2; ++kIndex) {
                auto weight = mWeight->host<int8_t>() + kIndex * mWeight->stride(0);
                sourceTransformFunc(xIndex, xC, kIndex * 3, srcOrigin, srcBlock, _srcOrigin);
                if (threadNumber != tileStep) {
                    gemmConcurrencyFunc(xC, xC * 4, _srcOrigin, weight, _dstOriginTemp);
                } else {
                    gemmFunc(xC, 0, SRC_UNIT, xC * 4, _srcOrigin, weight, _dstOriginTemp);
                }
                if (kIndex == 0) {
                    destTransformFunc(xC, _dstOriginTemp, _dstOrigin[0]);
                } else {
                    destTransformFunc(xC, _dstOriginTemp, _dstOrigin[1]);
                    MNNMatrixAdd(_dstOrigin[0], _dstOrigin[0], _dstOrigin[1], xC * DST_UNIT, stride, stride, stride, dc_4);
                }
            }
            
            if (kernelSize % 3 != 0) {
                auto weightLeftOver = mWeight->host<int8_t>() + (mWeight->length(0) - 1) * mWeight->stride(0);
                for (int i = 0; i < kernelSize % 3; ++i) {
                    auto weight = weightLeftOver + i * mWeight->stride(1);
                    int index = xIndex, offset = 0, gemmCount = xC * DST_UNIT, gemmDone = 0;
                    do {
                        int fetchNum = ALIMIN(gemmCount - gemmDone, GEMM_TILE_UNIT);
                        sourceFetchFunc(srcOrigin, _srcOrigin, index, offset, kernelSize / 3 * 3 + i, fetchNum);
                        gemmFunc(fetchNum, 0, 1, gemmCount * 4, _srcOrigin, weight, _dstOrigin[1] + gemmDone * 4);
                        gemmDone += fetchNum;
                    } while(gemmDone < gemmCount);
                    MNNMatrixAdd(_dstOrigin[0], _dstOrigin[0], _dstOrigin[1], xC * DST_UNIT, stride, stride, stride, dc_4);
                }
            }
            outAddBiasQuantizeStore(xIndex, xC, mBiasFloat->host<float>(), mScaleFloat->host<float>(),
                                    _dstOrigin[0], dstOrigin, (float*)srcBlock, 8, false);
        }
    };
    int batchSize = input->batch();
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        auto srcOrigin = mTempInput->host<int8_t>() + batchIndex * mTempInput->stride(0);
        auto dstOrigin = output->host<int8_t>() + batchIndex * output->stride(0), _dstOrigin = dstOrigin;
        if (mTranspose && ow != 1 && oh != 1) {
            _dstOrigin = mTransBuffer->host<int8_t>();
        }
        
        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, srcOrigin, _dstOrigin);
            }
            MNN_CONCURRENCY_END();
        }
        if (tileCount % threadNumber != 0) {
            tFunction(0, tileCount / threadNumber * threadNumber, 1, tileCount, srcOrigin, _dstOrigin);
        }
        
        // do C4 real transpose if need
        if (mTranspose && ow != 1 && oh != 1) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                int step = UP_DIV(dc_4, threadNumber), start = (int)tId * step, num = ALIMIN(start + step, dc_4) - start;
                if (num > 0) {
                    int dims[] = {oh, ow, ow * 4, oh * 4};
                    for (int z = start; z < start + num; ++z) {
                        MNNTranspose8Bit(dstOrigin + z * ih * iw * 4, _dstOrigin + z * ih * iw * 4, &dims[0], 4); // C4
                    }
                }
            } MNN_CONCURRENCY_END();
        }
        
        if (mRelu) {
            const int dstZStep = ow * oh * 4;
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                for (int z = (int)tId; z < dc_4; z += threadNumber) {
                    MNNReluInt8(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, dstZStep);
                }
            }
            MNN_CONCURRENCY_END();
        }
    }
    return NO_ERROR;
}

}
