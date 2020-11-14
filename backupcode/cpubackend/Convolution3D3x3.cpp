//
//  Convolution3D3x3.cpp
//  MNN
//
//  Created by MNN on 2019/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "../CPUBackend.hpp"
#include "backend/cpu/compute/Convolution3D3x3.hpp"
#include <MNN/AutoTime.hpp>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;

#define SOURCE_BLOCK 64
#define WEIGHT_BLOCK 256
#define SOURCE_BLOCK_VEC 16
#define SRC_BLOCK_UNIT 3
#define SRC_BLOCK_UNIT2 9
#define BLOCK_UNIT 4
#define BLOCK_UNIT2 16
#define SOURCE_BLOCK 64
#define BLOCK_UNIT2 16

namespace MNN {
static void sourceTransform(const float* srcBlock, float* dstStart, size_t step) {
    auto _x = (float*)srcBlock;
    float4 m00;
    float4 m01;
    float4 m02;
    float4 m03;
    float4 m10;
    float4 m11;
    float4 m12;
    float4 m13;
    float4 m20;
    float4 m21;
    float4 m22;
    float4 m23;
    float4 m30;
    float4 m31;
    float4 m32;
    float4 m33;
    auto _y = dstStart;
    m00     = Vec4::load(_x + 4 * 0) - Vec4::load(_x + 4 * 8);
    m01     = Vec4::load(_x + 4 * 1) - Vec4::load(_x + 4 * 9);
    m02     = Vec4::load(_x + 4 * 2) - Vec4::load(_x + 4 * 10);
    m03     = Vec4::load(_x + 4 * 3) - Vec4::load(_x + 4 * 11);
    m10     = Vec4::load(_x + 4 * 4) + Vec4::load(_x + 4 * 8);
    m11     = Vec4::load(_x + 4 * 5) + Vec4::load(_x + 4 * 9);
    m12     = Vec4::load(_x + 4 * 6) + Vec4::load(_x + 4 * 10);
    m13     = Vec4::load(_x + 4 * 7) + Vec4::load(_x + 4 * 11);
    m20     = Vec4::load(_x + 4 * 8) - Vec4::load(_x + 4 * 4);
    m21     = Vec4::load(_x + 4 * 9) - Vec4::load(_x + 4 * 5);
    m22     = Vec4::load(_x + 4 * 10) - Vec4::load(_x + 4 * 6);
    m23     = Vec4::load(_x + 4 * 11) - Vec4::load(_x + 4 * 7);
    m30     = Vec4::load(_x + 4 * 12) - Vec4::load(_x + 4 * 4);
    m31     = Vec4::load(_x + 4 * 13) - Vec4::load(_x + 4 * 5);
    m32     = Vec4::load(_x + 4 * 14) - Vec4::load(_x + 4 * 6);
    m33     = Vec4::load(_x + 4 * 15) - Vec4::load(_x + 4 * 7);

    Vec4::save(_y + step * 0, m00 - m02);
    Vec4::save(_y + step * 1, m01 + m02);
    Vec4::save(_y + step * 2, m02 - m01);
    Vec4::save(_y + step * 3, m03 - m01);
    Vec4::save(_y + step * 4, m10 - m12);
    Vec4::save(_y + step * 5, m11 + m12);
    Vec4::save(_y + step * 6, m12 - m11);
    Vec4::save(_y + step * 7, m13 - m11);
    Vec4::save(_y + step * 8, m20 - m22);
    Vec4::save(_y + step * 9, m21 + m22);
    Vec4::save(_y + step * 10, m22 - m21);
    Vec4::save(_y + step * 11, m23 - m21);
    Vec4::save(_y + step * 12, m30 - m32);
    Vec4::save(_y + step * 13, m31 + m32);
    Vec4::save(_y + step * 14, m32 - m31);
    Vec4::save(_y + step * 15, m33 - m31);
}

static void destTransform(const float* srcZ, float* dstBlock, size_t step) {
    auto yy = dstBlock;
    float4 m00;
    float4 m01;
    float4 m02;
    float4 m03;
    float4 m10;
    float4 m11;
    float4 m12;
    float4 m13;
    auto x = srcZ;
    m00    = Vec4::load(x + step * 0) + Vec4::load(x + step * 4) + Vec4::load(x + step * 8);
    m01    = Vec4::load(x + step * 1) + Vec4::load(x + step * 5) + Vec4::load(x + step * 9);
    m02    = Vec4::load(x + step * 2) + Vec4::load(x + step * 6) + Vec4::load(x + step * 10);
    m03    = Vec4::load(x + step * 3) + Vec4::load(x + step * 7) + Vec4::load(x + step * 11);
    m10    = Vec4::load(x + step * 4) - Vec4::load(x + step * 8) + Vec4::load(x + step * 12);
    m11    = Vec4::load(x + step * 5) - Vec4::load(x + step * 9) + Vec4::load(x + step * 13);
    m12    = Vec4::load(x + step * 6) - Vec4::load(x + step * 10) + Vec4::load(x + step * 14);
    m13    = Vec4::load(x + step * 7) - Vec4::load(x + step * 11) + Vec4::load(x + step * 15);
    Vec4::save(yy + 4 * 0, m00 + m01 + m02);
    Vec4::save(yy + 4 * 1, m01 - m02 + m03);
    Vec4::save(yy + 4 * 2, m10 + m11 + m12);
    Vec4::save(yy + 4 * 3, m11 - m12 + m13);
}

static void kernelTransform(float* reorderedWeight, const float* srcWeight, int srcCount, int outputCount) {
    float weight[BLOCK_UNIT2];
    int srcDepthD4 = UP_DIV((int)srcCount, 4);
    int dstDepthD4 = UP_DIV((int)outputCount, 4);

    for (int dz = 0; dz < outputCount; ++dz) {
        auto dz_4   = dz / BLOCK_UNIT;
        auto mx     = dz % BLOCK_UNIT;
        auto dst_dz = reorderedWeight + dz_4 * srcDepthD4 * 16;
        for (int sz = 0; sz < srcCount; ++sz) {
            auto sz_4   = sz / BLOCK_UNIT;
            auto my     = sz % BLOCK_UNIT;
            auto dst_sz = dst_dz + sz_4 * BLOCK_UNIT2;
            auto src    = srcWeight + SRC_BLOCK_UNIT2 * (sz + dz * srcCount);
            auto dst    = weight;
            float* k    = (float*)src;
            float m00;
            float m01;
            float m02;
            float m10;
            float m11;
            float m12;
            float m20;
            float m21;
            float m22;
            float m30;
            float m31;
            float m32;
            m00 = k[0];
            m01 = k[1];
            m02 = k[2];
            m10 = 0.500000 * k[0] + 0.500000 * k[3] + 0.500000 * k[6];
            m11 = 0.500000 * k[1] + 0.500000 * k[4] + 0.500000 * k[7];
            m12 = 0.500000 * k[2] + 0.500000 * k[5] + 0.500000 * k[8];
            m20 = 0.500000 * k[0] + -0.500000 * k[3] + 0.500000 * k[6];
            m21 = 0.500000 * k[1] + -0.500000 * k[4] + 0.500000 * k[7];
            m22 = 0.500000 * k[2] + -0.500000 * k[5] + 0.500000 * k[8];
            m30 = 0 + k[6];
            m31 = 0 + k[7];
            m32 = 0 + k[8];

            k     = dst;
            k[0]  = m00;
            k[1]  = 0.500000 * m00 + 0.500000 * m01 + 0.500000 * m02;
            k[2]  = 0.500000 * m00 + -0.500000 * m01 + 0.500000 * m02;
            k[3]  = 0 + m02;
            k[4]  = m10;
            k[5]  = 0.500000 * m10 + 0.500000 * m11 + 0.500000 * m12;
            k[6]  = 0.500000 * m10 + -0.500000 * m11 + 0.500000 * m12;
            k[7]  = 0 + m12;
            k[8]  = m20;
            k[9]  = 0.500000 * m20 + 0.500000 * m21 + 0.500000 * m22;
            k[10] = 0.500000 * m20 + -0.500000 * m21 + 0.500000 * m22;
            k[11] = 0 + m22;
            k[12] = m30;
            k[13] = 0.500000 * m30 + 0.500000 * m31 + 0.500000 * m32;
            k[14] = 0.500000 * m30 + -0.500000 * m31 + 0.500000 * m32;
            k[15] = 0 + m32;

            for (int ki = 0; ki < BLOCK_UNIT2; ++ki) {
                auto dst_i         = dst_sz + ki * srcDepthD4 * dstDepthD4 * 16;
                dst_i[4 * my + mx] = weight[ki];
            }
        }
    }
}

Convolution3D3x3::Convolution3D3x3(const Convolution3DCommon* convOp, Backend *b, const float* originWeight,
                                   int originWeightSize, const float* bias, int biasSize) : Execution(b) {
    mPadMode = convOp->padMode();
    if (mPadMode != PadMode_SAME) {
        for (int32_t pad: *(convOp->pads())) {
            mPads.push_back(pad);
        }
    }
    mKernelDepth = (*(convOp->kernels()))[0];
    mPostFunction = CPUConvolution3D::getPostFunction(convOp);

    int inputChannel = convOp->inputCount();
    int outputChannel = convOp->outputCount();

    mWeight.reset(Tensor::createDevice<float>({ALIGN_UP4(inputChannel) * ALIGN_UP4(outputChannel) * mKernelDepth * BLOCK_UNIT2}));
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    bool valid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    valid = valid && backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!valid) {
        return;
    }

    memset(mBias->host<float>(), 0, mBias->size());
    memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));

    if (inputChannel % 4 != 0 || outputChannel % 4 != 0) {
        memset(mWeight->host<float>(), 0, mWeight->size());
    }
    const int srcDepthStep = inputChannel * outputChannel * 9;
    const int dstDepthStep = ALIGN_UP4(inputChannel) * ALIGN_UP4(outputChannel) * BLOCK_UNIT2;
    for (int d = 0; d < mKernelDepth; ++d) {
        kernelTransform(mWeight->host<float>() + d * dstDepthStep, originWeight + d * srcDepthStep, inputChannel, outputChannel);
    }
}
Convolution3D3x3::~Convolution3D3x3() {
    MNN_ASSERT(nullptr != mWeight);
    MNN_ASSERT(nullptr != mBias);
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}
ErrorCode Convolution3D3x3::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    const int oc = output->length(1), od = output->length(2);
    const int ic = input->length(1), id = input->length(2);
    const int threadNumber = ((CPUBackend*)backend())->threadNumber();

    if (mPadMode == PadMode_SAME) {
        mPads.clear();
        auto kernels = std::vector<int32_t>({mKernelDepth, 3, 3});
        for (int i = 0; i < 3; ++i) {
            int inputNeeded = output->length(i + 2) - 1 + kernels[i]; // stride = dialate = 1
            mPads.push_back((inputNeeded - input->length(i + 2)) / 2);
        }
    }
    auto CONVOLUTION_TILED_NUMBER = MNNGetConvolutionTileNumber();

    mSourceBuffer.reset(Tensor::createDevice<float>({threadNumber, id, BLOCK_UNIT2, UP_DIV(ic, 4), CONVOLUTION_TILED_NUMBER, 4}));
    mDestBuffer.reset(Tensor::createDevice<float>({threadNumber, od + 1, BLOCK_UNIT2, UP_DIV(oc, 4), CONVOLUTION_TILED_NUMBER, 4}));
    mTempBuffer.reset(Tensor::createDevice<float>({threadNumber, BLOCK_UNIT2, 4}));

    bool succ = backend()->onAcquireBuffer(mSourceBuffer.get(), Backend::DYNAMIC);
    succ = succ && backend()->onAcquireBuffer(mDestBuffer.get(), Backend::DYNAMIC);
    succ = succ && backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    if (!succ) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mSourceBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mDestBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode Convolution3D3x3::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    AUTOTIME;
    auto input  = inputs[0];
    auto output = outputs[0];
    auto CONVOLUTION_TILED_NUMBER = MNNGetConvolutionTileNumber();

    const int inputWidth = input->length(4), inputHeight = input->length(3), inputDepth = input->length(2), ic_4 = UP_DIV(input->length(1), 4);
    const int outputWidth = output->length(4), outputHeight = output->length(3), outputDepth = output->length(2), dc_4 = UP_DIV(output->length(1), 4);
    const int padDepth = mPads[0], padHeight = mPads[1], padWidth = mPads[2], kernelDepth = mKernelDepth;
    const int wUnit = UP_DIV(outputWidth, 2), hUnit = UP_DIV(outputHeight, 2), totalCount = wUnit * hUnit;
    const int tileCount = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);

    auto postFunction = mPostFunction;
    // MNN_PRINT("outputWidth=%d, outputHeight=%d\n", outputWidth, outputHeight);
    const int threadNumber = ((CPUBackend*)backend())->threadNumber();

    auto sourceTransformFunc = [=](int xIndex, int xC, const float* srcOrigin, float* _srcOrigin, float* dstBlock) {
        const int dstStepD = BLOCK_UNIT2 * ic_4 * xC * 4;
        // Source Transform
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto dstUnit = _srcOrigin + 4 * xi;

            int wIndex = index % wUnit;
            int hIndex = index / wUnit;

            int srcX = wIndex * 2 - padWidth;
            int srcY = hIndex * 2 - padHeight;
            int sy   = ALIMAX(0, srcY) - srcY;
            int ey   = ALIMIN(srcY + 4, inputHeight) - srcY;
            int sx   = ALIMAX(0, srcX) - srcX;
            int ex   = ALIMIN(srcX + 4, inputWidth) - srcX;

            auto srcStart = srcOrigin + (srcX + srcY * inputWidth) * 4;
            memset(dstBlock, 0, SOURCE_BLOCK * sizeof(float));

            for (int z = 0; z < ic_4; ++z) {
                auto dstStart = dstUnit + z * 4 * xC;
                auto src_z = srcStart + z * 4 * inputWidth * inputHeight * inputDepth;
                for (int d = 0; d < inputDepth; ++d) {
                    if (ex > sx) {
                        // Extract One Block
                        for (int yy = sy; yy < ey; ++yy) {
                            auto dst_yy = dstBlock + yy * 16;
                            auto src_yy = src_z + (d * inputHeight + yy) * inputWidth * 4;
                            memcpy(dst_yy + 4 * sx, src_yy + sx * 4, 4 * (ex - sx) * sizeof(float));
                        }
                    }
                    // Transform
                    sourceTransform(dstBlock, dstStart + d * dstStepD, 4 * xC * ic_4);
                }
            }
        }
    };

    auto destTransformFunc = [=](int xIndex, int xC, const float* srcOrigin, float* dstOrigin, float* dstBlock) {
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto srcUnit = srcOrigin + 4 * xi;

            int wIndex = index % wUnit;
            int hIndex = index / wUnit;

            int dstX = wIndex * 2;
            int dstY = hIndex * 2;

            auto dstStart = dstOrigin + 4 * (dstX + dstY * outputWidth);

            for (int od = 0; od < outputDepth; ++od) {
                auto _srcUnit = srcUnit + od * BLOCK_UNIT2 * dc_4 * xC * 4;
                auto _dstStart = dstStart + od * outputHeight * outputWidth * 4;
                for (int z = 0; z < dc_4; ++z) {
                    auto srcZ = _srcUnit + z * xC * 4;
                    auto dstZ = _dstStart + z * outputDepth * outputWidth * outputHeight * 4;
                    destTransform(srcZ, dstBlock, dc_4 * 4 * xC);

                    Vec4::save(dstZ, Vec4::load(dstBlock));
                    if (wIndex * 2 + 1 < outputWidth) {
                        Vec4::save(dstZ + 4, Vec4::load(dstBlock + 4));
                    }
                    if (hIndex * 2 + 1 < outputHeight) {
                        Vec4::save(dstZ + outputWidth * 4, Vec4::load(dstBlock + 8));
                        if (wIndex * 2 + 1 < outputWidth) {
                            Vec4::save(dstZ + outputWidth * 4 + 4, Vec4::load(dstBlock + 12));
                        }
                    }
                }
            }
        }
    };

    auto gemmFunc = [=](int xC, int start, int end, const float* srcOrigin, const float* weight, float* dstOrigin) {
        float* tempDst = dstOrigin + outputDepth * BLOCK_UNIT2 * dc_4 * xC * 4;
        const int element = (end - start) * dc_4 * xC * 4, offset = start * dc_4 * xC * 4;
        for (int od = 0; od < outputDepth; ++od) {
            bool add = false;
            float* _dstOrigin = dstOrigin + (od * BLOCK_UNIT2 + start) * dc_4 * xC * 4;
            const int srcD = od - padDepth, kdStart = -ALIMIN(srcD, 0), kdEnd = kernelDepth - ALIMAX(srcD + kernelDepth - inputDepth, 0);
            for (int kd = kdStart; kd < kdEnd; ++kd) {
                const float* _srcOrigin = srcOrigin + (kd + srcD) * BLOCK_UNIT2 * ic_4 * xC * 4;
                const float* _weight = weight + kd * BLOCK_UNIT2 * dc_4 * ic_4 * 16;
                for (int i = start; i < end; ++i) {
                    if (xC == CONVOLUTION_TILED_NUMBER) {
                        MNNGemmFloatUnit_4(tempDst + i * dc_4 * xC * 4, _srcOrigin + i * ic_4 * 4 * xC,
                                           _weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
                    } else {
                        MNNGemmFloatCommon_4(tempDst + i * dc_4 * xC * 4, _srcOrigin + i * ic_4 * 4 * xC,
                                             _weight + (i * dc_4) * ic_4 * 16, ic_4, xC * 4, dc_4, xC, 0);
                    }
                }
                if (add) {
                    MNNMatrixAdd(_dstOrigin, _dstOrigin, tempDst + offset, element / 4, 0, 0, 0, 1);
                } else {
                    memcpy(_dstOrigin, tempDst + offset, element * sizeof(float));
                }
                add = true;
            }
        }
    };

    auto gemmConcurrencyFunc = [=, &gemmFunc](int xC, const float* _srcOrigin, const float* weight, float* _dstOrigin) {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            const int step = UP_DIV(BLOCK_UNIT2, threadNumber);
            gemmFunc(xC, tId * step, ALIMIN((tId + 1) * step, BLOCK_UNIT2), _srcOrigin, weight, _dstOrigin);
        }
        MNN_CONCURRENCY_END()
    };

    auto tFunction = [&](const int tId, const int tileStart, const int tileStep, const int tileEnd, const float* srcOrigin, float* dstOrigin) {
        auto _srcOrigin = mSourceBuffer->host<float>() + tId * mSourceBuffer->stride(0);
        auto _dstOrigin = mDestBuffer->host<float>() + tId * mDestBuffer->stride(0);
        auto dstBlock   = mTempBuffer->host<float>() + tId * mTempBuffer->stride(0);
        for (int tIndex = tileStart; tIndex < tileEnd; tIndex += tileStep) {
            int xIndex      = (int)tIndex * CONVOLUTION_TILED_NUMBER;
            int xReamin     = totalCount - xIndex;
            int xC          = xReamin > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : xReamin;

            sourceTransformFunc(xIndex, xC, srcOrigin, _srcOrigin, dstBlock);

            if (threadNumber != tileStep) {
                gemmConcurrencyFunc(xC, _srcOrigin, mWeight->host<float>(), _dstOrigin);
            } else {
                gemmFunc(xC, 0, BLOCK_UNIT2, _srcOrigin, mWeight->host<float>(), _dstOrigin);
            }

            destTransformFunc(xIndex, xC, _dstOrigin, dstOrigin, dstBlock);
        }
    };

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);
        auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);

        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, srcOrigin, dstOrigin);
            }
            MNN_CONCURRENCY_END();
        }

        if (tileCount % threadNumber != 0) {
            tFunction(0, tileCount / threadNumber * threadNumber, 1, tileCount, srcOrigin, dstOrigin);
        }

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int channelStep = UP_DIV(dc_4, threadNumber);
            int channelStart = channelStep * tId, channelNum = ALIMIN(channelStep * (tId + 1), dc_4) - channelStart;
            if (channelNum > 0) {
                postFunction(dstOrigin + channelStart * outputHeight * outputWidth * outputDepth * 4, mBias->host<float>() + 4 * channelStart, outputWidth * outputHeight * outputDepth, channelNum);
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
