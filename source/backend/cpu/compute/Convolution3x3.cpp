//
//  Convolution3x3.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution3x3.hpp"
#include "AutoTime.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "ConvOpt.h"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "Vec4.hpp"
using namespace MNN::Math;

typedef Vec4 float4;

#define SOURCE_BLOCK 64
#define WEIGHT_BLOCK 256
#define SOURCE_BLOCK_VEC 16
#define SRC_BLOCK_UNIT 3
#define SRC_BLOCK_UNIT2 9
#define BLOCK_UNIT 4
#define BLOCK_UNIT2 16

namespace MNN {

void Convolution3x3::sourceTransform(const float* srcBlock, float* dstStart, size_t step) {
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

void Convolution3x3::destTransform(const float* srcZ, float* dstBlock, size_t step) {
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

void Convolution3x3::kernelTransform(float* reorderedWeight, const float* srcWeight, int srcCount, int outputCount) {
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

Convolution3x3::Convolution3x3(const Convolution2DCommon* convOp, Backend* b, const float* originWeight,
                               size_t originWeightSize, const float* bias, size_t biasSize)
    : MNN::CPUConvolution(convOp, b) {
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    mValid = backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
    auto outputCount                   = (int)biasSize;
    auto weightSize                    = originWeightSize;
    auto srcCount                      = (int)weightSize / 9 / outputCount;
    int number                         = std::max(((CPUBackend*)b)->threadNumber(), 1);
    mTempBuffer.buffer().dim[0].extent = number;
    mTempBuffer.buffer().dim[1].extent = CONVOLUTION_TILED_NUMBER;
    mTempBuffer.buffer().dim[2].extent = UP_DIV(srcCount, 4) + UP_DIV(outputCount, 4) + 1;
    mTempBuffer.buffer().dim[3].extent = SOURCE_BLOCK;

    TensorUtils::setLinearLayout(&mTempBuffer);

    auto srcWeight = originWeight;
    {
        // Reorder
        int srcDepthD4 = UP_DIV((int)srcCount, 4);
        int dstDepthD4 = UP_DIV((int)outputCount, 4);
        mWeight.reset(Tensor::createDevice<float>({srcDepthD4 * dstDepthD4 * WEIGHT_BLOCK}));
        mValid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        if (srcCount % 4 != 0 || outputCount % 4 != 0) {
            ::memset(mWeight->host<float>(), 0, mWeight->size());
        }
        float* reorderedWeight = mWeight->host<float>();
        kernelTransform(reorderedWeight, srcWeight, srcCount, outputCount);
    }
}
Convolution3x3::~Convolution3x3() {
    MNN_ASSERT(nullptr != mWeight);
    MNN_ASSERT(nullptr != mBias);
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}
ErrorCode Convolution3x3::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto output = outputs[0];
    int ow      = output->width();
    int oh      = output->height();
    int wUnit   = UP_DIV(ow, 2);
    int hUnit   = UP_DIV(oh, 2);

    int totalCount = hUnit * wUnit;
    int tileCount  = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);
    int number     = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    number         = std::min(number, tileCount);
    mInsideThread  = tileCount <= 1;
    mOutsideThread = tileCount > 1;
    if ((float)wUnit * hUnit * 16 * inputs[0]->channel() * output->channel() / 1024.0f / 1024.0f < 1) {
        // TODO Use common logic to reduce thread number
        number         = 1;
        mInsideThread  = false;
        mOutsideThread = false;
    }
    mTempBuffer.buffer().dim[0].extent = number;
    bool success                       = backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode Convolution3x3::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    AUTOTIME;
    auto input  = inputs[0];
    auto output = outputs[0];

    int ow   = output->width();
    int oh   = output->height();
    int iw   = input->width();
    int ih   = input->height();
    int ic_4 = UP_DIV(input->channel(), 4);
    int dc_4 = UP_DIV(output->channel(), 4);

    int padY = mPadY;
    int padX = mPadX;

    int wUnit = UP_DIV(ow, 2);
    int hUnit = UP_DIV(oh, 2);

    auto postFunction = mPostFunction;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    int threadNumber = mTempBuffer.length(0);

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = input->host<float>() + iw * ih * ic_4 * 4 * batchIndex;
        auto dstOrigin = output->host<float>() + ow * oh * dc_4 * 4 * batchIndex;
        int totalCount = hUnit * wUnit;

        int tileCount = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);
        threadNumber  = std::min(threadNumber, tileCount);

        auto weight                = mWeight->host<float>();
        auto bias                  = mBias->host<float>();
        auto sourceTransformLambda = [&](int xIndex, int xC, float* _srcOrigin, float* dstBlock) {
            // Source Transform
            for (int xi = 0; xi < xC; ++xi) {
                auto index   = xIndex + xi;
                auto dstUnit = _srcOrigin + 4 * xi;

                int wIndex = index % wUnit;
                int hIndex = index / wUnit;

                int srcX = wIndex * 2 - padX;
                int srcY = hIndex * 2 - padY;
                int sy   = ALIMAX(0, srcY) - srcY;
                int ey   = ALIMIN(srcY + 4, ih) - srcY;
                int sx   = ALIMAX(0, srcX) - srcX;
                int ex   = ALIMIN(srcX + 4, iw) - srcX;

                auto srcStart = srcOrigin + (srcX + srcY * iw) * 4;

                for (int z = 0; z < ic_4; ++z) {
                    ::memset(dstBlock, 0, SOURCE_BLOCK * sizeof(float));

                    auto _dstStart = dstUnit + z * 4 * xC;

                    auto src_z = srcStart + z * 4 * iw * ih;
                    if (ex > sx) {
                        // Extract One Block
                        for (int yy = sy; yy < ey; ++yy) {
                            auto dst_yy = dstBlock + yy * 16;
                            auto src_yy = src_z + 4 * iw * yy;
                            ::memcpy(dst_yy + 4 * sx, src_yy + sx * 4, 4 * (ex - sx) * sizeof(float));
                        }
                    }
                    // Transform
                    sourceTransform(dstBlock, _dstStart, 4 * xC * ic_4);
                }
            }
        };
        std::function<void(int, const float*, float*)> gemmFunctionLambda = [&](int xC, const float* _srcOrigin,
                                                                                float* _dstOrigin) {
            // Multi
            if (xC == CONVOLUTION_TILED_NUMBER) {
                for (int i = 0; i < BLOCK_UNIT2; ++i) {
                    MNNGemmFloatUnit_4(_dstOrigin + i * dc_4 * 4 * xC, _srcOrigin + i * ic_4 * 4 * xC,
                                       weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
                }
            } else {
                for (int i = 0; i < BLOCK_UNIT2; ++i) {
                    MNNGemmFloatCommon_4(_dstOrigin + (i * dc_4) * xC * 4, _srcOrigin + i * ic_4 * 4 * xC,
                                         weight + (i * dc_4) * ic_4 * 16, ic_4, xC * 4, dc_4, xC, 0);
                }
            }
        };
        if (mInsideThread) {
            auto insideThreadNumber = ((CPUBackend*)backend())->threadNumber();
            gemmFunctionLambda      = [&](int xC, const float* _srcOrigin, float* _dstOrigin) {
                // Multi
                if (xC == CONVOLUTION_TILED_NUMBER) {
                    MNN_CONCURRENCY_BEGIN(tId, insideThreadNumber) {
                        for (int i = (int)tId; i < BLOCK_UNIT2; i += insideThreadNumber) {
                            MNNGemmFloatUnit_4(_dstOrigin + i * dc_4 * 4 * xC, _srcOrigin + i * ic_4 * 4 * xC,
                                               weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
                        }
                    }
                    MNN_CONCURRENCY_END();
                } else {
                    MNN_CONCURRENCY_BEGIN(tId, insideThreadNumber) {
                        for (int i = (int)tId; i < BLOCK_UNIT2; i += insideThreadNumber) {
                            MNNGemmFloatCommon_4(_dstOrigin + (i * dc_4) * xC * 4, _srcOrigin + i * ic_4 * 4 * xC,
                                                 weight + (i * dc_4) * ic_4 * 16, ic_4, xC * 4, dc_4, xC, 0);
                        }
                    }
                    MNN_CONCURRENCY_END();
                }
            };
        }
        auto outsideFunction = [&](int tId) {
            auto _srcOrigin = mTempBuffer.host<float>() + tId * mTempBuffer.buffer().dim[0].stride;
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex      = (int)tIndex * CONVOLUTION_TILED_NUMBER;
                int xReamin     = totalCount - xIndex;
                int xC          = xReamin > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : xReamin;
                auto dstBlock   = _srcOrigin + xC * SOURCE_BLOCK * (ic_4 + dc_4);
                auto _dstOrigin = _srcOrigin + xC * SOURCE_BLOCK * ic_4;

                sourceTransformLambda(xIndex, xC, _srcOrigin, dstBlock);
                gemmFunctionLambda(xC, _srcOrigin, _dstOrigin);

                // Dest Transform
                for (int xi = 0; xi < xC; ++xi) {
                    auto index   = xIndex + xi;
                    auto srcUnit = _dstOrigin + 4 * xi;

                    int wIndex = index % wUnit;
                    int hIndex = index / wUnit;

                    int dstX = wIndex * 2;
                    int dstY = hIndex * 2;

                    auto dstStart = dstOrigin + 4 * (dstX + dstY * ow);

                    for (int z = 0; z < dc_4; ++z) {
                        auto srcZ = srcUnit + z * xC * 4;
                        auto dstZ = dstStart + z * ow * oh * 4;
                        destTransform(srcZ, dstBlock, dc_4 * 4 * xC);

                        float* bias_z = bias + 4 * z;
                        postFunction(dstBlock, bias_z, 4, 1);
                        Vec4::save(dstZ, Vec4::load(dstBlock));
                        if (wIndex * 2 + 1 < ow) {
                            Vec4::save(dstZ + 4, Vec4::load(dstBlock + 4));
                        }
                        if (hIndex * 2 + 1 < oh) {
                            Vec4::save(dstZ + ow * 4, Vec4::load(dstBlock + 8));
                            if (wIndex * 2 + 1 < ow) {
                                Vec4::save(dstZ + ow * 4 + 4, Vec4::load(dstBlock + 12));
                            }
                        }
                    }
                }
            }
        };
        if (mOutsideThread) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                outsideFunction((int)tId);
            };
            MNN_CONCURRENCY_END();
        } else {
            outsideFunction(0);
        }
    }

    return NO_ERROR;
}

} // namespace MNN
