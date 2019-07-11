//
//  Convolution3x3Int8.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution3x3Int8.hpp"
#include "AutoTime.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "Int8FunctionsOpt.h"
#include "Macro.h"
#include "TensorUtils.hpp"

#define SOURCE_BLOCK 64
#define WEIGHT_BLOCK 256
#define SOURCE_BLOCK_VEC 16
#define BLOCK_UNIT 4
#define BLOCK_UNIT2 16
#define EXTRA_SHIFT 2
#define CONVOLUTION1x1_INT16_UNIT 12

extern "C" {
void MNNGemmInt16to32_4x4_Unit(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad,
                               size_t dst_step, size_t dst_depth_quad);
void MNNGemmInt16to32_4x4_Common(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad,
                                 size_t width, size_t dst_step, size_t dst_depth_quad);
}

#ifndef MNN_USE_NEON
void MNNGemmInt16to32_4x4_Unit(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad,
                               size_t dst_step, size_t dst_depth_quad) {
    MNNGemmInt16to32_4x4_Common(dst, src, weight, src_depth_quad, CONVOLUTION1x1_INT16_UNIT, dst_step, dst_depth_quad);
}

void MNNGemmInt16to32_4x4_Common(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad,
                                 size_t width, size_t dst_step, size_t dst_depth_quad) {
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto weight_dz = weight + src_depth_quad * dz * 16;
        auto dst_z     = dst + dz * dst_step;
        for (int w = 0; w < width; ++w) {
            auto dst_x = dst_z + 4 * w;
            ::memset(dst_x, 0, 4 * sizeof(int32_t));
            auto src_x = src + 4 * w;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                auto weight_sz = weight_dz + 16 * sz;
                auto src_z     = src_x + sz * width * 4;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        dst_x[j] += (int32_t)src_z[i] * (int32_t)weight_sz[4 * i + j];
                    }
                }
            }
        }
    }
}
#endif

namespace MNN {
Convolution3x3Int8::Convolution3x3Int8(const Convolution2DCommon* convOp, Backend* b,
                                       const ConvolutionIntFactory::Int8Common* common, const float* bias,
                                       size_t biasSize)
    : MNN::CPUConvolution(convOp, b) {
    mSrcCopyInt8Buffer.buffer().type  = halide_type_of<int8_t>();
    mTileDstFloatBuffer.buffer().type = halide_type_of<float>();
    mTileDstInt32Buffer.buffer().type = halide_type_of<int32_t>();
    mTileSrcInt16Buffer.buffer().type = halide_type_of<int16_t>();
    mQuan                             = common->quan;
    int outputCount                   = (int)biasSize;
    int srcCount                      = common->weight.size() / 9 / outputCount;

    AutoStorage<int16_t> tempWeight(16 * srcCount * outputCount);
    float scale = 1 << EXTRA_SHIFT;
    mAlpha.reset(ALIGN_UP4(outputCount));
    mAlpha.clear();
    for (int i = 0; i < biasSize; ++i) {
        mAlpha.get()[i] = common->alpha.get()[i] / scale;
    }
    mAMin      = common->quan->aMin();
    mAMax      = common->quan->aMax();
    for (int i=0; i<4; ++i) {
        mQuanScale[i] = common->quan->quantScale();
    }
    mBias.reset(ALIGN_UP4(outputCount));
    mBias.clear();
    ::memcpy(mBias.get(), bias, biasSize * sizeof(float));
    auto weight     = tempWeight.get();
    auto quanWeight = common->weight.get();
    for (int dz = 0; dz < outputCount; ++dz) {
        for (int sz = 0; sz < srcCount; ++sz) {
            auto src = quanWeight + 9 * (sz + dz * srcCount);
            auto dst = weight + 16 * (sz + dz * srcCount);
            int16_t m00;
            int16_t m01;
            int16_t m02;
            int16_t m10;
            int16_t m11;
            int16_t m12;
            int16_t m20;
            int16_t m21;
            int16_t m22;
            int16_t m30;
            int16_t m31;
            int16_t m32;
            {
                int8_t* k = src;
                m00       = 2 * (int16_t)k[0];
                m01       = 2 * (int16_t)k[1];
                m02       = 2 * (int16_t)k[2];
                m10       = (int16_t)k[0] + (int16_t)k[3] + (int16_t)k[6];
                m11       = (int16_t)k[1] + (int16_t)k[4] + (int16_t)k[7];
                m12       = (int16_t)k[2] + (int16_t)k[5] + (int16_t)k[8];
                m20       = (int16_t)k[0] - (int16_t)k[3] + (int16_t)k[6];
                m21       = (int16_t)k[1] - (int16_t)k[4] + (int16_t)k[7];
                m22       = (int16_t)k[2] - (int16_t)k[5] + (int16_t)k[8];
                m30       = 2 * (int16_t)k[6];
                m31       = 2 * (int16_t)k[7];
                m32       = 2 * (int16_t)k[8];
            }

            {
                auto k = dst;
                k[0]   = 2 * m00;
                k[1]   = m00 + m01 + m02;
                k[2]   = m00 - m01 + m02;
                k[3]   = 2 * m02;
                k[4]   = 2 * m10;
                k[5]   = m10 + m11 + m12;
                k[6]   = m10 - m11 + m12;
                k[7]   = 2 * m12;
                k[8]   = 2 * m20;
                k[9]   = m20 + m21 + m22;
                k[10]  = m20 - m21 + m22;
                k[11]  = 2 * m22;
                k[12]  = 2 * m30;
                k[13]  = m30 + m31 + m32;
                k[14]  = m30 - m31 + m32;
                k[15]  = 2 * m32;
            }
        }
    }
    // Reorder
    int reorderWeightSize = UP_DIV(srcCount, 4) * UP_DIV(outputCount, 4) * WEIGHT_BLOCK;
    mWeight.reset(reorderWeightSize);
    mWeight.clear();
    int16_t* reorderedWeight = mWeight.get();
    int cur                  = 0;
    int srcDepthD4           = UP_DIV(srcCount, 4);
    int dstDepthD4           = UP_DIV(outputCount, 4);
    for (int dz = 0; dz < outputCount; ++dz) {
        auto dz_4   = dz / 4;
        auto mx     = dz % 4;
        auto dst_dz = reorderedWeight + dz_4 * srcDepthD4 * 16;
        for (int sz = 0; sz < srcCount; ++sz) {
            auto sz_4   = sz / 4;
            auto my     = sz % 4;
            auto dst_sz = dst_dz + sz_4 * 16;
            for (int ki = 0; ki < BLOCK_UNIT2; ++ki) {
                auto dst_i         = dst_sz + ki * srcDepthD4 * dstDepthD4 * 16;
                dst_i[4 * my + mx] = weight[cur++];
            }
        }
    }
}

ErrorCode Convolution3x3Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];
    int ow      = output->width();
    int oh      = output->height();
    int wUnit   = UP_DIV(ow, 2);
    int hUnit   = UP_DIV(oh, 2);
    int ic_4    = UP_DIV(input->channel(), 4);
    int dc_4    = UP_DIV(output->channel(), 4);

    int totalCount = hUnit * wUnit;
    int tileCount  = UP_DIV(totalCount, CONVOLUTION1x1_INT16_UNIT);

    int number = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    number     = std::min(number, tileCount);

    TensorUtils::copyShape(input, &mSrcCopyInt8Buffer);
    mSrcCopyInt8Buffer.buffer().dim[0].extent = 1;
    TensorUtils::setLinearLayout(&mSrcCopyInt8Buffer);

    mTileSrcInt16Buffer.buffer().dimensions    = 2;
    mTileSrcInt16Buffer.buffer().dim[0].extent = number;
    mTileSrcInt16Buffer.buffer().dim[1].extent = CONVOLUTION1x1_INT16_UNIT * SOURCE_BLOCK * ic_4 + SOURCE_BLOCK;
    TensorUtils::setLinearLayout(&mTileSrcInt16Buffer);

    mTileDstInt32Buffer.buffer().dimensions    = 2;
    mTileDstInt32Buffer.buffer().dim[0].extent = number;
    mTileDstInt32Buffer.buffer().dim[1].extent = dc_4 * CONVOLUTION1x1_INT16_UNIT * SOURCE_BLOCK;
    TensorUtils::setLinearLayout(&mTileDstInt32Buffer);

    mTileDstFloatBuffer.buffer().dimensions    = 2;
    mTileDstFloatBuffer.buffer().dim[0].extent = number;
    mTileDstFloatBuffer.buffer().dim[1].extent = SOURCE_BLOCK;
    TensorUtils::setLinearLayout(&mTileDstFloatBuffer);

    auto bn = backend();
    bn->onAcquireBuffer(&mSrcCopyInt8Buffer, Backend::DYNAMIC);
    bn->onAcquireBuffer(&mTileDstFloatBuffer, Backend::DYNAMIC);
    bn->onAcquireBuffer(&mTileDstInt32Buffer, Backend::DYNAMIC);
    bn->onAcquireBuffer(&mTileSrcInt16Buffer, Backend::DYNAMIC);

    bn->onReleaseBuffer(&mSrcCopyInt8Buffer, Backend::DYNAMIC);
    bn->onReleaseBuffer(&mTileDstFloatBuffer, Backend::DYNAMIC);
    bn->onReleaseBuffer(&mTileDstInt32Buffer, Backend::DYNAMIC);
    bn->onReleaseBuffer(&mTileSrcInt16Buffer, Backend::DYNAMIC);

    return NO_ERROR;
}

typedef int16_t int16_4[4];
typedef int32_t int32_4[4];
typedef float float_4[4];

static void __SourceTransform(const int16_t* srcBlock, int16_t* dstStart, size_t step) {
    auto _x = (int16_t*)srcBlock;
    int16_4 m00;
    int16_4 m01;
    int16_4 m02;
    int16_4 m03;
    int16_4 m10;
    int16_4 m11;
    int16_4 m12;
    int16_4 m13;
    int16_4 m20;
    int16_4 m21;
    int16_4 m22;
    int16_4 m23;
    int16_4 m30;
    int16_4 m31;
    int16_4 m32;
    int16_4 m33;

    // Transoform The Block
    for (int j = 0; j < 4; ++j) {
        m00[j]         = _x[j + 4 * 0] + -_x[j + 4 * 8];
        m01[j]         = _x[j + 4 * 1] + -_x[j + 4 * 9];
        m02[j]         = _x[j + 4 * 2] + -_x[j + 4 * 10];
        m03[j]         = _x[j + 4 * 3] + -_x[j + 4 * 11];
        m10[j]         = _x[j + 4 * 4] + _x[j + 4 * 8];
        m11[j]         = _x[j + 4 * 5] + _x[j + 4 * 9];
        m12[j]         = _x[j + 4 * 6] + _x[j + 4 * 10];
        m13[j]         = _x[j + 4 * 7] + _x[j + 4 * 11];
        m20[j]         = -_x[j + 4 * 4] + _x[j + 4 * 8];
        m21[j]         = -_x[j + 4 * 5] + _x[j + 4 * 9];
        m22[j]         = -_x[j + 4 * 6] + _x[j + 4 * 10];
        m23[j]         = -_x[j + 4 * 7] + _x[j + 4 * 11];
        m30[j]         = -_x[j + 4 * 4] + _x[j + 4 * 12];
        m31[j]         = -_x[j + 4 * 5] + _x[j + 4 * 13];
        m32[j]         = -_x[j + 4 * 6] + _x[j + 4 * 14];
        m33[j]         = -_x[j + 4 * 7] + _x[j + 4 * 15];
        _x[j + 4 * 0]  = m00[j] - m02[j];
        _x[j + 4 * 1]  = m01[j] + m02[j];
        _x[j + 4 * 2]  = -m01[j] + m02[j];
        _x[j + 4 * 3]  = -m01[j] + m03[j];
        _x[j + 4 * 4]  = m10[j] - m12[j];
        _x[j + 4 * 5]  = m11[j] + m12[j];
        _x[j + 4 * 6]  = -m11[j] + m12[j];
        _x[j + 4 * 7]  = -m11[j] + m13[j];
        _x[j + 4 * 8]  = m20[j] - m22[j];
        _x[j + 4 * 9]  = m21[j] + m22[j];
        _x[j + 4 * 10] = -m21[j] + m22[j];
        _x[j + 4 * 11] = -m21[j] + m23[j];
        _x[j + 4 * 12] = m30[j] - m32[j];
        _x[j + 4 * 13] = m31[j] + m32[j];
        _x[j + 4 * 14] = -m31[j] + m32[j];
        _x[j + 4 * 15] = -m31[j] + m33[j];
    }

    for (int v = 0; v < BLOCK_UNIT2; ++v) {
        auto __dst = dstStart + v * step;
        //        ic_4*4*xC;
        auto __src = srcBlock + 4 * v;
        for (int u = 0; u < 4; ++u) {
            __dst[u] = __src[u];
        }
    }
}
static void __DestTransform(const int32_t* srcZ, float* dstBlock, size_t step, const float* alpha) {
    float yy[SOURCE_BLOCK];
    float_4 m00;
    float_4 m01;
    float_4 m02;
    float_4 m03;
    float_4 m10;
    float_4 m11;
    float_4 m12;
    float_4 m13;
    for (int v = 0; v < BLOCK_UNIT2; ++v) {
        auto __src = srcZ + v * step;
        auto __dst = yy + 4 * v;
        for (int u = 0; u < 4; ++u) {
            __dst[u] = __src[u];
        }
    }
    for (int j = 0; j < 4; ++j) {
        auto a              = alpha[j];
        m00[j]              = yy[j + 4 * 0] + yy[j + 4 * 4] + yy[j + 4 * 8];
        m01[j]              = yy[j + 4 * 1] + yy[j + 4 * 5] + yy[j + 4 * 9];
        m02[j]              = yy[j + 4 * 2] + yy[j + 4 * 6] + yy[j + 4 * 10];
        m03[j]              = yy[j + 4 * 3] + yy[j + 4 * 7] + yy[j + 4 * 11];
        m10[j]              = yy[j + 4 * 4] - yy[j + 4 * 8] + yy[j + 4 * 12];
        m11[j]              = yy[j + 4 * 5] - yy[j + 4 * 9] + yy[j + 4 * 13];
        m12[j]              = yy[j + 4 * 6] - yy[j + 4 * 10] + yy[j + 4 * 14];
        m13[j]              = yy[j + 4 * 7] - yy[j + 4 * 11] + yy[j + 4 * 15];
        dstBlock[j + 4 * 0] = a * (m00[j] + m01[j] + m02[j]);
        dstBlock[j + 4 * 1] = a * (m01[j] - m02[j] + m03[j]);
        dstBlock[j + 4 * 2] = a * (m10[j] + m11[j] + m12[j]);
        dstBlock[j + 4 * 3] = a * (m11[j] - m12[j] + m13[j]);
    }
}

ErrorCode Convolution3x3Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    AUTOTIME;
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
    int totalCount    = hUnit * wUnit;
    int tileCount     = UP_DIV(totalCount, CONVOLUTION1x1_INT16_UNIT);
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);

    int threadNumber = std::min(((CPUBackend*)backend())->threadNumber(), tileCount);

    int batchSize = input->batch();
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        auto srcOrigin = input->host<float>() + batchIndex * input->buffer().dim[0].stride;
        auto dstOrigin = output->host<float>() + batchIndex * output->buffer().dim[0].stride;

        int inputTotalSize = iw * ih * ALIGN_UP4(input->channel());
        int8_t* srcCopy    = mSrcCopyInt8Buffer.host<int8_t>();

        MNNFloat2Int8(srcOrigin, srcCopy, inputTotalSize / 4, mQuanScale, mAMin, mAMax);
        // MNN_PRINT("%d, %d, %d, %d\n", wUnit, hUnit, layer->aMin, layer->aMax);
        auto threadFunction = [&](size_t tId) {
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex  = (int)tIndex * CONVOLUTION1x1_INT16_UNIT;
                int xReamin = totalCount - xIndex;
                int xC      = xReamin > CONVOLUTION1x1_INT16_UNIT ? CONVOLUTION1x1_INT16_UNIT : xReamin;

                auto _srcOrigin =
                    mTileSrcInt16Buffer.host<int16_t>() + mTileSrcInt16Buffer.buffer().dim[0].stride * tId;
                auto srcBlock = _srcOrigin + xC * SOURCE_BLOCK * (ic_4);

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

                    auto srcStart = srcCopy + (srcX + srcY * iw) * 4;

                    for (int z = 0; z < ic_4; ++z) {
                        ::memset(srcBlock, 0, SOURCE_BLOCK * sizeof(int16_t));

                        auto _dstStart = dstUnit + z * 4 * xC;

                        auto src_z = srcStart + z * 4 * iw * ih;
                        // Extract One Block
                        for (int yy = sy; yy < ey; ++yy) {
                            auto dst_yy = srcBlock + yy * 16;
                            auto src_yy = src_z + 4 * iw * yy;
                            for (int xx = sx; xx < ex; ++xx) {
                                for (int v = 0; v < 4; ++v) {
                                    dst_yy[xx * 4 + v] = src_yy[4 * xx + v];
                                }
                            }
                        }
                        // Transform
                        __SourceTransform(srcBlock, _dstStart, 4 * xC * ic_4);
                    }
                }

                auto _dstOrigin =
                    mTileDstInt32Buffer.host<int32_t>() + mTileDstInt32Buffer.buffer().dim[0].stride * tId;

                auto dstBlock = mTileDstFloatBuffer.host<float>() + mTileDstFloatBuffer.buffer().dim[0].stride * tId;

                // Multi
                if (xC == CONVOLUTION1x1_INT16_UNIT) {
                    for (int i = 0; i < BLOCK_UNIT2; ++i) {
                        auto dstI    = _dstOrigin + i * dc_4 * 4 * xC;
                        auto srcI    = _srcOrigin + i * ic_4 * 4 * xC;
                        auto weightI = mWeight.get() + i * 16 * ic_4 * dc_4;
                        MNNGemmInt16to32_4x4_Unit(dstI, srcI, weightI, ic_4, xC * 4, dc_4);
                    }
                } else {
                    for (int i = 0; i < BLOCK_UNIT2; ++i) {
                        auto dstI    = _dstOrigin + i * dc_4 * 4 * xC;
                        auto srcI    = _srcOrigin + i * ic_4 * 4 * xC;
                        auto weightI = mWeight.get() + i * 16 * ic_4 * dc_4;
                        MNNGemmInt16to32_4x4_Common(dstI, srcI, weightI, ic_4, xC, xC * 4, dc_4);
                    }
                }

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
                        __DestTransform(srcZ, dstBlock, dc_4 * 4 * xC, mAlpha.get() + 4 * z);

                        float* bias_z = mBias.get() + 4 * z;
                        postFunction(dstBlock, bias_z, 4, 1);
                        ::memcpy(dstZ, dstBlock, 4 * sizeof(float));
                        if (wIndex * 2 + 1 < ow) {
                            ::memcpy(dstZ + 4, dstBlock + 4, 4 * sizeof(float));
                        }
                        if (hIndex * 2 + 1 < oh) {
                            ::memcpy(dstZ + ow * 4, dstBlock + 8, 4 * sizeof(float));
                            if (wIndex * 2 + 1 < ow) {
                                ::memcpy(dstZ + ow * 4 + 4, dstBlock + 12, 4 * sizeof(float));
                            }
                        }
                    }
                }
            }
        };
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            threadFunction(tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
} // namespace MNN
