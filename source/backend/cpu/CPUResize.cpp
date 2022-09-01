//
//  CPUResize.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUResize.hpp"
#include <math.h>
#include "core/AutoStorage.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;

extern "C" {
void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, size_t number);
void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t,
                    size_t number);
}
using namespace MNN::Math;
namespace MNN {

static void CPUBilinearSampleC4(const float* src, float* dst, const int32_t* position, const float* factor,
                                size_t number) {
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
        Vec4 df(f);
        Vec4 sf(1.0f - f);
        Vec4 A = Vec4::load(src + position[2 * i] * 4);
        Vec4 B = Vec4::load(src + position[2 * i + 1] * 4);
        Vec4::save(dst + 4 * i, B * df + A * sf);
    }
}

static void CPUBilinearLineC4(float* dst, const float* A, const float* B, const float* t, size_t number) {
    Vec4 df(*t);
    Vec4 sf(1.0f - *t);
    for (int i = 0; i < number; ++i) {
        Vec4 value = Vec4::load(A + 4 * i) * sf + Vec4::load(B + 4 * i) * df;
        Vec4::save(dst + 4 * i, value);
    }
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

void CPUResizeCommon::CPUResizeCubicC4(halide_buffer_t &input, halide_buffer_t &output, float xFactor, float yFactor, float wOffset, float hOffset) {
    const int batches      = input.dim[0].extent;
    const int inBatchSize  = input.dim[0].stride;
    const int outBatchSize = output.dim[0].stride;
    const int inW          = input.dim[3].extent;
    const int inH          = input.dim[2].extent;
    const int N            = input.dim[1].extent;
    const int outW         = output.dim[3].extent;
    const int outH         = output.dim[2].extent;
    const int depthQuad    = UP_DIV(N, 4);

    AutoStorage<int> linePosition(4 * outW);
    AutoStorage<float> lineFactor(outW);
    auto _linePosition = linePosition.get();
    auto _lineFactor   = lineFactor.get();

    // Compute Line Position
    for (int dx = 0; dx < outW; ++dx) {
        float x                   = (float)dx * xFactor + wOffset;
        int xInt                  = (int)x;
        _lineFactor[dx]           = (float)(x - floor(x));
        _linePosition[4 * dx + 0] = CLAMP(xInt - 1, 0, inW - 1);
        _linePosition[4 * dx + 1] = CLAMP(xInt + 0, 0, inW - 1);
        _linePosition[4 * dx + 2] = CLAMP(xInt + 1, 0, inW - 1);
        _linePosition[4 * dx + 3] = CLAMP(xInt + 2, 0, inW - 1);
    }

    for (int b = 0; b < batches; ++b) {
        MNN_CONCURRENCY_BEGIN(n, depthQuad);
        {
            int yUsed[4]  = {0, 0, 0, 0};
            int yCache[4] = {-1, -1, -1, -1};

            AutoStorage<float> lineBuffer(16 * outW);
            auto _lineBuffer              = lineBuffer.get();
            auto _line0                   = _lineBuffer + 4 * outW * 0;
            auto _line1                   = _lineBuffer + 4 * outW * 1;
            auto _line2                   = _lineBuffer + 4 * outW * 2;
            auto _line3                   = _lineBuffer + 4 * outW * 3;
            float* yCacheLine[4]          = {_line0, _line1, _line2, _line3};
            float* const yCacheStorage[4] = {_line0, _line1, _line2, _line3};
            auto bottomData = reinterpret_cast<const float*>(input.host) + b * inBatchSize + (int)n * 4 * inW * inH;
            auto topData    = reinterpret_cast<float*>(output.host) + b * outBatchSize + (int)n * 4 * outW * outH;
            for (int dy = 0; dy < outH; dy++) {
                float y  = (float)dy * yFactor + hOffset;
                int yInt = (int)y;
                int yp[4];
                yp[0] = CLAMP(yInt - 1, 0, inH - 1);
                yp[1] = CLAMP(yInt, 0, inH - 1);
                yp[2] = CLAMP(yInt + 1, 0, inH - 1);
                yp[3] = CLAMP(yInt + 2, 0, inH - 1);
                // Search cache
                for (int j = 0; j < 4; ++j) {
                    yUsed[j] = 0;
                }
                for (int j = 0; j < 4; ++j) {
                    int find = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (yp[j] == yCache[k]) {
                            yUsed[k]      = 1;
                            yCacheLine[j] = yCacheStorage[k];
                            find          = 1;
                            break;
                        }
                    }
                    if (!find) {
                        const float* bottomY0 = bottomData + yp[j] * inW * 4;
                        for (int k = 0; k < 4; ++k) {
                            if (!yUsed[k]) {
                                yCache[k]     = yp[j];
                                yUsed[k]      = 1;
                                yCacheLine[j] = yCacheStorage[k];
                                MNNCubicSampleC4(bottomY0, yCacheLine[j], _linePosition, _lineFactor, outW);
                                break;
                            }
                        }
                    }
                }

                // Sample Input
                float yFract = (float)(y - floor(y));
                auto topY    = topData + outW * 4 * dy;
                MNNCubicLineC4(topY, yCacheLine[0], yCacheLine[1], yCacheLine[2], yCacheLine[3], &yFract, outW);
            }
        }
        MNN_CONCURRENCY_END();
    }
}

void CPUResizeCommon::CPUResizeBilinearC4(halide_buffer_t& input, halide_buffer_t& output, const int* widthPosition,
                                          const float* widthFactor, const int* heightPosition,
                                          const float* heightFactor, float* lineBuffer, int threadNumber) {
    const int batches         = input.dim[0].extent;
    const int inputBatchSize  = input.dim[0].stride;
    const int outputBatchSize = output.dim[0].stride;
    const int inW             = input.dim[3].extent;
    const int inH             = input.dim[2].extent;
    const int outW            = output.dim[3].extent;
    const int outH            = output.dim[2].extent;

    int depthQuad = UP_DIV(input.dim[1].extent, 4) * batches;

    auto threadFunction = [&](size_t tId) {
        for (int n = (int)tId; n < depthQuad; n += threadNumber) {
            auto _lineBuffer = lineBuffer + 2 * 4 * outW * tId;
            auto _line0      = _lineBuffer + 4 * outW * 0;
            auto _line1      = _lineBuffer + 4 * outW * 1;
            int yUsed[2]     = {0, 0};
            int yCache[2]    = {-1, -1};

            float* yCacheLine[2]          = {_line0, _line1};
            float* const yCacheStorage[2] = {_line0, _line1};

            auto bottomData =
                reinterpret_cast<const float*>(input.host)  + (int)n * 4 * inW * inH;
            auto topData = reinterpret_cast<float*>(output.host) + (int)n * 4 * outW * outH;
            for (int dy = 0; dy < outH; dy++) {
                int yp[2];
                yp[0] = heightPosition[2 * dy + 0];
                yp[1] = heightPosition[2 * dy + 1];
                // Search cache
                for (int j = 0; j < 2; ++j) {
                    yUsed[j] = 0;
                }
                for (int j = 0; j < 2; ++j) {
                    int find = 0;
                    for (int k = 0; k < 2; ++k) {
                        if (yp[j] == yCache[k]) {
                            yUsed[k]      = 1;
                            yCacheLine[j] = yCacheStorage[k];
                            find          = 1;
                            break;
                        }
                    }
                    if (!find) {
                        const float* bottomY0 = bottomData + yp[j] * inW * 4;
                        for (int k = 0; k < 2; ++k) {
                            if (!yUsed[k]) {
                                yCache[k]     = yp[j];
                                yUsed[k]      = 1;
                                yCacheLine[j] = yCacheStorage[k];
                                CPUBilinearSampleC4(bottomY0, yCacheLine[j], widthPosition, widthFactor, outW);
                                break;
                            }
                        }
                    }
                }
                auto topY = topData + outW * 4 * dy;
                // Sample Input
                CPUBilinearLineC4(topY, yCacheLine[0], yCacheLine[1], &heightFactor[dy], outW);
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        threadFunction(tId);
    }
    MNN_CONCURRENCY_END();
}

void CPUResizeCommon::CPUResizeNearestneighborRoundC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale, float wOffset, float hOffset) {
    const int batches         = input.dim[0].extent;
    const int inputBatchSize  = input.dim[0].stride;
    const int outputBatchSize = output.dim[0].stride;
    const int inW             = input.dim[3].extent;
    const int inH             = input.dim[2].extent;
    const int outW            = output.dim[3].extent;
    const int outH            = output.dim[2].extent;
    const float xScaling      = wScale;
    const float yScaling      = hScale;
    const int depthQuad       = UP_DIV(input.dim[1].extent, 4);

    AutoStorage<int> linePosition(outW);
    auto _linePosition = linePosition.get();
    for (int x = 0; x < outW; ++x) {
        float src_x      = x * xScaling + wOffset;
        int x1           = static_cast<int>(floorf(src_x + 0.499f));
        _linePosition[x] = CLAMP(x1, 0, inW - 1);
    }

    for (int b = 0; b < batches; ++b) {
        MNN_CONCURRENCY_BEGIN(n, depthQuad) {
            auto srcData =
                reinterpret_cast<const float*>(input.host) + b * inputBatchSize + static_cast<int>(n) * 4 * inW * inH;
            auto dstData =
                reinterpret_cast<float*>(output.host) + b * outputBatchSize + static_cast<int>(n) * 4 * outW * outH;
            for (int dy = 0; dy < outH; ++dy) {
                float srcY       = dy * yScaling + hOffset;
                const int y_     = CLAMP(static_cast<int>(floorf(srcY + 0.499f)), 0, inH - 1);
                auto srcDataLine = srcData + inW * 4 * y_;
                auto dstDataLine = dstData + outW * 4 * dy;
                for (int dx = 0; dx < outW; ++dx) {
                    ::memcpy(dstDataLine + dx * 4, srcDataLine + _linePosition[dx] * 4, sizeof(float) * 4);
                }
            }
        }
        MNN_CONCURRENCY_END();
    }
}

void CPUResizeCommon::CPUResizeNearestneighborC4(halide_buffer_t& input, halide_buffer_t& output,
                                                 float wScale, float hScale, float wOffset, float hOffset) {
    const int batches         = input.dim[0].extent;
    const int inputBatchSize  = input.dim[0].stride;
    const int outputBatchSize = output.dim[0].stride;
    const int inW             = input.dim[3].extent;
    const int inH             = input.dim[2].extent;
    const int outW            = output.dim[3].extent;
    const int outH            = output.dim[2].extent;
    const float xScaling      = wScale;
    const float yScaling      = hScale;
    const int depthQuad       = UP_DIV(input.dim[1].extent, 4);

    AutoStorage<int> linePosition(outW);
    auto _linePosition = linePosition.get();
    for (int x = 0; x < outW; ++x) {
        float src_x      = x * xScaling + wOffset;
        int x1           = static_cast<int>(floor(src_x));
        _linePosition[x] = CLAMP(x1, 0, inW - 1);
    }

    for (int b = 0; b < batches; ++b) {
        MNN_CONCURRENCY_BEGIN(n, depthQuad) {
            auto srcData =
                reinterpret_cast<const float*>(input.host) + b * inputBatchSize + static_cast<int>(n) * 4 * inW * inH;
            auto dstData =
                reinterpret_cast<float*>(output.host) + b * outputBatchSize + static_cast<int>(n) * 4 * outW * outH;
            for (int dy = 0; dy < outH; ++dy) {
                float srcY       = dy * yScaling + hOffset;
                const int y_     = CLAMP(static_cast<int>(floor(srcY)), 0, inH - 1);
                auto srcDataLine = srcData + inW * 4 * y_;
                auto dstDataLine = dstData + outW * 4 * dy;
                for (int dx = 0; dx < outW; ++dx) {
                    ::memcpy(dstDataLine + dx * 4, srcDataLine + _linePosition[dx] * 4, sizeof(float) * 4);
                }
            }
        }
        MNN_CONCURRENCY_END();
    }
}

void CPUResizeCommon::CPUResizeNearestneighbor3DRoundC4(halide_buffer_t &input, halide_buffer_t &output,
                                                        float wScale, float hScale, float dScale,
                                                        float wOffset, float hOffset, float dOffset) {
    const int batches         = input.dim[0].extent;
    const int inputBatchSize  = input.dim[0].stride;
    const int outputBatchSize = output.dim[0].stride;
    const int inW             = input.dim[4].extent;
    const int inH             = input.dim[3].extent;
    const int inD             = input.dim[2].extent;
    const int outW            = output.dim[4].extent;
    const int outH            = output.dim[3].extent;
    const int outD            = output.dim[2].extent;
    const float xScaling      = wScale;
    const float yScaling      = hScale;
    const float zScaling      = dScale;
    const int depthQuad       = UP_DIV(input.dim[1].extent, 4);

    AutoStorage<int> linePosition(outW);
    auto _linePosition = linePosition.get();
    for (int x = 0; x < outW; ++x) {
        float src_x      = x * xScaling + wOffset;
        int x1           = static_cast<int>(floorf(src_x + 0.499f));
        _linePosition[x] = CLAMP(x1, 0, inW - 1);
    }

    AutoStorage<int> columnPosition(outH);
    auto _columnPosition = columnPosition.get();
    for (int y = 0; y < outH; ++y) {
        float src_y      = y * yScaling + hOffset;
        int y1           = static_cast<int>(floorf(src_y + 0.499f));
        _columnPosition[y] = CLAMP(y1, 0, inH - 1);
    }

    for (int b = 0; b < batches; ++b) {
        MNN_CONCURRENCY_BEGIN(n, depthQuad) {
            auto srcData = reinterpret_cast<const float*>(input.host)
                    + b * inputBatchSize + static_cast<int>(n) * 4 * inW * inH * inD;
            auto dstData = reinterpret_cast<float*>(output.host)
                    + b * outputBatchSize + static_cast<int>(n) * 4 * outW * outH * inD;
            for (int dz = 0; dz < outD; ++dz) {
                float srcZ       = dz * zScaling + dOffset;
                const int z_     = CLAMP(static_cast<int>(floorf(srcZ + 0.499f)), 0, inD - 1);
                auto srcDataArea = srcData + inH * inW * 4 * z_;
                auto dstDataArea = dstData + outH * outW * 4 * dz;
                for (int dy = 0; dy < outH; ++dy) {
                    auto srcDataLine = srcDataArea + inW * 4 * _columnPosition[dy];
                    auto dstDataLine = dstDataArea + outW * 4 * dy;
                    for (int dx = 0; dx < outW; ++dx) {
                        ::memcpy(dstDataLine + dx * 4, srcDataLine + _linePosition[dx] * 4, sizeof(float) * 4);
                    }
                }
            }

        }
        MNN_CONCURRENCY_END();
    }
}

void CPUResizeCommon::CPUResizeNearestneighbor3DC4(halide_buffer_t& input, halide_buffer_t& output,
                                                 float wScale, float hScale, float dScale,
                                                 float wOffset, float hOffset, float dOffset) {
    const int batches         = input.dim[0].extent;
    const int inputBatchSize  = input.dim[0].stride;
    const int outputBatchSize = output.dim[0].stride;
    const int inW             = input.dim[4].extent;
    const int inH             = input.dim[3].extent;
    const int inD             = input.dim[2].extent;
    const int outW            = output.dim[4].extent;
    const int outH            = output.dim[3].extent;
    const int outD            = output.dim[2].extent;
    const float xScaling      = wScale;
    const float yScaling      = hScale;
    const float zScaling      = dScale;
    const int depthQuad       = UP_DIV(input.dim[1].extent, 4);

    AutoStorage<int> linePosition(outW);
    auto _linePosition = linePosition.get();
    for (int x = 0; x < outW; ++x) {
        float src_x      = x * xScaling + wOffset;
        int x1           = static_cast<int>(floor(src_x));
        _linePosition[x] = CLAMP(x1, 0, inW - 1);
    }

    AutoStorage<int> columnPosition(outH);
    auto _columnPosition = columnPosition.get();
    for (int y = 0; y < outH; ++y) {
        float src_y      = y * yScaling + hOffset;
        int y1           = static_cast<int>(floor(src_y));
        _columnPosition[y] = CLAMP(y1, 0, inH - 1);
    }

    for (int b = 0; b < batches; ++b) {
        MNN_CONCURRENCY_BEGIN(n, depthQuad) {
            auto srcData = reinterpret_cast<const float*>(input.host)
                    + b * inputBatchSize + static_cast<int>(n) * 4 * inW * inH * inD;
            auto dstData = reinterpret_cast<float*>(output.host)
                    + b * outputBatchSize + static_cast<int>(n) * 4 * outW * outH * outD;
            for (int dz = 0; dz < outD; ++dz){
                float srcZ       = dz * zScaling + dOffset;
                const int z_     = CLAMP(static_cast<int>(floor(srcZ)), 0, inD - 1);
                auto srcDataArea = srcData + inH * inW * 4 * z_;
                auto dstDataArea = dstData + outH * outW * 4 * dz;
                for (int dy = 0; dy < outH; ++dy) {
                    auto srcDataLine = srcDataArea + _columnPosition[dy] * inW * 4;
                    auto dstDataLine = dstDataArea + dy * outW * 4;
                    for (int dx = 0; dx < outW; ++dx) {
                        ::memcpy(dstDataLine + dx * 4, srcDataLine + _linePosition[dx] * 4, sizeof(float) * 4);
                    }
                }
            }

        }
        MNN_CONCURRENCY_END();
    }
}

} // namespace MNN
