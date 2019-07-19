//
//  CPUResize.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUResize.hpp"
#include <math.h>
#include "AutoStorage.h"
#include "CPUBackend.hpp"
#include "Concurrency.h"
#include "Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

extern "C" {
void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, size_t number);
void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t,
                    size_t number);
}

namespace MNN {

static void CPUBilinearSampleC4(const float* src, float* dst, const int32_t* position, const float* factor,
                                size_t number) {
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
#ifdef MNN_USE_NEON
        float32x4_t df = vdupq_n_f32(f);
        float32x4_t sf = vdupq_n_f32(1.0f - f);
        float32x4_t A  = vld1q_f32(src + position[2 * i] * 4);
        float32x4_t B  = vld1q_f32(src + position[2 * i + 1] * 4);
        vst1q_f32(dst + 4 * i, B * df + A * sf);
#else
        for (int k = 0; k < 4; ++k) {
            float A        = src[4 * position[2 * i + 0] + k];
            float B        = src[4 * position[2 * i + 1] + k];
            dst[4 * i + k] = B * f + A * (1 - f);
        }
#endif
    }
}

static void CPUBilinearLineC4(float* dst, const float* A, const float* B, const float* t, size_t number) {
#ifdef MNN_USE_NEON
    float32x4_t df = vdupq_n_f32(*t);
    float32x4_t sf = vdupq_n_f32(1.0f) - df;
    for (int i = 0; i < number; ++i) {
        float32x4_t value = vld1q_f32(A + 4 * i) * sf + vld1q_f32(B + 4 * i) * df;
        vst1q_f32(dst + 4 * i, value);
    }
#else
    float f = *t;
    for (int i = 0; i < number; ++i) {
        for (int j = 0; j < 4; ++j) {
            int k = i * 4 + j;
            dst[k] = A[k] * (1 - f) + B[k] * f;
        }
    }
#endif
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

void CPUResizeCommon::CPUResizeCubicC4(halide_buffer_t& input, halide_buffer_t& output) {
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
        float u                   = ((float)dx) / ((float)(outW - 1));
        float x                   = u * inW - 0.5f;
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
                float v  = ((float)dy) / ((float)(outH - 1));
                float y  = v * inH - 0.5f;
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

    int depthQuad = UP_DIV(input.dim[1].extent, 4);

    for (int b = 0; b < batches; ++b) {
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
                    reinterpret_cast<const float*>(input.host) + b * inputBatchSize + (int)n * 4 * inW * inH;
                auto topData = reinterpret_cast<float*>(output.host) + b * outputBatchSize + (int)n * 4 * outW * outH;
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
}

void CPUResizeCommon::CPUReiseNearstneighborC4(halide_buffer_t& input, halide_buffer_t& output, float wScale,
                                               float hScale) {
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
        float src_x      = x * xScaling;
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
                float srcY       = dy * yScaling;
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

CPUResize::CPUResize(Backend* backend, float xScale, float yScale)
    : CPUResizeCommon(backend), mXScale(xScale), mYScale(yScale) {
    // nothing to do
}

CPUResize::~CPUResize() {
}

ErrorCode CPUResize::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const int inW        = inputs[0]->buffer().dim[3].extent;
    const int inH        = inputs[0]->buffer().dim[2].extent;
    const int outW       = outputs[0]->buffer().dim[3].extent;
    const int outH       = outputs[0]->buffer().dim[2].extent;
    const float xScaling = 1.0f / mXScale;
    const float yScaling = 1.0f / mYScale;

    mWidthPosition.buffer().dim[0].extent = 2 * outW;
    mWidthPosition.buffer().dimensions    = 1;
    mWidthPosition.setType(DataType_DT_INT32);
    backend()->onAcquireBuffer(&mWidthPosition, Backend::DYNAMIC_SEPERATE);

    mWidthFactor.buffer().dim[0].extent = outW;
    mWidthFactor.buffer().dimensions    = 1;
    mWidthFactor.setType(DataType_DT_FLOAT);
    backend()->onAcquireBuffer(&mWidthFactor, Backend::DYNAMIC_SEPERATE);

    auto _wPosition = mWidthPosition.host<int>();
    auto _wFactor   = mWidthFactor.host<float>();

    // Compute Line Position
    for (int x = 0; x < outW; ++x) {
        float srcX     = x * xScaling;
        int x1         = floor(srcX);
        float x2Factor = srcX - x1;

        _wFactor[x]           = x2Factor;
        _wPosition[2 * x + 0] = CLAMP(x1, 0, inW - 1);
        _wPosition[2 * x + 1] = CLAMP(x1 + 1, 0, inW - 1);
    }

    mHeightPosition.buffer().dim[0].extent = 2 * outH;
    mHeightPosition.buffer().dimensions    = 1;
    mHeightPosition.setType(DataType_DT_INT32);
    backend()->onAcquireBuffer(&mHeightPosition, Backend::DYNAMIC_SEPERATE);

    mHeightFactor.buffer().dim[0].extent = outH;
    mHeightFactor.buffer().dimensions    = 1;
    mHeightFactor.setType(DataType_DT_FLOAT);
    backend()->onAcquireBuffer(&mHeightFactor, Backend::DYNAMIC_SEPERATE);

    auto _hPosition = mHeightPosition.host<int>();
    auto _hFactor   = mHeightFactor.host<float>();

    for (int y = 0; y < outH; ++y) {
        float srcY     = y * yScaling;
        int y1         = floor(srcY);
        float y2Factor = srcY - y1;

        _hFactor[y]           = y2Factor;
        _hPosition[2 * y + 0] = CLAMP(y1, 0, inH - 1);
        _hPosition[2 * y + 1] = CLAMP(y1 + 1, 0, inH - 1);
    }

    int threadNumber = ((CPUBackend*)backend())->threadNumber();

    mLineBuffer.buffer().dim[0].extent = 2 * 4 * outW * threadNumber;
    mLineBuffer.buffer().dimensions    = 1;
    mLineBuffer.setType(DataType_DT_FLOAT);
    backend()->onAcquireBuffer(&mLineBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mLineBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUResize::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& input  = inputs[0]->buffer();
    auto& output = outputs[0]->buffer();
    CPUResizeBilinearC4(input, output, mWidthPosition.host<int>(), mWidthFactor.host<float>(),
                        mHeightPosition.host<int>(), mHeightFactor.host<float>(), mLineBuffer.host<float>(),
                        ((CPUBackend*)backend())->threadNumber());
    return NO_ERROR;
}

class CPUResizeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto resize = op->main_as_Resize();
        return new CPUResize(backend, resize->xScale(), resize->yScale());
    }
};
REGISTER_CPU_OP_CREATOR(CPUResizeCreator, OpType_Resize);
} // namespace MNN
