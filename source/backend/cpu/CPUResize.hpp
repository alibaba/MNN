//
//  CPUResize.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUResize_hpp
#define CPUResize_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include "core/Macro.h"
#include <math.h>

using Vec4 = MNN::Math::Vec<float, 4>;
#ifdef __cplusplus
extern "C" {
#endif
void CPUBilinearSampleC4(const float* src, float* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void CPUBilinearLineC4(float* dst, const float* A, const float* B, const float* t, int8_t* zeroPoint, size_t number);
void MNNBilinearSampleC8(const int8_t* src, int16_t* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void MNNBilinearLineC8(int8_t* dst, const int16_t* A, const int16_t* B, const float* t, int8_t* zeroPoint, size_t number);
void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
                    size_t number, ssize_t minValue, ssize_t maxValue);
void MNNCubicSampleC16(const int8_t* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void MNNCubicLineC16(int8_t* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
                     size_t number, ssize_t minValue, ssize_t maxValue);
#ifdef __cplusplus
}
#endif

namespace MNN {
static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}
class CPUResizeCommon : public Execution {
public:
    CPUResizeCommon(Backend *backend) : Execution(backend) {
        // Do nothing
    }
    virtual ~CPUResizeCommon()                                                                             = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)  = 0;

    template<typename T, typename U>
    void CPUResizeBilinearC4(void sampleFunction(const T*, U*, const int32_t*, const float*, int8_t*, size_t), void lineFunction(T*, const U*, const U*, const float*, int8_t*, size_t), const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const int* widthPosition, const float* widthFactor, const int* heightPosition,
        const float* heightFactor, U* lineBuffer, int threadNumber, int8_t* inputQuantZero, int8_t* outputQuantZero) {
        auto input = inputs[0];
        auto output = outputs[0];
        const int batches         = input->batch();
        const int inW             = input->width();
        const int inH             = input->height();
        const int outW            = output->width();
        const int outH            = output->height();
        int pack = 4;
        if(sizeof(T) == 1) {
            pack = 8;
        }
        int depthQuad = UP_DIV(input->channel(), pack) * batches;
        auto threadFunction = [&](size_t tId) {
            for (int n = (int)tId; n < depthQuad; n += threadNumber) {
                U* _lineBuffer = lineBuffer + 2 * pack * outW * tId;
                U* _line0      = _lineBuffer + pack * outW * 0;
                U* _line1      = _lineBuffer + pack * outW * 1;
                int yUsed[2]     = {0, 0};
                int yCache[2]    = {-1, -1};

                U* yCacheLine[2]          = {_line0, _line1};
                U* const yCacheStorage[2] = {_line0, _line1};

                const T* bottomData = reinterpret_cast<const T*>(input->host<uint8_t>())  + (int)n * pack * inW * inH;
                T* topData = reinterpret_cast<T*>(output->host<uint8_t>()) + (int)n * pack * outW * outH;
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
                            const T* bottomY0 = bottomData + yp[j] * inW * pack;
                            for (int k = 0; k < 2; ++k) {
                                if (!yUsed[k]) {
                                    yCache[k]     = yp[j];
                                    yUsed[k]      = 1;
                                    yCacheLine[j] = yCacheStorage[k];
                                    sampleFunction(bottomY0, yCacheLine[j], widthPosition, widthFactor, inputQuantZero, outW);
                                    break;
                                }
                            }
                        }
                    }
                    T* topY = topData + outW * pack * dy;
                    // Sample Input
                    lineFunction(topY, yCacheLine[0], yCacheLine[1], &heightFactor[dy], outputQuantZero, outW);
                    
                }
            }
        };
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            threadFunction(tId);
        }
        MNN_CONCURRENCY_END();
    }

    template<typename T>
    void CPUResizeCubicC4(void sampleFunction(const T*, float*, int32_t*, const float*, int8_t*, size_t), void lineFunction(T*, const float*, const float*, const float*, const float*, float*, int8_t*, size_t, ssize_t, ssize_t),
                          const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, float xFactor, float yFactor, float wOffset, float hOffset, int8_t* inputQuantZero, int8_t* outputQuantZero, ssize_t minValue, ssize_t maxValue) {
        auto input = inputs[0];
        auto output = outputs[0];
        const int batches      = input->batch();
        const int inBatchSize  = input->stride(0);
        const int outBatchSize = output->stride(0);
        const int inW          = input->width();
        const int inH          = input->height();
        const int N            = input->channel();
        const int outW         = output->width();
        const int outH         = output->height();
        int pack = 16/sizeof(T);
        const int depthQuad    = UP_DIV(N, pack);

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

                AutoStorage<float> lineBuffer(4 * pack * outW);
                auto _lineBuffer              = lineBuffer.get();
                auto _line0                   = _lineBuffer + pack * outW * 0;
                auto _line1                   = _lineBuffer + pack * outW * 1;
                auto _line2                   = _lineBuffer + pack * outW * 2;
                auto _line3                   = _lineBuffer + pack * outW * 3;
                float* yCacheLine[4]          = {_line0, _line1, _line2, _line3};
                float* const yCacheStorage[4] = {_line0, _line1, _line2, _line3};
                auto bottomData = reinterpret_cast<const T*>(input->host<uint8_t>()) + b * inBatchSize + (int)n * pack * inW * inH;
                auto topData    = reinterpret_cast<T*>(output->host<uint8_t>()) + b * outBatchSize + (int)n * pack * outW * outH;
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
                            const T* bottomY0 = bottomData + yp[j] * inW * pack;
                            for (int k = 0; k < 4; ++k) {
                                if (!yUsed[k]) {
                                    yCache[k]     = yp[j];
                                    yUsed[k]      = 1;
                                    yCacheLine[j] = yCacheStorage[k];
                                    sampleFunction(bottomY0, yCacheLine[j], _linePosition, _lineFactor, inputQuantZero, outW);
                                    break;
                                }
                            }
                        }
                    }

                    // Sample Input
                    float yFract = (float)(y - floor(y));
                    auto topY    = topData + outW * pack * dy;
                    lineFunction(topY, yCacheLine[0], yCacheLine[1], yCacheLine[2], yCacheLine[3], &yFract, outputQuantZero, outW, minValue, maxValue);
                }
            }
            MNN_CONCURRENCY_END();
        }
    }

    template<typename T>
    void CPUResizeNearestneighborRoundC4(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, float wScale, float hScale, float wOffset, float hOffset) {
        auto input = inputs[0];
        auto output = outputs[0];
        const int batches         = input->batch();
        const int inputBatchSize  = input->stride(0);
        const int outputBatchSize = output->stride(0);
        const int inW             = input->width();
        const int inH             = input->height();
        const int outW            = output->width();
        const int outH            = output->height();
        const float xScaling      = wScale;
        const float yScaling      = hScale;
        int pack = 16/sizeof(T);
        const int depthQuad       = UP_DIV(input->channel(), pack);

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
                    reinterpret_cast<const T*>(input->host<uint8_t>()) + b * inputBatchSize + static_cast<int>(n) * pack * inW * inH;
                auto dstData =
                    reinterpret_cast<T*>(output->host<uint8_t>()) + b * outputBatchSize + static_cast<int>(n) * pack * outW * outH;
                for (int dy = 0; dy < outH; ++dy) {
                    float srcY       = dy * yScaling + hOffset;
                    const int y_     = CLAMP(static_cast<int>(floorf(srcY + 0.499f)), 0, inH - 1);
                    auto srcDataLine = srcData + inW * pack * y_;
                    auto dstDataLine = dstData + outW * pack * dy;
                    for (int dx = 0; dx < outW; ++dx) {
                        ::memcpy(dstDataLine + dx * pack, srcDataLine + _linePosition[dx] * pack, sizeof(T) * pack);
                    }
                }
            }
            MNN_CONCURRENCY_END();
        }
    }
    
    template<typename T>
    void CPUResizeNearestneighborC4(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                    float wScale, float hScale, float wOffset, float hOffset) {
        auto input = inputs[0];
        auto output = outputs[0];
        const int batches         = input->batch();
        const int inputBatchSize  = input->stride(0);
        const int outputBatchSize = output->stride(0);
        const int inW             = input->width();
        const int inH             = input->height();
        const int outW            = output->width();
        const int outH            = output->height();
        const float xScaling      = wScale;
        const float yScaling      = hScale;
        int pack = 4;
        if (sizeof(T) == 1) {
            pack = 8;
        }
        const int depthQuad       = UP_DIV(input->channel(), pack);

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
                    reinterpret_cast<const T*>(input->host<uint8_t>()) + b * inputBatchSize + static_cast<int>(n) * pack * inW * inH;
                auto dstData =
                    reinterpret_cast<T*>(output->host<uint8_t>()) + b * outputBatchSize + static_cast<int>(n) * pack * outW * outH;
                for (int dy = 0; dy < outH; ++dy) {
                    float srcY       = dy * yScaling + hOffset;
                    const int y_     = CLAMP(static_cast<int>(floor(srcY)), 0, inH - 1);
                    auto srcDataLine = srcData + inW * pack * y_;
                    auto dstDataLine = dstData + outW * pack * dy;
                    for (int dx = 0; dx < outW; ++dx) {
                        ::memcpy(dstDataLine + dx * pack, srcDataLine + _linePosition[dx] * pack, sizeof(T) * pack);
                    }
                }
            }
            MNN_CONCURRENCY_END();
        }
    }
    
    template<typename T>
    void CPUResizeNearestneighbor3DRoundC4(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                                            float wScale, float hScale, float dScale,
                                                            float wOffset, float hOffset, float dOffset) {
        auto input = inputs[0];
        auto output = outputs[0];
        
        const int batches         = input->buffer().dim[0].extent;
        const int inputBatchSize  = input->buffer().dim[0].stride;
        const int outputBatchSize = output->buffer().dim[0].stride;
        const int inW             = input->buffer().dim[4].extent;
        const int inH             = input->buffer().dim[3].extent;
        const int inD             = input->buffer().dim[2].extent;
        const int outW            = output->buffer().dim[4].extent;
        const int outH            = output->buffer().dim[3].extent;
        const int outD            = output->buffer().dim[2].extent;
        const float xScaling      = wScale;
        const float yScaling      = hScale;
        const float zScaling      = dScale;
        int pack = 16 / sizeof(T);
        const int depthQuad       = UP_DIV(input->buffer().dim[1].extent, pack);

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
                auto srcData = reinterpret_cast<const T*>(input->host<uint8_t>())
                        + b * inputBatchSize + static_cast<int>(n) * pack * inW * inH * inD;
                auto dstData = reinterpret_cast<T*>(output->host<uint8_t>())
                        + b * outputBatchSize + static_cast<int>(n) * pack * outW * outH * inD;
                for (int dz = 0; dz < outD; ++dz) {
                    float srcZ       = dz * zScaling + dOffset;
                    const int z_     = CLAMP(static_cast<int>(floorf(srcZ + 0.499f)), 0, inD - 1);
                    auto srcDataArea = srcData + inH * inW * pack * z_;
                    auto dstDataArea = dstData + outH * outW * pack * dz;
                    for (int dy = 0; dy < outH; ++dy) {
                        auto srcDataLine = srcDataArea + inW * pack * _columnPosition[dy];
                        auto dstDataLine = dstDataArea + outW * pack * dy;
                        for (int dx = 0; dx < outW; ++dx) {
                            ::memcpy(dstDataLine + dx * pack, srcDataLine + _linePosition[dx] * pack, sizeof(T) * pack);
                        }
                    }
                }

            }
            MNN_CONCURRENCY_END();
        }
    }
    
    template<typename T>
    void CPUResizeNearestneighbor3DC4(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                     float wScale, float hScale, float dScale,
                                     float wOffset, float hOffset, float dOffset) {
        auto input = inputs[0];
        auto output = outputs[0];
        const int batches         = input->buffer().dim[0].extent;
        const int inputBatchSize  = input->buffer().dim[0].stride;
        const int outputBatchSize = output->buffer().dim[0].stride;
        const int inW             = input->buffer().dim[4].extent;
        const int inH             = input->buffer().dim[3].extent;
        const int inD             = input->buffer().dim[2].extent;
        const int outW            = output->buffer().dim[4].extent;
        const int outH            = output->buffer().dim[3].extent;
        const int outD            = output->buffer().dim[2].extent;
        const float xScaling      = wScale;
        const float yScaling      = hScale;
        const float zScaling      = dScale;
        int pack = 16 / sizeof(T);
        const int depthQuad       = UP_DIV(input->buffer().dim[1].extent, pack);

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
                auto srcData = reinterpret_cast<const T*>(input->host<uint8_t>())
                        + b * inputBatchSize + static_cast<int>(n) * pack * inW * inH * inD;
                auto dstData = reinterpret_cast<T*>(output->host<uint8_t>())
                        + b * outputBatchSize + static_cast<int>(n) * pack * outW * outH * outD;
                for (int dz = 0; dz < outD; ++dz){
                    float srcZ       = dz * zScaling + dOffset;
                    const int z_     = CLAMP(static_cast<int>(floor(srcZ)), 0, inD - 1);
                    auto srcDataArea = srcData + inH * inW * pack * z_;
                    auto dstDataArea = dstData + outH * outW * pack * dz;
                    for (int dy = 0; dy < outH; ++dy) {
                        auto srcDataLine = srcDataArea + _columnPosition[dy] * inW * pack;
                        auto dstDataLine = dstDataArea + dy * outW * pack;
                        for (int dx = 0; dx < outW; ++dx) {
                            ::memcpy(dstDataLine + dx * pack, srcDataLine + _linePosition[dx] * pack, sizeof(T) * pack);
                        }
                    }
                }

            }
            MNN_CONCURRENCY_END();
        }
    }

};
} // namespace MNN

#endif /* CPUResize_hpp */
