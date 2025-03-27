//
//  CPUTexture.cpp
//  MNN
//
//  Created by MNN on 2023/06/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../CPUBackend.hpp"
#include "../compute/CommonOptFunction.h"

namespace MNN {
#ifdef MNN_SUPPORT_RENDER

template<bool grad>
void Execute_2D(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, SampleMode mMode, BorderMode mPadMode, Backend* backend) {
    Tensor* inputTensor;
    Tensor* gridTensor;
    Tensor* outputTensor;
    if (grad) {
        inputTensor = outputs[0];
        gridTensor = inputs[1];
        outputTensor = inputs[0];
    } else {
        inputTensor = inputs[0];
        gridTensor = inputs[1];
        outputTensor = outputs[0];
    }
    auto inputPtr = inputTensor->host<uint8_t>();
    auto gridPtr = gridTensor->host<uint8_t>();
    auto outputPtr = outputTensor->host<uint8_t>();
    auto core = static_cast<CPUBackend*>(backend)->functions();
    auto bytes = core->bytes;
    auto batches = inputTensor->length(0);
    auto unit = inputTensor->length(3);
    auto ih = inputTensor->length(1);
    auto iw = inputTensor->length(2);
    auto oh = outputTensor->length(1);
    auto ow = outputTensor->length(2);
    MNN_ASSERT(batches == 1);
    MNN_ASSERT(bytes == 4);
    auto srcFloat = (float*)inputPtr;
    if (grad) {
        ::memset(inputPtr, 0, ih * iw * batches * unit * bytes);
    }
    for (int y=0; y<oh; ++y) {
        auto dstY = outputPtr + y * bytes * unit * ow;
        auto gridY = gridPtr + y * bytes * ow * 2;
        for (int x=0; x<ow; ++x) {
            auto dstX = (float*)(dstY + x * bytes * unit);
            auto gridX = (const float*)(gridY + x * bytes * 2);
            float u = gridX[0];
            float v = gridX[1];
            if (mMode == SampleMode_NEAREST) {
                u = u * (float)(iw);
                v = v * (float)(ih);
                int ui = (int)floorf(u);
                int vi = (int)floorf(v);
                if (mPadMode == BorderMode_ZEROS) {
                    if (ui < 0 || ui >= iw || vi < 0 || vi >= ih) {
                        if (!grad) {
                            for (int c=0; c<unit; ++c) {
                                dstX[c] = 0.0f;
                            }
                        }
                        continue;
                    }
                }
                ui = fminf(fmaxf(ui, 0), iw-1);
                vi = fminf(fmaxf(vi, 0), ih-1);

                auto srcX = srcFloat + (ui + vi * iw) * unit;
                if (grad) {
                    for (int c=0; c<unit; ++c) {
                        srcX[c] += dstX[c];
                    }
                } else {
                    for (int c=0; c<unit; ++c) {
                        dstX[c] = srcX[c];
                    }
                }
            } else {
                u = u * (float)iw - 0.5f;
                v = v * (float)ih - 0.5f;
                if (mPadMode == BorderMode_CLAMP) {
                    // Clamp to center of edge texels.
                    u = fminf(fmaxf(u, 0.f), iw - 1.f);
                    v = fminf(fmaxf(v, 0.f), ih - 1.f);
                }
                MNN_ASSERT(mMode == SampleMode_BILINEAR);
                int xs = (int)floorf(u);
                int xe = (int)ceilf(u);
                int ys = (int)floorf(v);
                int ye = (int)ceilf(v);
                float xef = u - xs;
                float xsf = 1.0f - xef;
                float yef = v - ys;
                float ysf = 1.0f - yef;
                int index[4];
                index[0] = xs + ys * iw;
                index[1] = xe + ys * iw;
                index[2] = xs + ye * iw;
                index[3] = xe + ye * iw;
                if (mPadMode == BorderMode_ZEROS) {
                    if (xs < 0 || xs >= iw) {
                        index[0] = -1;
                        index[2] = -1;
                    }
                    if (xe < 0 || xe >= iw) {
                        index[1] = -1;
                        index[3] = -1;
                    }
                    if (ys < 0 || ys >= ih) {
                        index[0] = -1;
                        index[1] = -1;
                    }
                    if (ye < 0 || ye >= ih) {
                        index[2] = -1;
                        index[3] = -1;
                    }
                }
                auto s00 = srcFloat + (xs + ys * iw) * unit;
                auto s01 = srcFloat + (xe + ys * iw) * unit;
                auto s10 = srcFloat + (xs + ye * iw) * unit;
                auto s11 = srcFloat + (xe + ye * iw) * unit;
                float* s[4] = {s00, s01, s10, s11};
                float f00 = xsf * ysf;
                float f01 = xef * ysf;
                float f10 = xsf * yef;
                float f11 = xef * yef;
                float f[4] = {f00, f01, f10, f11};
                if (!grad) {
                    for (int c=0; c<unit; ++c) {
                        dstX[c] = 0.0f;
                    }
                }
                for (int v=0; v<4; ++v) {
                    if (index[v] >= 0) {
                        if (grad) {
                            for (int c=0; c<unit; ++c) {
                                s[v][c] += f[v] * dstX[c];
                            }
                        } else {
                            for (int c=0; c<unit; ++c) {
                                dstX[c] += f[v] * s[v][c];
                            }
                        }
                    }
                }
            }
        }
    }
}

class CPUTexture2D : public Execution {
public:
    CPUTexture2D(Backend *backend, const Op* op) : Execution(backend) {
        mMode = op->main_as_GridSample()->mode();
        mPadMode = op->main_as_GridSample()->paddingMode();
        MNN_ASSERT(mPadMode != BorderMode_CUBE);
    }
    virtual ~CPUTexture2D() {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        Execute_2D<false>(inputs, outputs, mMode, mPadMode, backend());
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }

protected:
    SampleMode mMode;
    BorderMode mPadMode;
    
};

class CPUTexture2DGrad : public CPUTexture2D {
public:
    CPUTexture2DGrad(Backend *backend, const Op* op) : CPUTexture2D(backend, op) {
        // Do nothing
    }
    virtual ~CPUTexture2DGrad() {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        Execute_2D<true>(inputs, outputs, mMode, mPadMode, backend());
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }
};
static uint32_t __float_as_uint(float u) {
    float* ptr = &u;
    uint32_t* ptru = (uint32_t*)ptr;
    return *ptru;
}
static float __uint_as_float(uint32_t u) {
    uint32_t* ptru = &u;
    float* ptr = (float*)ptru;
    return *ptr;
}
static int indexCubeMap(float& x, float& y, float z)
{
    float ax = fabsf(x);
    float ay = fabsf(y);
    float az = fabsf(z);
    int idx;
    float c;
    if (az > fmaxf(ax, ay)) { idx = 4; c = z; }
    else if (ay > ax)       { idx = 2; c = y; y = z; }
    else                    { idx = 0; c = x; x = z; }
    if (c < 0.f) idx += 1;
    float m = 1.0f / (fabsf(c)) * .5;
    float m0 = __uint_as_float(__float_as_uint(m) ^ ((0x21u >> idx) << 31));
    float m1 = (idx != 2) ? -m : m;
    x = x * m0 + .5;
    y = y * m1 + .5;
    if (!isfinite(x) || !isfinite(y))
        return -1; // Invalid uv.
    x = fminf(fmaxf(x, 0.f), 1.f);
    y = fminf(fmaxf(y, 0.f), 1.f);
    return idx;
}
template<bool grad>
void Execute_Cube(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, SampleMode mMode, Backend* backend) {
    Tensor* inputTensor;
    Tensor* gridTensor;
    Tensor* outputTensor;
    if (grad) {
        inputTensor = outputs[0];
        gridTensor = inputs[1];
        outputTensor = inputs[0];
    } else {
        inputTensor = inputs[0];
        gridTensor = inputs[1];
        outputTensor = outputs[0];
    }
    auto inputPtr = inputTensor->host<uint8_t>();
    auto gridPtr = gridTensor->host<uint8_t>();
    auto outputPtr = outputTensor->host<uint8_t>();
    auto core = static_cast<CPUBackend*>(backend)->functions();
    auto bytes = core->bytes;
    auto batches = inputTensor->length(0);
    auto unit = outputTensor->length(3);
    MNN_ASSERT(6 == inputTensor->length(1));
    auto ih = inputTensor->length(2);
    auto iw = inputTensor->length(3);
    auto oh = outputTensor->length(1);
    auto ow = outputTensor->length(2);
    MNN_ASSERT(batches == 1);
    MNN_ASSERT(bytes == 4);
    MNN_ASSERT(6 == inputTensor->length(1));
    auto srcFloatCube = (float*)inputPtr;
    const int cordUnit = 3;
    if (grad) {
        ::memset(inputPtr, 0, ih * iw * batches * unit * bytes * 6);
    }
    for (int y=0; y<oh; ++y) {
        auto dstY = outputPtr + y * bytes * unit * ow;
        auto gridY = gridPtr + y * bytes * ow * cordUnit;
        for (int x=0; x<ow; ++x) {
            auto dstX = (float*)(dstY + x * bytes * unit);
            auto gridX = (const float*)(gridY + x * bytes * cordUnit);
            float u = gridX[0];
            float v = gridX[1];
            float z = gridX[2];
            auto index = indexCubeMap(u, v, z);
            if (index < 0 || (u == 0.0f && v == 0.0f)) {
                if (!grad) {
                    for (int c=0; c<unit; ++c) {
                        dstX[c] = 0.0f;
                    }
                }
                continue;
            }
            auto srcFloat = srcFloatCube + index * iw * ih * unit;
            if (mMode == SampleMode_NEAREST) {
                u = u * (float)(iw);
                v = v * (float)(ih);
                int ui = (int)floorf(u);
                int vi = (int)floorf(v);
                auto srcX = srcFloat + (ui + vi * iw) * unit;
                if (grad) {
                    for (int c=0; c<unit; ++c) {
                        srcX[c] += dstX[c];
                    }
                } else {
                    for (int c=0; c<unit; ++c) {
                        dstX[c] = srcX[c];
                    }
                }
            } else {
                MNN_ASSERT(mMode == SampleMode_BILINEAR);
                u = u * (float)(iw) - 0.5f;
                v = v * (float)(ih) - 0.5f;
                u = fminf(u, iw-1);
                u = fmaxf(u, 0.0f);
                v = fminf(v, ih-1);
                v = fmaxf(v, 0.0f);
                int xs = (int)floorf(u);
                int xe = (int)ceilf(u);
                int ys = (int)floorf(v);
                int ye = (int)ceilf(v);
                float xef = u - xs;
                float xsf = 1.0f - xef;
                float yef = v - ys;
                float ysf = 1.0f - yef;
                auto s00 = srcFloat + (xs + ys * iw) * unit;
                auto s01 = srcFloat + (xe + ys * iw) * unit;
                auto s10 = srcFloat + (xs + ye * iw) * unit;
                auto s11 = srcFloat + (xe + ye * iw) * unit;
                float f00 = xsf * ysf;
                float f01 = xef * ysf;
                float f10 = xsf * yef;
                float f11 = xef * yef;
                if (grad) {
                    for (int c=0; c<unit; ++c) {
                        s00[c] += f00 * dstX[c];
                        s01[c] += f01 * dstX[c];
                        s10[c] += f10 * dstX[c];
                        s11[c] += f11 * dstX[c];
                    }
                } else {
                    for (int c=0; c<unit; ++c) {
                        dstX[c] = f00 * s00[c] + f01 * s01[c] + f10 * s10[c] + f11 * s11[c];
                    }
                }
            }
        }
    }
}

class CPUTextureCube : public Execution {
public:
    CPUTextureCube(Backend *backend, const Op* op) : Execution(backend) {
        mMode = op->main_as_GridSample()->mode();
        mGrad = op->main_as_GridSample()->backward();
    }
    virtual ~CPUTextureCube() {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        if (mGrad) {
            Execute_Cube<true>(inputs, outputs, mMode, backend());
        } else {
            Execute_Cube<false>(inputs, outputs, mMode, backend());
        }
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }
protected:
    SampleMode mMode;
    bool mGrad;
};


class CPUTextureCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto gridSampleParam = op->main_as_GridSample();
        auto mode = gridSampleParam->paddingMode();
        if (mode != BorderMode_CUBE) {
            if (gridSampleParam->backward()) {
                return new CPUTexture2DGrad(backend, op);
            }
            return new CPUTexture2D(backend, op);
        }
        return new CPUTextureCube(backend, op);
    }
};
#endif

REGISTER_CPU_OP_CREATOR_RENDER(CPUTextureCreator, OpType_Texture);

} // namespace MNN
