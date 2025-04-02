#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#include "../CPUBackend.hpp"
#include <cmath>
#include <algorithm>
#include <array>
#include "half.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {
#ifdef MNN_SUPPORT_RENDER

struct Point  {
    float x;
    float y;
    float z;
    float w;
};

class CPURasterAndInterpolate : public Execution {
public:
    CPURasterAndInterpolate(Backend* bn, bool hasIndice, int primitiveType) : Execution(bn) {
        mIndice = hasIndice;
        mType = primitiveType;
    }
    virtual ~ CPURasterAndInterpolate() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }

    void _rasterPointWithPointsize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        auto rasterOutput = outputs[0];
        auto width = rasterOutput->length(2);
        auto height = rasterOutput->length(1);
        auto rasterBufferPtr = rasterOutput->host<float>();
        int pixelSize = height * width;
        auto pointSize = inputs[1];
        auto position = inputs[2];
        auto pointSizePtr = pointSize->host<float>();
        auto posPtr = position->host<float>();
        auto numberPoint = pointSize->length(0);
        auto tY = [height](float x) {
            return ((x + 1.0f) * 0.5f * height) - 0.5f;
        };
        auto tX = [width](float x) {
            return ((x + 1.0f) * 0.5f * width) - 0.5f;
        };
        for (int index=0; index<numberPoint; ++index) {
            auto x = posPtr[4 * index + 0];
            auto y = posPtr[4 * index + 1];
            auto z = posPtr[4 * index + 2];
            auto w = posPtr[4 * index + 3];
            auto PSize = (int)pointSizePtr[index];
            x = tX(x / w);
            y = tY(y / w);
            z = z / w;
            int yMin = y - PSize;
            yMin = ALIMIN(ALIMAX(yMin, 0), height-1);
            int yMax = y + PSize;
            yMax = ALIMIN(ALIMAX(yMax, 0), height-1);
            int xMin = x - PSize;
            xMin = ALIMIN(ALIMAX(xMin, 0), width-1);
            int xMax = x + PSize;
            xMax = ALIMIN(ALIMAX(xMax, 0), width-1);
            for (int yi=yMin; yi<yMax; ++yi) {
                for (int xi=xMin; xi<xMax; ++xi) {
                    auto dstPtr = rasterBufferPtr + 4 * (yi*width + xi);
                    auto curZ = z;
                    if (curZ > dstPtr[2]) {
                        continue;
                    }
                    dstPtr[2] = curZ;
                    dstPtr[3] = index;
                    dstPtr[0] = 0.0;
                    dstPtr[1] = 0.0;
                    for (int i=1; i<outputs.size(); ++i) {
                        auto unit = outputs[i]->length(3);
                        auto srcPtr = inputs[i+2]->host<float>();
                        auto ndstPtr = outputs[i]->host<float>() + (yi*width + xi) * unit;
                        auto src = srcPtr + index * unit;
                        for (int j=0; j<unit; ++j) {
                            ndstPtr[j] = src[j];
                        }
                    }
                }
            }

        }
    }

    void _rasterTriangleWithIndice(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        auto rasterOutput = outputs[0];
        auto width = rasterOutput->length(2);
        auto height = rasterOutput->length(1);
        auto rasterBufferPtr = rasterOutput->host<float>();
        int pixelSize = height * width;
        auto indice = inputs[1];
        auto position = inputs[2];
        auto indicePtr = indice->host<int>();
        auto posPtr = position->host<float>();

        auto numberTriangle = indice->elementSize() / 3;;
        for (int triangleIndex=0; triangleIndex < numberTriangle; ++triangleIndex) {
            auto i0 = indicePtr[3 * triangleIndex + 0];
            auto i1 = indicePtr[3 * triangleIndex + 1];
            auto i2 = indicePtr[3 * triangleIndex + 2];
            auto x0 = posPtr[4 * i0 + 0];
            auto x1 = posPtr[4 * i1 + 0];
            auto x2 = posPtr[4 * i2 + 0];

            auto y0 = posPtr[4 * i0 + 1];
            auto y1 = posPtr[4 * i1 + 1];
            auto y2 = posPtr[4 * i2 + 1];

            auto z0 = posPtr[4 * i0 + 2];
            auto z1 = posPtr[4 * i1 + 2];
            auto z2 = posPtr[4 * i2 + 2];

            auto w0 = posPtr[4 * i0 + 3];
            auto w1 = posPtr[4 * i1 + 3];
            auto w2 = posPtr[4 * i2 + 3];

            std::vector<Point> points {
                {x0, y0, z0, w0},
                {x1, y1, z1, w1},
                {x2, y2, z2, w2},
            };

            // TODO: Clip
            auto tY = [height](float x) {
                return ((x + 1.0f) * 0.5f * height);
            };
            auto rY = [height](int y) {
                return (float)y / (float)height * 2.0f - 1.0f;
            };
            auto tX = [width](float x) {
                return ((x + 1.0f) * 0.5f * width);
            };
            auto rX = [width](int y) {
                return (float)y / (float)width * 2.0f - 1.0f;
            };

            for (int i=2; i<points.size(); ++i) {
                auto& p0 = points[i-2];
                auto& p1 = points[i-1];
                auto& p2 = points[i-0];
                x0 = p0.x / p0.w;
                x1 = p1.x / p1.w;
                x2 = p2.x / p2.w;
                y0 = p0.y / p0.w;
                y1 = p1.y / p1.w;
                y2 = p2.y / p2.w;
                z0 = p0.z / p0.w;
                z1 = p1.z / p1.w;
                z2 = p2.z / p2.w;
                // MNN_PRINT("[%f,%f,%f], [%f,%f,%f], [%f,%f,%f]\n", x0, y0, z1, x1, y1, z1, x2, y2, z2);

                // Reorder triangle, make mid y as y1
                if (y1 < y0) {
                    std::swap(y1, y0);
                    std::swap(x1, x0);
                }
                if (y2 < y0) {
                    std::swap(y2, y0);
                    std::swap(x2, x0);
                }
                if (y2 < y1) {
                    std::swap(y2, y1);
                    std::swap(x2, x1);
                }
                // Split triangle by mid y
                float c01 = (x1-x0) / (y1-y0);
                if (std::fabs(y1-y0) < 0.0001f) {
                    c01 = 0.0f;
                }
                float c02 = (x2-x0) / (y2-y0);
                if (std::fabs(y2-y0) < 0.0001f) {
                    c02 = 0.0f;
                }
                float c12 = (x2-x1) / (y2-y1);
                if (std::fabs(y2-y1) < 0.0001f) {
                    c12 = 0.0f;
                }
                float my = y1;
                
                // Compute Y Range and iter step
                int y0i = floorf(tY(y0));
                int y2i = ceilf(tY(y2));
                int myil = floorf(tY(my));
                int myir = ceilf(tY(my));

                auto rasterTriangle = [width, height, tY, tX, rY, rX, rasterBufferPtr, &outputs, &inputs, i0, i1, i2, p0, p1, p2] (int y0i, int myi, float c01, float c02, float x0, float y0, float z0, float z1, float z2, int triangleIndex) {
                    if (y0i > myi) {
                        std::swap(y0i, myi);
                    }
                    if (y0i < 0) {
                        y0i = 0;
                    }
                    if (myi > height) {
                        myi = height;
                    }
                    for (int yi=y0i; yi<myi; ++yi) {
                        float yif = rY(yi);
                        float xli = tX(c01 * (yif - y0) + x0);
                        float xri = tX(c02 * (yif - y0) + x0);
                        if (xli > xri) {
                            std::swap(xli, xri);
                        }
                        int xs = floorf(xli);
                        int xe = ceilf(xri);
                        if (xs < 0) {
                            xs = 0;
                        }
                        if (xe > width) {
                            xe = width;
                        }
                        for (int xi=xs; xi<xe; ++xi) {
                            auto dstPtr = rasterBufferPtr + 4 * (yi*width + xi);
                            // Evaluate edge functions.
                            float fx = rX(xi);
                            float fy = yif;
                            float p0x = p0.x - fx * p0.w;
                            float p0y = p0.y - fy * p0.w;
                            float p1x = p1.x - fx * p1.w;
                            float p1y = p1.y - fy * p1.w;
                            float p2x = p2.x - fx * p2.w;
                            float p2y = p2.y - fy * p2.w;
                            float a0 = p1x*p2y - p1y*p2x;
                            float a1 = p2x*p0y - p2y*p0x;
                            float a2 = p0x*p1y - p0y*p1x;

                            // Perspective correct, normalized barycentrics.
                            float iw = 1.f / (a0 + a1 + a2);
                            float b0 = a0 * iw;
                            float b1 = a1 * iw;

                            // Compute z/w for depth buffer.
                            float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
                            float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
                            float zw = z / w;
                            float b2 = 1.0f - b0 - b1;
                            if (b2 < 0.0f || b2 > 1.0f) {
                                continue;
                            }
                            if (b1 < 0.0f || b1 > 1.0f) {
                                continue;
                            }
                            if (b0 < 0.0f || b0 > 1.0f) {
                                continue;
                            }
                            // Clamps to avoid NaNs.
                            b0 = fmaxf(fminf(b0, 1.0f), 0.0f); // Clamp to [+0.0, 1.0].
                            b1 = fmaxf(fminf(b1, 1.0f), 0.0f); // Clamp to [+0.0, 1.0].
                            zw = fmaxf(fminf(zw, 1.f), -1.f);
                            float curZ = zw;
                            if (curZ > dstPtr[2]) {
                                continue;
                            }
                            dstPtr[2] = curZ;
                            dstPtr[3] = triangleIndex;
                            dstPtr[0] = b0;
                            dstPtr[1] = b1;
                            for (int i=1; i<outputs.size(); ++i) {
                                auto unit = outputs[i]->length(3);
                                auto srcPtr = inputs[i+2]->host<float>();
                                auto ndstPtr = outputs[i]->host<float>() + (yi*width + xi) * unit;
                                auto src0 = srcPtr + i0 * unit;
                                auto src1 = srcPtr + i1 * unit;
                                auto src2 = srcPtr + i2 * unit;
                                for (int j=0; j<unit; ++j) {
                                    ndstPtr[j] = b0 * src0[j] + b1 * src1[j] + b2 * src2[j];
                                }
                            }
                        }
                    }
                };
                // Up triangle
                rasterTriangle(y0i, myir, c01, c02, x0, y0, z0, z1, z2, triangleIndex + 1);
                // Down triangle
                rasterTriangle(myil, y2i, c12, c02, x2, y2, z0, z1, z2, triangleIndex + 1);
            }
        }
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto rasterOutput = outputs[0];
        auto width = rasterOutput->length(2);
        auto height = rasterOutput->length(1);
        
        // Init Output, index + 1, w0, w1, z
        int pixelSize = height * width;
        auto rasterBufferPtr = rasterOutput->host<float>();
        for (int i=0; i<pixelSize; ++i) {
            rasterBufferPtr[4 * i + 0] = 0.0f;
            rasterBufferPtr[4 * i + 1] = 0.0f;
            rasterBufferPtr[4 * i + 2] = 1.0f;
            rasterBufferPtr[4 * i + 3] = 0.0f;
        }
        for (int i=1; i<outputs.size(); ++i) {
            auto ptr = outputs[i]->host<float>();
            ::memset(ptr, 0, pixelSize * outputs[i]->length(3) * sizeof(float));
        }
        if (mIndice && mType == 4) {
            _rasterTriangleWithIndice(inputs, outputs);
        } else if ((!mIndice) && (mType == 0)) {
            _rasterPointWithPointsize(inputs, outputs);
        }
        return NO_ERROR;
    }
private:
    bool mIndice;
    int mType;
};


class CPURasterSort : public Execution {
private:
    std::vector<std::pair<uint32_t, uint32_t>> mKV;
    std::array<uint32_t, 65536> mHistorm;
    std::array<uint32_t, 65536> mHistormOffset;
public:
    CPURasterSort(Backend* bn) :  Execution(bn) {
        
    }
    virtual ~ CPURasterSort() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto number = inputs[0]->length(0);
        mKV.resize(number);
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto number = inputs[0]->length(0);
        auto attr = inputs[0];
        auto viewProj = inputs[1]->host<float>();
        auto numAttr = attr->length(1);
        auto outputPtr = outputs[1]->host<float>();
        for (int i=0; i<mHistorm.size(); ++i) {
            mHistorm[i] = 0;
        }
//        for (int i=0; i<16; ++i) {
//            printf("%f, ", viewProj[i]);
//        }
//        printf("\n");
        // Make KV
        int validNumber = 0;
        float maxDepth = 1.0f;
        const float minDepth = 0.0f;
        float scale = 1.0f / (maxDepth - minDepth);
        if (numAttr == 4) {
            // Float
            for (int i=0; i<number; ++i) {
                auto srcPtr = (inputs[0]->host<float>() + numAttr * i);
                auto x = srcPtr[0];
                auto y = srcPtr[1];
                auto z = srcPtr[2];
                auto depth =
                viewProj[2] * x +
                viewProj[6] * y +
                viewProj[10] * z + viewProj[14];
                auto dw =
                viewProj[3] * x +
                viewProj[7] * y +
                viewProj[11] * z + viewProj[15];
                depth = depth / dw;
                if (depth  < minDepth || depth > maxDepth) {
                    continue;
                }
                uint32_t key = scale * (depth - minDepth) * 65535.0f;
                mKV[validNumber].second = i;
                mKV[validNumber].first = key;
                validNumber++;
                mHistorm[key]++;
            }

        } else {
            for (int i=0; i<number; ++i) {
                auto srcPtr = (half_float::half*)(inputs[0]->host<float>() + numAttr * i);
                auto x = srcPtr[0];
                auto y = srcPtr[1];
                auto z = srcPtr[2];
                auto depth =
                viewProj[2] * x +
                viewProj[6] * y +
                viewProj[10] * z + viewProj[14];
                auto dw =
                viewProj[3] * x +
                viewProj[7] * y +
                viewProj[11] * z + viewProj[15];
                depth = depth / dw;
                if (depth  < minDepth || depth > maxDepth) {
                    continue;
                }
                uint32_t key = scale * (depth - minDepth) * 65535.0f;
                mKV[validNumber].second = i;
                mKV[validNumber].first = key;
                validNumber++;
                mHistorm[key]++;
            }

        }
        mHistormOffset[0] = 0;
        for (int i=1; i<mHistormOffset.size(); ++i) {
            mHistormOffset[i] = mHistormOffset[i-1] + mHistorm[i-1];
        }
        auto mKVMid = (std::pair<uint32_t, uint32_t>*)(outputs[1]->host<void>());
        for (int i=0; i<validNumber; ++i) {
            auto key = mKV[i].first;
            auto offset = mHistormOffset[key];
            mKVMid[offset] = mKV[i];
            offset++;
            mHistormOffset[key]=offset;
        }
        outputs[0]->host<int>()[0] = validNumber;
        return NO_ERROR;
    }
};


class CPURasterAndInterpolateCreator : public CPUBackend::Creator {
    virtual Execution* onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                        const MNN::Op *op, Backend *backend) const override {
        bool hasIndice = true;
        int type = 4;
        if (op->main_type() == OpParameter_Extra) {
            auto extra = op->main_as_Extra();
            if (nullptr != extra->attr()) {
                for (int i=0; i<extra->attr()->size(); ++i) {
                    auto attr = extra->attr()->GetAs<Attribute>(i);
                    if (attr->key()->str() == "index") {
                        hasIndice = attr->b();
                        continue;
                    }
                    if (attr->key()->str() == "primitiveType") {
                        type = attr->i();
                        continue;
                    }
                }
            }
        }
        if (6 == type) {
            return new CPURasterSort(backend);
        }
        return new CPURasterAndInterpolate(backend, hasIndice, type);
    }
};

#endif
REGISTER_CPU_OP_CREATOR_RENDER(CPURasterAndInterpolateCreator, OpType_RasterAndInterpolate);
} // namespace MNN
