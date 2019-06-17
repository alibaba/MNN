//
//  CPUConv2DBackPropFilter.cpp
//  MNN
//
//  Created by MNN on 2019/4/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUConv2DBackPropFilter.hpp"
#include "Concurrency.h"
#include "Macro.h"
#include "Vec4.hpp"
using namespace MNN::Math;
namespace MNN {
CPUConv2DBackPropFilter::CPUConv2DBackPropFilter(const Convolution2DCommon *convOp, Backend *bn)
    : CPUConvolution(convOp, bn) {
}

ErrorCode CPUConv2DBackPropFilter::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto originWeight = inputs[0];
    auto input        = inputs[1];
    auto outputDiff   = inputs[2];

    // Compute Pad
    CPUConvolution::onResize({input}, {outputDiff});

    if (mCommon->group() > 1) {
        mTempWeight.reset(Tensor::createDevice<float>(
            {UP_DIV(outputDiff->channel(), 4), originWeight->width() * originWeight->height(), 4}));
    } else {
        mTempWeight.reset(Tensor::createDevice<float>({UP_DIV(outputDiff->channel(), 4),
                                                       originWeight->width() * originWeight->height(),
                                                       UP_DIV(input->channel(), 4), 16}));
    }
    mTempCol.reset(
        Tensor::createDevice<float>({originWeight->width() * originWeight->height(), UP_DIV(input->channel(), 4),
                                     outputDiff->height(), outputDiff->width(), 4}));

    backend()->onAcquireBuffer(mTempCol.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);

    backend()->onReleaseBuffer(mTempCol.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);

    mStrideX = mCommon->strideX();
    mStrideY = mCommon->strideY();
    mDilateX = mCommon->dilateX();
    mDilateY = mCommon->dilateY();

    return NO_ERROR;
}
ErrorCode CPUConv2DBackPropFilter::onExecute(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    auto originWeight = inputs[0];
    auto input        = inputs[1];
    auto outputDiff   = inputs[2];
    int batch         = input->batch();
    auto dstWeight    = mTempWeight->host<float>();
    ::memset(dstWeight, 0, mTempWeight->size());
    auto kw          = originWeight->width();
    auto kh          = originWeight->height();
    int ic           = input->channel();
    int oc           = outputDiff->channel();
    auto icC4        = UP_DIV(input->channel(), 4);
    auto ocC4        = UP_DIV(outputDiff->channel(), 4);
    auto ow          = outputDiff->width();
    auto oh          = outputDiff->height();
    auto iw          = input->width();
    auto ih          = input->height();
    auto colBuffer   = mTempCol->host<float>();
    auto plane       = ow * oh;
    auto kernelCount = icC4 * kw * kh;

    // Compute
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto inputBatch = input->host<float>() + batchIndex * input->stride(0);
        // Im2Col
        ::memset(colBuffer, 0, mTempCol->size());
        for (int oy = 0; oy < oh; ++oy) {
            for (int ky = 0; ky < kh; ++ky) {
                auto sy = oy * mStrideY - mPadY + ky * mDilateY;
                if (sy < 0 || sy >= ih) {
                    continue;
                }
                for (int ox = 0; ox < ow; ++ox) {
                    for (int kx = 0; kx < kw; ++kx) {
                        auto sx = ox * mStrideX - mPadX + kx * mDilateX;
                        if (sx < 0 || sx >= iw) {
                            continue;
                        }
                        auto colChannel = colBuffer + (kx + ky * kw) * plane * icC4 * 4 + (oy * ow + ox) * 4;
                        auto srcChannel = inputBatch + (sy * iw + sx) * 4;
                        for (int z = 0; z < icC4; ++z) {
                            Vec4::save(colChannel + 4 * ow * oh * z, Vec4::load(srcChannel + 4 * iw * ih * z));
                        }
                    }
                }
            }
        }
        auto outputBatch = outputDiff->host<float>() + batchIndex * outputDiff->stride(0);
        for (int dz = 0; dz < ocC4; ++dz) {
            auto outputZ  = outputBatch + dz * 4 * plane;
            auto weightDz = dstWeight + dz * kernelCount * 16;

            for (int sz = 0; sz < kernelCount; ++sz) {
                auto weightSz = weightDz + 16 * sz;
                auto colSz    = colBuffer + sz * 4 * plane;
                Vec4 w0       = Vec4::load(weightSz + 4 * 0);
                Vec4 w1       = Vec4::load(weightSz + 4 * 1);
                Vec4 w2       = Vec4::load(weightSz + 4 * 2);
                Vec4 w3       = Vec4::load(weightSz + 4 * 3);
                for (int x = 0; x < plane; ++x) {
                    Vec4 s = Vec4::load(colSz + 4 * x);
                    Vec4 d = Vec4::load(outputZ + 4 * x);

                    w0 = w0 + d * s[0];
                    w1 = w1 + d * s[1];
                    w2 = w2 + d * s[2];
                    w3 = w3 + d * s[3];
                }
                Vec4::save(weightSz + 4 * 0, w0);
                Vec4::save(weightSz + 4 * 1, w1);
                Vec4::save(weightSz + 4 * 2, w2);
                Vec4::save(weightSz + 4 * 3, w3);
            }
        }
    }

    // Reorder
    auto weightDiff      = outputs[0];
    auto targetWeightPtr = weightDiff->host<float>();
    int cur              = 0;
    for (int oz = 0; oz < oc; ++oz) {
        auto srcWeightOz = mTempWeight->host<float>() + (oz / 4) * mTempWeight->stride(0) + (oz % 4);
        for (int sz = 0; sz < ic; ++sz) {
            auto srcWeightSz = srcWeightOz + (sz / 4) * 16 + 4 * (sz % 4);
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    targetWeightPtr[cur++] = srcWeightSz[mTempWeight->stride(1) * (ky * kw + kx)];
                }
            }
        }
    }
    return NO_ERROR;
}

class CPUConv2DBackPropFilterDepthwise : public CPUConv2DBackPropFilter {
public:
    CPUConv2DBackPropFilterDepthwise(const Convolution2DCommon *common, Backend *bn)
        : CPUConv2DBackPropFilter(common, bn) {
    }
    virtual ~CPUConv2DBackPropFilterDepthwise() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto originWeight = inputs[0];
        auto input        = inputs[1];
        auto outputDiff   = inputs[2];
        int batch         = input->batch();
        auto dstWeight    = mTempWeight->host<float>();
        ::memset(dstWeight, 0, mTempWeight->size());
        auto kw        = originWeight->width();
        auto kh        = originWeight->height();
        int oc         = outputDiff->channel();
        auto icC4      = UP_DIV(input->channel(), 4);
        auto ocC4      = UP_DIV(outputDiff->channel(), 4);
        auto ow        = outputDiff->width();
        auto oh        = outputDiff->height();
        auto iw        = input->width();
        auto ih        = input->height();
        auto colBuffer = mTempCol->host<float>();
        auto plane     = ow * oh;

        // Compute
        for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
            auto inputBatch = input->host<float>() + batchIndex * input->stride(0);
            // Im2Col
            ::memset(colBuffer, 0, mTempCol->size());
            for (int oy = 0; oy < oh; ++oy) {
                for (int ky = 0; ky < kh; ++ky) {
                    auto sy = oy * mStrideY - mPadY + ky * mDilateY;
                    if (sy < 0 || sy >= ih) {
                        continue;
                    }
                    for (int ox = 0; ox < ow; ++ox) {
                        for (int kx = 0; kx < kw; ++kx) {
                            auto sx = ox * mStrideX - mPadX + kx * mDilateX;
                            if (sx < 0 || sx >= iw) {
                                continue;
                            }
                            auto colChannel = colBuffer + (kx + ky * kw) * plane * icC4 * 4 + (oy * ow + ox) * 4;
                            auto srcChannel = inputBatch + (sy * iw + sx) * 4;
                            for (int z = 0; z < icC4; ++z) {
                                Vec4::save(colChannel + 4 * ow * oh * z, Vec4::load(srcChannel + 4 * iw * ih * z));
                            }
                        }
                    }
                }
            }
            auto outputBatch = outputDiff->host<float>() + batchIndex * outputDiff->stride(0);
            for (int dz = 0; dz < ocC4; ++dz) {
                auto outputZ  = outputBatch + dz * 4 * plane;
                auto weightDz = dstWeight + dz * kw * kh * 4;

                auto weightSz = weightDz;
                auto colSz    = colBuffer + dz * 4 * plane;
                for (int k = 0; k < kw * kh; ++k) {
                    auto colK = colSz + k * plane * icC4 * 4;
                    Vec4 w0   = Vec4::load(weightSz + 4 * k);
                    for (int x = 0; x < plane; ++x) {
                        Vec4 s = Vec4::load(colK + 4 * x);
                        Vec4 d = Vec4::load(outputZ + 4 * x);

                        w0 = w0 + d * s;
                    }
                    Vec4::save(weightSz + 4 * k, w0);
                }
            }
        }

        // Reorder
        auto weightDiff      = outputs[0];
        auto targetWeightPtr = weightDiff->host<float>();
        int cur              = 0;
        for (int oz = 0; oz < oc; ++oz) {
            auto srcWeightOz = mTempWeight->host<float>() + (oz / 4) * mTempWeight->stride(0) + (oz % 4);
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    targetWeightPtr[cur++] = srcWeightOz[4 * (ky * kw + kx)];
                }
            }
        }
        return NO_ERROR;
    }
};
class CPUConv2DBackPropFilterCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto conv2DCommon = op->main_as_Convolution2D()->common();
        if (inputs[1]->channel() == inputs[2]->channel() && inputs[1]->channel() == conv2DCommon->group()) {
            return new CPUConv2DBackPropFilterDepthwise(conv2DCommon, backend);
        }
        return new CPUConv2DBackPropFilter(conv2DCommon, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConv2DBackPropFilterCreator, OpType_Conv2DBackPropFilter);

} // namespace MNN
