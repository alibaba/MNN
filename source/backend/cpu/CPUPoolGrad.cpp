//
//  CPUPoolGrad.cpp
//  MNN
//
//  Created by jiangxiaotang on 2019/4/19.
//  Copyright © 2019 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUPoolGrad.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include "core/Concurrency.h"

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {
class CPUMaxPoolGrad : public CPUCommonPoolGrad {
public:
    CPUMaxPoolGrad(Backend *b, const Pool *parameter) : CPUCommonPoolGrad(b, parameter) {}

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto origin       = inputs[0];
        auto outputOrigin = inputs[1];
        auto inputDiff    = inputs[2];
        auto outputDiff   = outputs[0];

        auto ow = inputDiff->width();
        auto oh = inputDiff->height();
        auto iw = origin->width();
        auto ih = origin->height();

        //MNN_PRINT("%d, %d, %d, %d\n", origin->width(), outputOrigin->width(), inputDiff->width(), outputDiff->width());

        auto channelC4 = UP_DIV(inputDiff->channel(), 4);
        auto batch     = inputDiff->batch();
        auto totalChannelC4 = batch * channelC4;
        auto threadNumber = ((CPUBackend*)(backend()))->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int z = tId; z < totalChannelC4; z+=threadNumber) {
                auto inputZ0    = origin->host<float>() + z * iw * ih * 4;
                auto inputZ1    = inputDiff->host<float>() + z * ow * oh * 4;
                auto outputOriZ = outputOrigin->host<float>() + z * ow * oh * 4;
                auto outputZ    = outputDiff->host<float>() + z * iw * ih * 4;

                ::memset(outputZ, 0, sizeof(float) * iw * ih * 4);
                for (int y = 0; y < oh; ++y) {
                    for (int x = 0; x < ow; ++x) {
                        Vec4 maxValue = Vec4::load(outputOriZ + 4 * (x + y * ow));
                        Vec4 diffValue   = Vec4::load(inputZ1 + 4 * (x + y * ow));
                        bool unfinished[4] = {true, true, true, true};
                        for (int ky = 0; ky < mKernelY; ++ky) {
                            auto sy = y * mStrideY + ky - mPadY;
                            if (sy < 0 || sy >= ih) {
                                continue;
                            }
                            for (int kx = 0; kx < mKernelX; ++kx) {
                                auto sx = x * mStrideX + kx - mPadX;
                                if (sx < 0 || sx >= iw) {
                                    continue;
                                }
                                Vec4 originValue = Vec4::load(inputZ0 + 4 * (sx + sy * iw));
                                auto dst         = outputZ + 4 * (sx + sy * iw);
                                for (int j = 0; j < 4; ++j) {
                                    if (unfinished[j] && originValue[j] >= maxValue[j]) {
                                        unfinished[j] = false;
                                        dst[j] = dst[j] + diffValue[j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        MNN_CONCURRENCY_END();

        return NO_ERROR;
    }
};

class CPUAvgPoolGrad : public CPUCommonPoolGrad {
public:
    CPUAvgPoolGrad(Backend *b, const Pool *parameter) : CPUCommonPoolGrad(b, parameter) {}

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto origin       = inputs[0];
        auto inputDiff    = inputs[2];
        auto outputDiff   = outputs[0];

        auto ow = inputDiff->width();
        auto oh = inputDiff->height();
        auto iw = origin->width();
        auto ih = origin->height();

        auto channelC4 = UP_DIV(inputDiff->channel(), 4);
        auto batch     = inputDiff->batch();
        auto factor = Vec4(1.0f/((float)mKernelY*mKernelX));
        auto totalChannelC4 = batch * channelC4;
        auto threadNumber = ((CPUBackend*)(backend()))->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int z = tId; z < totalChannelC4; z+=threadNumber) {
                auto inputZ1    = inputDiff->host<float>() + z * ow * oh * 4;
                auto outputZ    = outputDiff->host<float>() + z * iw * ih * 4;

                ::memset(outputZ, 0, sizeof(float) * iw * ih * 4);
                for (int y = 0; y < oh; ++y) {
                    for (int x = 0; x < ow; ++x) {
                        Vec4 diffValue   = Vec4::load(inputZ1 + 4 * (x + y * ow)) * factor;
                        for (int ky = 0; ky < mKernelY; ++ky) {
                            auto sy = y * mStrideY + ky - mPadY;
                            if (sy < 0 || sy >= ih) {
                                continue;
                            }
                            for (int kx = 0; kx < mKernelX; ++kx) {
                                auto sx = x * mStrideX + kx - mPadX;
                                if (sx < 0 || sx >= iw) {
                                    continue;
                                }
                                auto dst         = outputZ + 4 * (sx + sy * iw);
                                Vec4::save(dst, Vec4::load(dst) + diffValue);
                            }
                        }
                    }
                }
            }
        };
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
};

class CPUPoolGradCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto pool = op->main_as_Pool();
        if (pool->type() == PoolType_MAXPOOL) {
            return new CPUMaxPoolGrad(backend, op->main_as_Pool());
        } else if (pool->type() == PoolType_AVEPOOL){
            return new CPUAvgPoolGrad(backend, op->main_as_Pool());
        }
        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(CPUPoolGradCreator, OpType_PoolGrad);
} // namespace MNN
