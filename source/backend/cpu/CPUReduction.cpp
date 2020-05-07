//
//  CPUReduction.cpp
//  MNN
//
//  Created by MNN on 2018/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReduction.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include <cmath>
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#define UNIT 4
#define UNIT_DUP(value) \
    { (value), (value), (value), (value) }

namespace MNN {
// outside, axis, inside

class Reduction : public Execution {
public:
    Reduction(Backend* backend, const Op* op) : Execution(backend) {
        mOp = op;
    }
    virtual ~Reduction() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];
        auto typeCode = input->getType().code;
        auto src = inputs[0];
        for (int i=0; i<mMidBuffer.size(); ++i) {
            auto reduceDim = mReduceDims[i];
            auto inside = std::get<2>(reduceDim);
            auto outside = std::get<0>(reduceDim);
            auto axis = std::get<1>(reduceDim);
            auto dst = mMidBuffer[i].get();
            if (halide_type_float == typeCode) {
                this->onReduce(src->host<float>(), dst->host<float>(), inside, outside, axis);
            } else if (halide_type_int == typeCode) {
                this->onReduce(src->host<int32_t>(), dst->host<int32_t>(), inside, outside, axis);
            }
            src = dst;
        }
        auto reduceDim = mReduceDims[mReduceDims.size()-1];
        auto inside = std::get<2>(reduceDim);
        auto outside = std::get<0>(reduceDim);
        auto axis = std::get<1>(reduceDim);
        auto dst = output;
        //MNN_ASSERT(output->elementSize() == inside * outside);
        if (halide_type_float == typeCode) {
            this->onReduce(src->host<float>(), dst->host<float>(), inside, outside, axis);
        } else if (halide_type_int == typeCode) {
            this->onReduce(src->host<int32_t>(), dst->host<int32_t>(), inside, outside, axis);
        }
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mReduceDims = OpCommonUtils::computeReduceDims(inputs, mOp);
        mMidBuffer.clear();
        auto input = inputs[0];
        std::vector<int> reducedAxis;
        for (int i = 0; i < mReduceDims.size() - 1; ++i) {
            const auto reduceDim = mReduceDims[i];
            auto inside = std::get<2>(reduceDim);
            auto outside = std::get<0>(reduceDim);
            auto tensor = Tensor::createDevice({inside*outside}, input->getType());
            mMidBuffer.push_back(std::unique_ptr<Tensor>(tensor));
        }
        for (auto& t : mMidBuffer) {
            backend()->onAcquireBuffer(t.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(t.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axis) const     = 0;
    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outsize, int axis) const = 0;
    std::vector<std::unique_ptr<Tensor>> mMidBuffer;
    std::vector<std::tuple<int, int, int>> mReduceDims;
    const Op* mOp;
};

class MeanReduce : public Reduction {
public:
    MeanReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MeanReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside % 4 == 0) {
                    ::memcpy(dstOutSide, srcOutSide, inside * sizeof(float));
                    for (int a = 1; a < axisSize; ++a) {
                        auto srcAxis = srcOutSide + a * inside;
                        MNNMatrixAddCommon(dstOutSide, dstOutSide, srcAxis, inside, 0, 0, 0, 1);
                    }
                    float divide = 1.0f / (float)axisSize;
                    for (int i=0; i<inside; ++i) {
                        dstOutSide[i] = dstOutSide[i] * divide;
                    }
                } else {
                    for (int ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutSide + ii;
                        auto dstInside = dstOutSide + ii;
                        float summer   = 0.0f;
                        for (int a = 0; a < axisSize; ++a) {
                            summer += srcInside[a * inside];
                        }
                        *dstInside = summer / (float)axisSize;
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += srcInside[a * inside];
                }
                *dstInside = summer / axisSize;
            }
        }
    }
};

class SumReduce : public Reduction {
public:
    SumReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~SumReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside % 4 == 0) {
                    ::memcpy(dstOutSide, srcOutSide, inside * sizeof(float));
                    for (int a = 1; a < axisSize; ++a) {
                        auto srcAxis = srcOutSide + a * inside;
                        MNNMatrixAddCommon(dstOutSide, dstOutSide, srcAxis, inside, 0, 0, 0, 1);
                    }
                } else {
                    for (int ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutSide + ii;
                        auto dstInside = dstOutSide + ii;
                        float summer   = 0.0f;
                        for (int a = 0; a < axisSize; ++a) {
                            summer += srcInside[a * inside];
                        }
                        *dstInside = summer;
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += srcInside[a * inside];
                }
                *dstInside = summer;
            }
        }
    }
};

class MinReduce : public Reduction {
public:
    MinReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MinReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float Min      = srcInside[0];
                if (1 == inside) {
                    int32_t inputCountUnit = axisSize / (UNIT * 2);
                    int32_t remain         = axisSize - (inputCountUnit * UNIT * 2);
                    float minArray[UNIT]   = UNIT_DUP(Min);
                    MNNMinFloat((float*)srcInside, minArray, inputCountUnit);

                    for (int i = 0; i < UNIT; i++) {
                        Min = std::min(Min, minArray[i]);
                    }
                    if (remain > 0) {
                        int currentIndex = inputCountUnit * UNIT * 2;
                        for (int i = 0; i < remain; i++) {
                            float currentInputData = srcInside[currentIndex + i];
                            Min                    = std::min(Min, currentInputData);
                        }
                    }
                } else {
                    for (int a = 0; a < axisSize; ++a) {
                        Min = std::min(Min, srcInside[a * inside]);
                    }
                }
                *dstInside = Min;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t Min    = srcInside[0];
                for (int a = 0; a < axisSize; ++a) {
                    Min = std::min(Min, srcInside[a * inside]);
                }
                *dstInside = Min;
            }
        }
    }
};

class MaxReduce : public Reduction {
public:
    MaxReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MaxReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float Max      = srcInside[0];
                if (1 == inside) {
                    int32_t inputCountUnit = axisSize / (UNIT * 2);
                    int32_t remain         = axisSize - (inputCountUnit * UNIT * 2);
                    float maxArray[UNIT]   = UNIT_DUP(Max);

                    MNNMaxFloat((float*)srcInside, maxArray, inputCountUnit);

                    for (int i = 0; i < UNIT; i++) {
                        Max = std::max(Max, maxArray[i]);
                    }
                    if (remain > 0) {
                        int currentIndex = inputCountUnit * UNIT * 2;
                        for (int i = 0; i < remain; i++) {
                            float currentInputData = srcInside[currentIndex + i];
                            Max                    = std::max(Max, currentInputData);
                        }
                    }
                } else {
                    for (int a = 0; a < axisSize; ++a) {
                        Max = std::max(Max, srcInside[a * inside]);
                    }
                }
                *dstInside = Max;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t Max    = srcInside[0];
                for (int a = 0; a < axisSize; ++a) {
                    Max = std::max(Max, srcInside[a * inside]);
                }
                *dstInside = Max;
            }
        }
    }
};

class ProdReduce : public Reduction {
public:
    ProdReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ProdReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float product  = 1.0f;
                for (int a = 0; a < axisSize; ++a) {
                    product *= srcInside[a * inside];
                }
                *dstInside = product;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t product = 1;
                for (int a = 0; a < axisSize; ++a) {
                    product *= srcInside[a * inside];
                }
                *dstInside = product;
            }
        }
    }
};

class AnyReduce : public Reduction {
public:
    AnyReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ AnyReduce() = default;
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        MNN_ASSERT(false);
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t result = 0;
                for (int a = 0; a < axisSize; ++a) {
                    if (srcInside[a * inside] > 0) {
                        result = 1;
                        break;
                    }
                }
                *dstInside = result;
            }
        }
    }
};

class AllReduce : public Reduction {
public:
    AllReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ AllReduce() = default;
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        MNN_ASSERT(false);
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t result = 1;
                for (int a = 0; a < axisSize; ++a) {
                    if (srcInside[a * inside] == 0) {
                        result = 0;
                        break;
                    }
                }
                *dstInside = result;
            }
        }
    }
};

Execution* CPUReductionCreator::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) const {
    auto type = inputs[0]->getType();
    if (type.bits != 32) {
        return nullptr;
    }
    if (type.code != halide_type_float && type.code != halide_type_int) {
        return nullptr;
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            return new MeanReduce(backend, op);
        case ReductionType_SUM:
            return new SumReduce(backend, op);
        case ReductionType_MINIMUM:
            return new MinReduce(backend, op);
        case ReductionType_MAXIMUM:
            return new MaxReduce(backend, op);
        case ReductionType_PROD:
            return new ProdReduce(backend, op);
        case ReductionType_ANY:
            return new AnyReduce(backend, op);
        case ReductionType_ALL:
            return new AllReduce(backend, op);
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

REGISTER_CPU_OP_CREATOR(CPUReductionCreator, OpType_Reduction);

} // namespace MNN
