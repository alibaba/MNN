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
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;

#define UNIT 4
#define UNIT_DUP(value) \
    { (value), (value), (value), (value) }

namespace MNN {
// outside, axis, inside

class Reduction : public Execution {
public:
    Reduction(Backend* backend, const Op* op) : Execution(backend) {
        // Do nothing
        mAxis = op->main_as_ReductionParam()->dim()->data()[0];
    }
    virtual ~Reduction() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];
        auto typeCode = input->getType().code;
        auto src = inputs[0];
        int outside = 1;
        for(int i=0; i<mAxis; ++i) {
            outside *= input->length(i);
        }
        int inside = 1;
        for(int i=mAxis+1; i<input->dimensions(); ++i) {
            inside *= input->length(i);
        }
        auto axis = input->length(mAxis);
        auto dst = output;
        //MNN_ASSERT(output->elementSize() == inside * outside);
        if (halide_type_float == typeCode) {
            this->onReduce(src->host<float>(), dst->host<float>(), inside, outside, axis);
        } else if (halide_type_int == typeCode) {
            this->onReduce(src->host<int32_t>(), dst->host<int32_t>(), inside, outside, axis);
        }
        return NO_ERROR;
    }
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axis) const     = 0;
    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outsize, int axis) const = 0;
private:
    int mAxis = -1;
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
        auto core = static_cast<CPUBackend*>(backend())->functions();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside == 1) {
                    float summer = 0.0f;
                    core->MNNAccumulateSequenceNumber(&summer, srcOutSide, axisSize);
                    *dstOutSide = summer;
                    continue;
                }
                auto insideC = inside / 32;
                auto insideR = inside % 32;
                for (int x=0; x<insideC; ++x) {
                    auto srcX = srcOutSide + x * 32;
                    auto dstX = dstOutSide + x * 32;
                    Vec4 c0(0.0f);
                    Vec4 c1(0.0f);
                    Vec4 c2(0.0f);
                    Vec4 c3(0.0f);
                    Vec4 c4(0.0f);
                    Vec4 c5(0.0f);
                    Vec4 c6(0.0f);
                    Vec4 c7(0.0f);
                    for (int a = 0; a < axisSize; ++a) {
                        auto srcAxis = srcX + a * inside;
                        c0 = c0 + Vec4::load(srcAxis + 4 * 0);
                        c1 = c1 + Vec4::load(srcAxis + 4 * 1);
                        c2 = c2 + Vec4::load(srcAxis + 4 * 2);
                        c3 = c3 + Vec4::load(srcAxis + 4 * 3);
                        c4 = c4 + Vec4::load(srcAxis + 4 * 4);
                        c5 = c5 + Vec4::load(srcAxis + 4 * 5);
                        c6 = c6 + Vec4::load(srcAxis + 4 * 6);
                        c7 = c7 + Vec4::load(srcAxis + 4 * 7);
                    }
                    Vec4::save(dstX + 4 * 0, c0);
                    Vec4::save(dstX + 4 * 1, c1);
                    Vec4::save(dstX + 4 * 2, c2);
                    Vec4::save(dstX + 4 * 3, c3);
                    Vec4::save(dstX + 4 * 4, c4);
                    Vec4::save(dstX + 4 * 5, c5);
                    Vec4::save(dstX + 4 * 6, c6);
                    Vec4::save(dstX + 4 * 7, c7);
                }
                auto remain = insideC * 32;
                if (insideR >= 4) {
                    auto insideC4 = insideR / 4;
                    for (int x=0; x<insideC4; ++x) {
                        Vec4 c0(0.0f);
                        auto srcX = srcOutSide + x * 4 + remain;
                        auto dstX = dstOutSide + x * 4 + remain;
                        for (int a = 0; a < axisSize; ++a) {
                            auto srcAxis = srcX + a * inside;
                            c0 = c0 + Vec4::load(srcAxis + 4 * 0);
                        }
                        Vec4::save(dstX, c0);
                    }
                    remain += insideC4 * 4;
                }
                for (int x=remain; x<inside; ++x) {
                    auto srcX = srcOutSide + x;
                    auto dstX = dstOutSide + x;
                    float sum = 0.0f;
                    for (int a = 0; a < axisSize; ++a) {
                        auto srcAxis = srcX + a * inside;
                        sum += srcAxis[0];
                    }
                    dstX[0] = sum;
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
                if (1 == inside && axisSize > UNIT * 2) {
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
    return create(inputs, outputs, op, backend);
}

Execution* CPUReductionCreator::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) {
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
