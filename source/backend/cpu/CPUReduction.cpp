//
//  CPUReduction.cpp
//  MNN
//
//  Created by MNN on 2018/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReduction.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include <cmath>

#define UNIT 4
#define UNIT_DUP(value) \
    { (value), (value), (value), (value) }

namespace MNN {
class Reduction : public Execution {
public:
    Reduction(Backend* backend, const Op* op) : Execution(backend) {
        auto reduct = op->main_as_ReductionParam();

        if (nullptr == reduct->dim()) {
            return;
        }
        for (int i = 0; i < reduct->dim()->size(); ++i) {
            mAxis.push_back(reduct->dim()->data()[i]);
        }
    }
    virtual ~Reduction() = default;

    void reduce(halide_buffer_t& srcBuffer, halide_buffer_t& dstBuffer, int axis) {
        int outsideSize = 1;
        for (int x = 0; x < axis; ++x) {
            outsideSize *= srcBuffer.dim[x].extent;
        }

        int insideSize = 1;
        for (int x = axis + 1; x < srcBuffer.dimensions; ++x) {
            insideSize *= srcBuffer.dim[x].extent;
        }

        int axisSize = srcBuffer.dim[axis].extent;

        if (halide_type_float == srcBuffer.type.code) {
            this->onReduce((const float*)srcBuffer.host, (float*)dstBuffer.host, insideSize, outsideSize, axisSize);
        } else if (halide_type_int == srcBuffer.type.code) {
            this->onReduce((const int32_t*)srcBuffer.host, (int32_t*)dstBuffer.host, insideSize, outsideSize, axisSize);
        }
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];
        auto typeCode = input->getType().code;
        if (mAxis.empty()) {
            int size = (int)input->size() / input->buffer().type.bytes();
            if (halide_type_float == typeCode) {
                this->onReduce(input->host<float>(), output->host<float>(), 1, 1, size);
            } else if (halide_type_int == typeCode) {
                this->onReduce(input->host<int32_t>(), output->host<int32_t>(), 1, 1, size);
            }
            return NO_ERROR;
        }
        auto srcBuffer = input->buffer();
        for (int i = 0; i < mAxis.size() - 1; ++i) {
            auto axis = mAxis[i];
            if (axis == -1) {
                axis = input->dimensions() - 1;
            }
            auto dstBuffer = mMidBuffer[i]->buffer();
            reduce(srcBuffer, dstBuffer, axis);
            srcBuffer = dstBuffer;
        }
        int lastAxis = mAxis[mAxis.size() - 1];
        if (lastAxis == -1) {
            lastAxis = input->dimensions() - 1;
        }
        reduce(srcBuffer, output->buffer(), lastAxis);
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() >= 2) {
            mAxis.clear();
            auto size = inputs[1]->elementSize();
            auto dims = inputs[1]->host<int32_t>();
            for (int i = 0; i < size; ++i) {
                mAxis.emplace_back(dims[i]);
            }
        }
        if (mAxis.empty()) {
            return NO_ERROR;
        }
        mMidBuffer.clear();
        auto input = inputs[0];
        std::vector<int> reducedAxis;
        for (int i = 0; i < mAxis.size() - 1; ++i) {
            const auto axis = mAxis[i];
            if (axis == -1) {
                reducedAxis.push_back(input->dimensions() - 1);
            } else {
                reducedAxis.push_back(mAxis[i]);
            }
            auto tensor = new Tensor(input->buffer().dimensions);
            ::memcpy(tensor->buffer().dim, input->buffer().dim,
                     input->buffer().dimensions * sizeof(halide_dimension_t));
            for (auto ra : reducedAxis) {
                tensor->buffer().dim[ra].extent = 1;
            }
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
    std::vector<int> mAxis;
    std::vector<std::unique_ptr<Tensor>> mMidBuffer;
};

class MeanReduce : public Reduction {
public:
    MeanReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MeanReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
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
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
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
