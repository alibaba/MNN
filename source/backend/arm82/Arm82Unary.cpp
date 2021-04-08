//
//  Arm82Unary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include <vector>
#include <cmath>
#include <algorithm>
#include "Arm82Unary.hpp"
#include "Arm82Backend.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "core/Concurrency.h"
#include "MNN_generated.h"


#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
Arm82Unary::Arm82Unary(Backend *b, UnaryOpOperation type) : MNN::Execution(b), mType(type) {
    // nothing to do
}

ErrorCode Arm82Unary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    auto dtype = inputs[0]->getType();
    MNN_ASSERT(dtype == halide_type_of<float>() || dtype == halide_type_of<int32_t>());
    return NO_ERROR;
}

template <typename Func, typename T>
static ErrorCode _unaryOp(void* inputPtr, void* outputPtr, int elementSize, Backend* bn) {
    Func f;
    auto backend = [bn]() {
        return bn;
    };
    const T *inputData = (T*)inputPtr;
    T *outputData      = (T *)outputPtr;
    auto numberThread = ((CPUBackend*)bn)->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int i=tId; i<elementSize; i+=numberThread) {
            outputData[i] = f(inputData[i]);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class UnarySquare {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        return x * x;
    }
#ifdef MNN_USE_NEON
    static float16x8_t vecFunc(const float16x8_t& x) {
        return vmulq_f16(x, x);
    }
#endif
};

class UnaryRsqrt {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        return 1.f / sqrt(x);
    }
#ifdef MNN_USE_NEON
    static float16x8_t vecFunc(const float16x8_t& x) {
        return vrsqrteq_f16(x);
    }
#endif
};

class UnarySqrt {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        return sqrt(x);
    }
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    static float16x8_t vecFunc(const float16x8_t& x) {
        return vsqrtq_f16(x);
    }
#endif
};

class UnaryNeg {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        return -x;
    }
#ifdef MNN_USE_NEON
    static float16x8_t vecFunc(const float16x8_t& x) {
        return vnegq_f16(x);
    }
#endif
};

class UnaryAbs {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        return abs(x);
    }
#ifdef MNN_USE_NEON
    static float16x8_t vecFunc(const float16x8_t& x) {
        return vabsq_f16(x);
    }
#endif
};

class UnaryRecipocal {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        return 1.f / x;
    }
#ifdef MNN_USE_NEON
    static float16x8_t vecFunc(const float16x8_t& x) {
        return vrecpeq_f16(x);
    }
#endif
};

class UnaryHardSwish {
public:
    static FLOAT16 scalarFunc(const FLOAT16& x) {
        if (x <= -3) {
            return 0;
        } else if (x >= 3) {
            return x;
        } else {
            return x * (x + 3) / 6;
        }
    }
#ifdef MNN_USE_NEON
    static float16x8_t vecFunc(const float16x8_t& x) {
        float16x8_t value_l = vmovq_n_f16(-3);
        float16x8_t value_h = vmovq_n_f16(3);
        float16x8_t value_d = vmovq_n_f16(1.f/6);
        float16x8_t value_z = vmovq_n_f16(0);
        uint16x8_t right = vcleq_f16(x, value_l);
        float16x8_t middle = vmulq_f16(vmulq_f16(x, vaddq_f16(x, value_h)), value_d);
        float16x8_t tmp = vbslq_f16(right, x, middle);
        uint16x8_t left = vcgtq_f16(x, value_l);
        return vbslq_f16(left, tmp, value_z);
    }
#endif
};

template <typename Helper>
ErrorCode Arm82Unary::onExecuteInternal(Tensor* input, Tensor* output) {
    const int threadNum = ((Arm82Backend*)backend())->threadNumber();
    const int count = ARM82TensorElementSizeHelper(output);
    const FLOAT16* inputData = input->host<FLOAT16>();
    FLOAT16* outputData      = output->host<FLOAT16>();
        
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        int realSize = UP_DIV(UP_DIV(count, ARMV82_CHANNEL_UNIT), threadNum) * ARMV82_CHANNEL_UNIT;
        int startIndex = tId * realSize, endIndex = ALIMIN(startIndex + realSize, count);
        if (endIndex > startIndex) {
            int index = startIndex, readSizeUnit = realSize / ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
            for (int i = 0; i < readSizeUnit; ++i, index += ARMV82_CHANNEL_UNIT) {
                float16x8_t in = vld1q_f16(inputData + index);
                vst1q_f16(outputData + index, Helper::vecFunc(in));
            }
#endif
            for (; index < endIndex; ++index) {
                outputData[index] = Helper::scalarFunc(inputData[index]);
            }
        }
    } MNN_CONCURRENCY_END();

    return NO_ERROR;
}

ErrorCode Arm82Unary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    ErrorCode code;

    switch (mType) {
        case UnaryOpOperation_ABS:
            code = onExecuteInternal<UnaryAbs>(input, output);
            break;
        case UnaryOpOperation_SQUARE:
            code = onExecuteInternal<UnarySquare>(input, output);
            break;
        case UnaryOpOperation_RSQRT:
            code = onExecuteInternal<UnaryRsqrt>(input, output);
            break;
        case UnaryOpOperation_NEG:
            code = onExecuteInternal<UnaryNeg>(input, output);
            break;
#if defined(__aarch64__)
        case UnaryOpOperation_SQRT:
            code = onExecuteInternal<UnarySqrt>(input, output);
            break;
#endif
        case UnaryOpOperation_RECIPROCAL:
            code = onExecuteInternal<UnaryRecipocal>(input, output);
            break;
        case UnaryOpOperation_HARDSWISH:
            code = onExecuteInternal<UnaryHardSwish>(input, output);
            break;
        default:
            MNN_ASSERT(false);
            break;
    }

    return code;
}

class Arm82UnaryCreator : public Arm82Backend::Arm82Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto type = op->main_as_UnaryOp()->opType();
        std::vector<UnaryOpOperation> supportOps = {
            UnaryOpOperation_ABS, UnaryOpOperation_SQUARE, UnaryOpOperation_RSQRT,
            UnaryOpOperation_NEG, UnaryOpOperation_SQRT, UnaryOpOperation_RECIPROCAL
        };
        if (std::find(supportOps.begin(), supportOps.end(), type) != supportOps.end()) {
            return new Arm82Unary(backend, type);
        }
        return nullptr;
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_UnaryOp, Arm82UnaryCreator);

} // namespace MNN

#endif
