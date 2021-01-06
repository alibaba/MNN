//
//  Arm82Binary.cpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2021, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "backend/arm82/Arm82Binary.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "core/Macro.h"
#include "MNN_generated.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

static void _MNNArm82BinaryAdd(FLOAT16 *dst, const FLOAT16 *src0, const FLOAT16 *src1, const int &elementSize, const int &needBroadcastIndex) {
    const int sizeDivUnit = elementSize / ARMV82_CHANNEL_UNIT;
    const int remainCount = elementSize - sizeDivUnit * ARMV82_CHANNEL_UNIT;

    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                float16x8_t a = vld1q_f16(src0Ptr);
                float16x8_t b = vld1q_f16(src1Ptr);
                vst1q_f16(dstPtr, vaddq_f16(a, b));
#else
                for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                    dstPtr[i] = src0Ptr[i] + src1Ptr[i];
                }
#endif
            }
        }
        
        if (remainCount > 0) {
            for (int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i) {
                dst[i] = src0[i] + src1[i];
            }
        }
    } else if (0 == needBroadcastIndex) {
        const FLOAT16 srcValue0 = src0[0];
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                float16x8_t a = vmovq_n_f16(srcValue0);
                float16x8_t b = vld1q_f16(src1Ptr);
                vst1q_f16(dstPtr, vaddq_f16(a, b));
#else
                for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                    dstPtr[i] = srcValue0 + src1Ptr[i];
                }
#endif
            }
        }
        
        if (remainCount > 0) {
            for (int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i) {
                dst[i] = srcValue0 + src1[i];
            }
        }
    } else {
        const FLOAT16 srcValue1 = src1[0];
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                float16x8_t a = vld1q_f16(src0Ptr);
                float16x8_t b = vmovq_n_f16(srcValue1);
                vst1q_f16(dstPtr, vaddq_f16(a, b));
#else
                for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                    dstPtr[i] = src0Ptr[i] + srcValue1;
                }
#endif
            }
        }
        
        if (remainCount > 0) {
            for (int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i) {
                dst[i] = src0[i] + srcValue1;
            }
        }
    }
}

static void _MNNArm82BinarySub(FLOAT16 *dst, const FLOAT16 *src0, const FLOAT16 *src1, const int &elementSize, const int &needBroadcastIndex) {
    const int sizeDivUnit = elementSize / ARMV82_CHANNEL_UNIT;
    const int remainCount = elementSize - sizeDivUnit * ARMV82_CHANNEL_UNIT;
    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                float16x8_t a = vld1q_f16(src0Ptr);
                float16x8_t b = vld1q_f16(src1Ptr);
                vst1q_f16(dstPtr, vsubq_f16(a, b));
#else
                for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                    dstPtr[i] = src0Ptr[i] - src1Ptr[i];
                }
#endif
            }
        }
        
        if (remainCount > 0) {
            for (int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i) {
                dst[i] = src0[i] - src1[i];
            }
        }
    } else if (0 == needBroadcastIndex) {
        const FLOAT16 srcValue0 = src0[0];
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                float16x8_t a = vmovq_n_f16(srcValue0);
                float16x8_t b = vld1q_f16(src1Ptr);
                vst1q_f16(dstPtr, vsubq_f16(a, b));
#else
                for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                    dstPtr[i] = srcValue0 - src1Ptr[i];
                }
#endif
            }
        }
        
        if (remainCount > 0) {
            for (int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i) {
                dst[i] = srcValue0 - src1[i];
            }
        }
    } else {
        const FLOAT16 srcValue1 = src1[0];
        if (sizeDivUnit > 0){
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                float16x8_t a = vld1q_f16(src0Ptr);
                float16x8_t b = vmovq_n_f16(srcValue1);
                vst1q_f16(dstPtr, vsubq_f16(a, b));
#else
                for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                    dstPtr[i] = src0Ptr[i] - srcValue1;
                }
#endif
            }
        }
        
        if (remainCount > 0) {
            for (int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i) {
                dst[i] = src0[i] - srcValue1;
            }
        }
    }
}

Arm82BinaryFloat::Arm82BinaryFloat(Backend *backend, int32_t type):Execution(backend), mType(type) {

}

ErrorCode Arm82BinaryFloat::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    const int input0DataCount = inputs[0]->elementSize();
    const int input1DataCount = inputs[1]->elementSize();
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
    } else {
        mNeedBroadcastIndex = 1;
    }
    return NO_ERROR;
}

ErrorCode Arm82BinaryFloat::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];

    const int elementSize = ARM82TensorElementSizeHelper(input0);
    
    const auto src0 = input0->host<FLOAT16>();
    const auto src1 = input1->host<FLOAT16>();
    auto dst = output->host<FLOAT16>();
    
    switch (mType) {
        case BinaryOpOperation_ADD:
            _MNNArm82BinaryAdd(dst, src0, src1, elementSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SUB:
            _MNNArm82BinarySub(dst, src0, src1, elementSize, mNeedBroadcastIndex);
            break;
        default:
            return NOT_SUPPORT;
            break;
    }
    return NO_ERROR;
}

class Arm82BinaryCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        int32_t type = op->main_as_BinaryOp()->opType();
        auto dataType = inputs[0]->getType();
        if (dataType.bits == 32) {
            if (dataType.code == halide_type_float) {
                if (type != BinaryOpOperation_ADD && type != BinaryOpOperation_SUB) {
//                    MNN_ERROR("Arm82 BinaryOp: unsupported data type (bits: %d, code: %d)\n",
//                              dataType.bits, dataType.code);
                    return nullptr;
                }
                if (inputs[0]->elementSize() == inputs[1]->elementSize() || 
                    (inputs[0]->elementSize() == 1 && inputs[1]->elementSize() != 1) ||
                    (inputs[0]->elementSize() != 1 && inputs[1]->elementSize() == 1)) {
                    return new Arm82BinaryFloat(backend, type);
                }
            }
        }
        MNN_ERROR("Arm82 BinaryOp: unsupported data type (bits: %d, code: %d)\n",
                  dataType.bits, dataType.code);
        return nullptr;
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_BinaryOp, Arm82BinaryCreator);


} // namespace MNN

#endif
