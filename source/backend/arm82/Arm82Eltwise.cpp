//
//  Arm82Eltwise.cpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82Eltwise.hpp"
#include "Arm82Backend.hpp"
#include "core/Macro.h"
#include "MNN_generated.h"
#include <arm_neon.h>

namespace MNN {

Arm82Eltwise::Arm82Eltwise(Backend *backend, EltwiseType type):Execution(backend), mType(type){

}

static ErrorCode _run(const Tensor* input0, const Tensor* input1, const Tensor* output, int elementSize, EltwiseType mType) {
    
    const int sizeDivUnit = elementSize / ARMV82_CHANNEL_UNIT;
    const int remainCount = elementSize - sizeDivUnit * ARMV82_CHANNEL_UNIT;
    
    const auto src0 = input0->host<FLOAT16>();
    const auto src1 = input1->host<FLOAT16>();
    auto dst = output->host<FLOAT16>();
    
    switch (mType) {
        case EltwiseType_SUM:
        {
            if(sizeDivUnit > 0){
                    for(int i = 0; i < sizeDivUnit; ++i){
                        const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                        const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                        auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
            #ifdef MNN_USE_NEON
                        float16x8_t a = vld1q_f16(src0Ptr);
                        float16x8_t b = vld1q_f16(src1Ptr);
                        vst1q_f16(dstPtr, vaddq_f16(a, b));
            #else
                        for(int i = 0; i < ARMV82_CHANNEL_UNIT; ++i){
                            dstPtr[i] = src0Ptr[i] + src1Ptr[i];
                        }
            #endif
                    }
                }
                
                if(remainCount > 0){
                    for(int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i){
                        dst[i] = src0[i] + src1[i];
                    }
                }
        }
            break;
        case EltwiseType_PROD:
        {
            if(sizeDivUnit > 0){
                    for(int i = 0; i < sizeDivUnit; ++i){
                        const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                        const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                        auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
            #ifdef MNN_USE_NEON
                        float16x8_t a = vld1q_f16(src0Ptr);
                        float16x8_t b = vld1q_f16(src1Ptr);
                        vst1q_f16(dstPtr, vmulq_f16(a, b));
            #else
                        for(int i = 0; i < ARMV82_CHANNEL_UNIT; ++i){
                            dstPtr[i] = src0Ptr[i] * src1Ptr[i];
                        }
            #endif
                    }
                }
                
                if(remainCount > 0){
                    for(int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i){
                        dst[i] = src0[i] * src1[i];
                    }
                }
        }
            break;
        case EltwiseType_SUB:
        {
            if(sizeDivUnit > 0){
                    for(int i = 0; i < sizeDivUnit; ++i){
                        const auto src0Ptr = src0 + i * ARMV82_CHANNEL_UNIT;
                        const auto src1Ptr = src1 + i * ARMV82_CHANNEL_UNIT;
                        auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
            #ifdef MNN_USE_NEON
                        float16x8_t a = vld1q_f16(src0Ptr);
                        float16x8_t b = vld1q_f16(src1Ptr);
                        vst1q_f16(dstPtr, vsubq_f16(a, b));
            #else
                        for(int i = 0; i < ARMV82_CHANNEL_UNIT; ++i){
                            dstPtr[i] = src0Ptr[i] - src1Ptr[i];
                        }
            #endif
                    }
                }
                
                if(remainCount > 0){
                    for(int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i){
                        dst[i] = src0[i] - src1[i];
                    }
                }
        }
            break;
        default:
            return NOT_SUPPORT;
            break;
    }
    
    return NO_ERROR;
}
ErrorCode Arm82Eltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    const int elementSize = ARM82TensorElementSizeHelper(input0);
    _run(input0, input1, output, elementSize, mType);
    for (int i = 2; i < inputs.size(); ++i) {
        _run(output, inputs[i], output, elementSize, mType);
    }
    return NO_ERROR;
}

class Arm82EltwiseCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto eltType = op->main_as_Eltwise()->type();

        if(eltType != EltwiseType_SUM && eltType != EltwiseType_PROD && eltType != EltwiseType_SUB){
            MNN_PRINT("[MNN Warning]Armv82 not support Eltwise type: [%s]\n", MNN::EnumNameEltwiseType(eltType));
            return nullptr;
        }
        return new Arm82Eltwise(backend, eltType);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Eltwise, Arm82EltwiseCreator);


} // namespace MNN

#endif
