//
//  Arm82Relu.cpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/arm82/Arm82Relu.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "core/Macro.h"
#include "MNN_generated.h"


#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

Arm82Relu::Arm82Relu(Backend *backend, const Op *op):Execution(backend){
}


ErrorCode Arm82Relu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    
    auto input = inputs[0];
    auto output = outputs[0];
    const int elementSize = input->elementSize();
    
    const int sizeDivUnit = elementSize / ARMV82_CHANNEL_UNIT;
    const int remainCount = elementSize - sizeDivUnit * ARMV82_CHANNEL_UNIT;
    
    const auto src = input->host<FLOAT16>();
    auto dst = output->host<FLOAT16>();

#ifdef MNN_USE_NEON
    float16x8_t value_0 = vmovq_n_f16(0);
#endif
    
    if(sizeDivUnit > 0){
        for(int i = 0; i < sizeDivUnit; ++i){
            const auto srcPtr = src + i * ARMV82_CHANNEL_UNIT;
            auto dstPtr = dst + i * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
            float16x8_t a = vld1q_f16(srcPtr);
            vst1q_f16(dstPtr, vmaxq_f16(a, value_0));
#else
            for(int i = 0; i < ARMV82_CHANNEL_UNIT; ++i){
                dstPtr[i] = srcPtr[i];
                if(srcPtr[i] < 0){
                    dstPtr[i] = 0;
                }
            }
#endif
        }
    }
    
    if(remainCount > 0){
        for(int i = sizeDivUnit * ARMV82_CHANNEL_UNIT; i < elementSize; ++i){
            dst[i] = src[i];
            if(src[i] < 0){
                dst[i] = 0;
            }
        }
    }
    
    return NO_ERROR;
}

class Arm82ReluCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_Relu();
        if(param->slope() != 0){
            return nullptr;
        }
        return new Arm82Relu(backend, op);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_ReLU, Arm82ReluCreator);


} // namespace MNN
