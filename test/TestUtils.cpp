//
//  TestUtils.cpp
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TestUtils.h"
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
#include "core/Session.hpp"
#include <MNN/MNNDefine.h>
#include <random>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include "core/TensorUtils.hpp"

using namespace MNN;

Session *createSession(MNN::Interpreter *net, MNNForwardType backend) {
    ScheduleConfig config;
    config.type = backend;
    return net->createSession(config);
}

#if defined(__APPLE__)
void dispatchMetal(std::function<void(MNNForwardType)> payload, MNNForwardType backend);
#endif

void dispatch(std::function<void(MNNForwardType)> payload) {
    for (int i = 0; i < MNN_FORWARD_ALL; i++) {
        MNNForwardType type = (MNNForwardType)i;
        if (MNNGetExtraRuntimeCreator(type))
            dispatch(payload, type);
    }
}

void dispatch(std::function<void(MNNForwardType)> payload, MNNForwardType backend) {
    switch (backend) {
#if defined(__APPLE__)
        case MNN_FORWARD_METAL:
            dispatchMetal(payload, backend);
            break;
#endif
        default:
            payload(backend);
            break;
    }
}
// simulate bf16, prune fp32 tailing precision to bf16 precision
float convertFP32ToBF16(float fp32Value) {
    uint32_t& s32Value = *(uint32_t*)(&fp32Value);
    s32Value &= 0xffff0000;
    return fp32Value;
}

// simulate fp16 in fp32 bits
float convertFP32ToFP16(float fp32Value) {

    uint32_t& u32Result = *(uint32_t*)(&fp32Value);

    uint32_t u32Value = u32Result & 0x7FFFFFFF;  //  digits
    int exp = u32Value >> 23;
    if(exp == 255) {
        return fp32Value;
    }
    u32Result = u32Result & 0x80000000;          // sign
    if(exp > 15 + 127) {
        // inf
        u32Result |= 0x7F800000;
        return fp32Value;
    }

    int g = 0;
    if(exp > -15 + 127) {
        g = (u32Value >> 12) & 1;
        u32Result |= (exp << 23) | (u32Value & (0x3ff << 13));
    }
    else if(exp > -26 + 127) {
        g = (u32Value >> 12) & 1;
        u32Result |= (exp << 23) | (u32Value & (0x3ff << 13));
    }
    u32Result += g << 13;
    return fp32Value;
}



