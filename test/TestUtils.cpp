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
