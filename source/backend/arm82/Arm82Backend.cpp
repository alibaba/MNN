//
//  Arm82Backend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Arm82Backend.hpp"
#include <algorithm>
#include "Arm82Convolution1x1.hpp"
#include "MNN_generated.h"
namespace MNN {
static const MNNForwardType gForwardType = MNN_FORWARD_USER_1;

Arm82Backend::Arm82Backend(int thread) : Backend(gForwardType) {
    auto creator  = MNNGetExtraBackendCreator(MNN_FORWARD_CPU);
    thread        = std::min(thread, 32);
    thread        = std::max(thread, 1);
    mNumberThread = thread;
    MNN_ASSERT(nullptr != creator);
    Backend::Info info;
    info.type      = MNN_FORWARD_CPU;
    info.numThread = thread;
    mCPUBackend.reset(creator->onCreate(info));
}
Execution* Arm82Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    switch (op->type()) {
        case OpType_Convolution:
            if (Arm82Convolution1x1::support(op)) {
                return new Arm82Convolution1x1(inputs, outputs, op, this);
            }
            break;
        default:
            break;
    }
    return mCPUBackend->onCreate(inputs, outputs, op);
}

class Arm82BackendCreator : public BackendCreator {
public:
    virtual Backend* onCreate(const Backend::Info& info) const override {
        return new Arm82Backend(info.numThread);
    };
};

static bool registerCPUBackendCreator = []() {
    static Arm82BackendCreator creator;
    MNNInsertExtraBackendCreator(gForwardType, &creator);
    return true;
}();
} // namespace MNN
