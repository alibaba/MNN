//
//  Optimizer.cpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Optimizer.hpp>
#include "MergeOptimizer.hpp"
#include "core/Backend.hpp"
namespace MNN {
namespace Express {
Optimizer::Parameters::Parameters(int n) {
    MNN_ASSERT(n > 0);
    mValue = new float[n];
    mSize  = n;
}
Optimizer::Parameters::~Parameters() {
    if (nullptr != mValue) {
        delete[] mValue;
    }
}
std::shared_ptr<Optimizer> Optimizer::create(Config config) {
    const int numThread = config.numThread;
    auto forwardType = config.forwardType;
    if (forwardType != MNN_FORWARD_ALL) {
        if (MNNGetExtraBackendCreator(forwardType) == nullptr) {
            return nullptr;
        }
        return std::shared_ptr<Optimizer>(new MergeOptimizer(config.forwardType, numThread, nullptr));
    }

    auto device = config.device;
    if (CPU == device) {
        return std::shared_ptr<Optimizer>(new MergeOptimizer(MNN_FORWARD_CPU, numThread, nullptr));
    }
    if (GPU == device) {
        std::vector<MNNForwardType> types {MNN_FORWARD_METAL, MNN_FORWARD_OPENCL, MNN_FORWARD_VULKAN, MNN_FORWARD_OPENGL};
        for (auto type : types) {
            auto creator = MNNGetExtraBackendCreator(type);
            if (nullptr != creator) {
                return std::shared_ptr<Optimizer>(new MergeOptimizer(type, numThread, nullptr));
            }
        }
    }
    return nullptr;
}

} // namespace Express
} // namespace MNN
