//
//  MergeOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MergeOptimizer.hpp"
#include <map>
#include "Utils.hpp"

namespace MNN {
namespace Express {

MergeOptimizer::MergeOptimizer(MNNForwardType type, int numberThread, BackendConfig* config) {
    if (nullptr != config) {
        mConfig = *config;
    }
    mType         = type;
    mNumberThread = numberThread;
}

Optimizer::Cost MergeOptimizer::onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    Cost cost;
    cost.compute = 0.0f;
    cost.memory  = 0.0f;
    return cost;
}
bool MergeOptimizer::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    // Deceperate
    return true;
}
} // namespace Express
} // namespace MNN
