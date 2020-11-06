//
//  Execution.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Execution.hpp"

namespace MNN {

const Execution::Creator* Execution::searchExtraCreator(const std::string& key, MNNForwardType type) {
    // Depercerate
    return nullptr;
}

bool Execution::insertExtraCreator(std::shared_ptr<Creator> creator, const std::string& key, MNNForwardType type) {
    // Depercerate
    return true;
}

bool Execution::removeExtraCreator(const std::string& key, MNNForwardType type) {
    // Depercerate
    return true;
}
} // namespace MNN
