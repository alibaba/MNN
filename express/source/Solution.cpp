//
//  Solution.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Solution.hpp"
namespace MNN {
namespace Express {
Solution::Requirement Solution::onGetRequirement() const {
    auto size = mInputSize;
    Solution::Requirement req;
    req.contentNeedContent.resize(size);
    req.shapeNeedContent.resize(size);
    req.supportError.resize(size);
    for (int i = 0; i < size; ++i) {
        req.contentNeedContent[i] = true;
        req.shapeNeedContent[i]   = false;
        req.supportError[i] = false;
    }
    return req;
}
} // namespace Express
} // namespace MNN
