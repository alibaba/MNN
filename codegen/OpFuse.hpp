//
//  OpFused.hpp
//  MNN
//
//  Created by MNN on 2020/9/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
    bool opFuse(std::vector<Schedule::OpCacheInfo>& infos, MNNForwardType type);
} // namespace MNN

