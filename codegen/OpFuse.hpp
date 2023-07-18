//
//  OpFused.hpp
//  MNN
//
//  Created by MNN on 2020/9/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputerUtils.hpp"
#include <map>
namespace MNN {
    bool opFuse(std::vector<Schedule::OpCacheInfo>& infos, MNNForwardType type, BackendConfig::PrecisionMode precision);
} // namespace MNN

