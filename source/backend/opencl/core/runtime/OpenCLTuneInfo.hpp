//
//  OpenCLTuneInfo.hpp
//  MNN
//
//  Created by MNN on 2021/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLTuneInfo_hpp
#define OpenCLTuneInfo_hpp
#include "CLCache_generated.h"
namespace MNN {
namespace OpenCL {
struct TuneInfo {
    std::vector<std::unique_ptr<CLCache::OpInfoT>> mInfos;
};
}
}

#endif
