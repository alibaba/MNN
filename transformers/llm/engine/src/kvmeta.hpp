//
//  kvmeta.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef KVMATE_hpp
#define KVMATE_hpp

#include <vector>

namespace MNN {
using namespace Express;
namespace Transformer {

struct KVMeta {
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::vector<int> reserveHost;
    void sync();
};

}
}
#endif // KVMATE_hpp