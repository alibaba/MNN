//
//  CPUBackend.cpp
//  MNN
//
//  Created by MNN on 2019/05/08.
//  Copyright © 2019, Alibaba Group Holding Limited
//

#include <mutex>

namespace MNN {
extern void registerCPUBackendCreator();
#ifdef MNN_BUILD_METAL
extern void registerMetalBackendCreator();
#endif
void registBackend() {
    static std::once_flag s_flag;
    std::call_once(s_flag, [&]() {
        registerCPUBackendCreator();
#ifdef MNN_BUILD_METAL
        registerMetalBackendCreator();
#endif
    });
}
} // namespace MNN
