//
//  TRTSoFinder.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTSoFinder.hpp"
#include <dlfcn.h>
#include <core/TensorUtils.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace MNN {

using namespace std;
static inline std::string join(const std::string& part1, const std::string& part2) {
    const char sep = '/';
    if (!part2.empty() && part2.front() == sep) {
        return part2;
    }
    std::string ret;
    ret.reserve(part1.size() + part2.size() + 1);
    ret = part1;
    if (!ret.empty() && ret.back() != sep) {
        ret += sep;
    }
    ret += part2;
    return ret;
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path, int dynload_flags) {
    MNN_PRINT("Try to find library path : %s \n", dso_path.c_str());
    void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);
    if (nullptr == dso_handle) {
        MNN_PRINT("dlopen path : %s error !!! \n", dso_path.c_str());
    }
    return dso_handle;
}

static inline void* GetDsoHandleFromSearchPath(const std::string& search_root, const std::string& dso_name,
                                               bool throw_on_error = true) {
    int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
    void* dso_handle  = nullptr;

    std::string dlPath = dso_name;
    if (search_root.empty()) {
        dso_handle = GetDsoHandleFromDefaultPath(dlPath, dynload_flags);
    } else {
        dlPath       = join(search_root, dso_name);
        dso_handle   = dlopen(dlPath.c_str(), dynload_flags);
        auto errorno = dlerror();
        if (nullptr == dso_handle) {
            MNN_PRINT("Failed to find dynamic library: %s  ( %s ) \n", dlPath.c_str(), errorno);
            dlPath     = dso_name;
            dso_handle = GetDsoHandleFromDefaultPath(dlPath, dynload_flags);
        }
    }

    auto errorno = dlerror();
    if (throw_on_error) {
        MNN_ASSERT(nullptr != dso_handle);
    } else if (nullptr == dso_handle) {
        MNN_PRINT("nullptr == dso_handle \n");
    }

    return dso_handle;
}

void* GetTRTDsoHandle() {
    string FLAGS_tensorrt_dir = "/lib/x86_64-linux-gnu/";
    return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.so");
}

} // namespace MNN