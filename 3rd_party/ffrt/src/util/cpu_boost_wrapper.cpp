/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <dlfcn.h>
#include "dfx/log/ffrt_log_api.h"
#include "cpu_boost_wrapper.h"

namespace ffrt {
int cpu_boost_start(int ctx_id);
int cpu_boost_end(int ctx_id);
int cpu_boost_save(int ctx_id);
int cpu_boost_restore(int ctx_id);
constexpr const char* CPU_BOOST_LIB_PATH = "lib_cpuboost.so";
class CPUBoostAdapter {
public:
    CPUBoostAdapter()
    {
        Load();
    }

    ~CPUBoostAdapter()
    {
        UnLoad();
    }

    static CPUBoostAdapter* Instance()
    {
        static CPUBoostAdapter instance;
        return &instance;
    }

#define REG_FUNC(func) using func##Type = decltype(func)*; func##Type func##Temp = nullptr
    REG_FUNC(cpu_boost_start);
    REG_FUNC(cpu_boost_end);
    REG_FUNC(cpu_boost_save);
    REG_FUNC(cpu_boost_restore);
#undef REG_FUNC

private:
    bool Load()
    {
        if (handle != nullptr) {
            FFRT_SYSEVENT_LOGD("handle exits");
            return true;
        }

        handle = dlopen(CPU_BOOST_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            FFRT_SYSEVENT_LOGE("load so[%s] fail", CPU_BOOST_LIB_PATH);
            return false;
        }

        bool loadFlag = true;

#define LOAD_FUNC(x) x##Temp = reinterpret_cast<x##Type>(dlsym(handle, #x)); \
        if (x##Temp == nullptr) { \
            FFRT_SYSEVENT_LOGE("load func %s from %s failed", #x, CPU_BOOST_LIB_PATH); \
            loadFlag = false; \
        }
            LOAD_FUNC(cpu_boost_start);
            LOAD_FUNC(cpu_boost_end);
            LOAD_FUNC(cpu_boost_save);
            LOAD_FUNC(cpu_boost_restore);
#undef LOAD_FUNC
        return loadFlag;
    }

    bool UnLoad()
    {
        if (handle != nullptr) {
            if (dlclose(handle) != 0) {
                FFRT_SYSEVENT_LOGE("failed to close the handle");
                return false;
            }
            handle = nullptr;
            return true;
        }
        return true;
    }

    void* handle = nullptr;
};
}

#define EXECUTE_CPU_BOOST_FUNC(x, ctxId, ret) auto func = ffrt::CPUBoostAdapter::Instance()->x##Temp; \
        if (func != nullptr) { \
            ret = (func)(ctxId); \
        } else { \
            ret = -1; \
        }

int CpuBoostStart(int ctxId)
{
    int ret = 0;
    EXECUTE_CPU_BOOST_FUNC(cpu_boost_start, ctxId, ret);
    return ret;
}

int CpuBoostEnd(int ctxId)
{
    int ret = 0;
    EXECUTE_CPU_BOOST_FUNC(cpu_boost_end, ctxId, ret);
    return ret;
}

int CpuBoostSave(int ctxId)
{
    int ret = 0;
    EXECUTE_CPU_BOOST_FUNC(cpu_boost_save, ctxId, ret);
    return ret;
}

int CpuBoostRestore(int ctxId)
{
    int ret = 0;
    EXECUTE_CPU_BOOST_FUNC(cpu_boost_restore, ctxId, ret);
    return ret;
}