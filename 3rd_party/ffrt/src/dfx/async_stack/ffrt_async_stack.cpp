/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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
#include "dfx/async_stack/ffrt_async_stack.h"

#include <cstdlib>
#include <mutex>

#include <dlfcn.h>

#include "dfx/log/ffrt_log_api.h"

using FFRTSetStackIdFunc = void(*)(uint64_t stackId);
using FFRTCollectAsyncStackFunc = uint64_t(*)();
namespace {
    FFRTCollectAsyncStackFunc g_collectAsyncStackFunc = nullptr;
    FFRTSetStackIdFunc g_setStackIdFunc = nullptr;
    void* g_asyncStackLibHandle = nullptr;
    bool g_enabledFFRTAsyncStack = false;
}

static void LoadDfxAsyncStackLib()
{
    const char* debuggableEnv = getenv("HAP_DEBUGGABLE");
    if ((debuggableEnv == nullptr) || (strcmp(debuggableEnv, "true") != 0)) {
        return;
    }

    // if async stack is not enabled, the lib should not be unloaded
    g_asyncStackLibHandle = dlopen("libasync_stack.z.so", RTLD_NOW);
    if (g_asyncStackLibHandle == nullptr) {
        return;
    }

    g_collectAsyncStackFunc = reinterpret_cast<FFRTCollectAsyncStackFunc>(dlsym(g_asyncStackLibHandle,
                                                                                "CollectAsyncStack"));
    if (g_collectAsyncStackFunc == nullptr) {
        dlclose(g_asyncStackLibHandle);
        g_asyncStackLibHandle = nullptr;
        return;
    }

    g_setStackIdFunc = reinterpret_cast<FFRTSetStackIdFunc>(dlsym(g_asyncStackLibHandle, "SetStackId"));
    if (g_setStackIdFunc == nullptr) {
        g_collectAsyncStackFunc = nullptr;
        dlclose(g_asyncStackLibHandle);
        g_asyncStackLibHandle = nullptr;
        return;
    }

    g_enabledFFRTAsyncStack = true;
}

static bool IsFFRTAsyncStackEnabled()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, LoadDfxAsyncStackLib);
    return g_enabledFFRTAsyncStack;
}

namespace ffrt {
uint64_t FFRTCollectAsyncStack(void)
{
    if (IsFFRTAsyncStackEnabled() &&
        (g_collectAsyncStackFunc != nullptr)) {
        return g_collectAsyncStackFunc();
    }

    return 0;
}

void FFRTSetStackId(uint64_t stackId)
{
    if (IsFFRTAsyncStackEnabled() &&
        (g_setStackIdFunc != nullptr)) {
        return g_setStackIdFunc(stackId);
    }
}

void CloseAsyncStackLibHandle()
{
    if (g_asyncStackLibHandle != nullptr) {
        dlclose(g_asyncStackLibHandle);
        g_asyncStackLibHandle = nullptr;
    }
}
}