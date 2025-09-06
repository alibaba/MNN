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

#ifndef __FFRT_EVENT_HANDLER_ADAPTER_H__
#define __FFRT_EVENT_HANDLER_ADAPTER_H__
#include <dlfcn.h>
#include <string>
#include <mutex>
#include "dfx/log/ffrt_log_api.h"
#include "c/type_def.h"

namespace ffrt {
#if (defined(__aarch64__) || defined(__x86_64__))
constexpr const char* EVENTHANDLER_LIB_PATH = "libeventhandler.z.so";
#else
constexpr const char* EVENTHANDLER_LIB_PATH = "libeventhandler.z.so";
#endif

enum class Priority : uint32_t {
    // The highest priority queue, should be distributed until the tasks in the queue are completed.
    VIP = 0,
    // Event that should be distributed at once if possible.
    IMMEDIATE,
    // High priority event, sorted by handle time, should be distributed before low priority event.
    HIGH,
    // Normal event, sorted by handle time.
    LOW,
    // Event that should be distributed only if no other event right now.
    IDLE,
};

struct TaskOptions {
    std::string dfxName_;
    int64_t delayTime_;
    Priority priority_;
    uintptr_t taskId_;
    TaskOptions(std::string dfxName, int64_t delayTime, Priority priority, uintptr_t taskId)
        : dfxName_(dfxName), delayTime_(delayTime), priority_(priority), taskId_(taskId) {}
};

void* GetMainEventHandlerForFFRT();
void* GetCurrentEventHandlerForFFRT();
bool PostTaskByFFRT(void* handler, const std::function<void()>& callback, const TaskOptions& task);
int RemoveTaskForFFRT(void* handler, const uintptr_t taskId);
int AddFdListenerByFFRT(void* handler, uint32_t fd, uint32_t event, void* data, ffrt_poller_cb cb);
int RemoveFdListenerByFFRT(void* handler, uint32_t fd);

using GetMainEventHandlerType = decltype(GetMainEventHandlerForFFRT)*;
using GetCurrentEventHandlerType = decltype(GetCurrentEventHandlerForFFRT)*;
using PostTaskType = decltype(PostTaskByFFRT)*;
using RemoveTaskType = decltype(RemoveTaskForFFRT)*;
using AddFdListenerType = decltype(AddFdListenerByFFRT)*;
using RemoveFdListenerType = decltype(RemoveFdListenerByFFRT)*;

class EventHandlerAdapter {
public:
    EventHandlerAdapter()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        Load();
    }

    ~EventHandlerAdapter()
    {
    }

    static EventHandlerAdapter* Instance()
    {
        static EventHandlerAdapter instance;
        return &instance;
    }

    Priority ConvertPriority(ffrt_queue_priority_t priority)
    {
        return static_cast<Priority>(priority + 1);
    }

    GetMainEventHandlerType GetMainEventHandler = nullptr;
    GetCurrentEventHandlerType GetCurrentEventHandler = nullptr;
    PostTaskType PostTask = nullptr;
    RemoveTaskType RemoveTask = nullptr;
    AddFdListenerType AddFdListener = nullptr;
    RemoveFdListenerType RemoveFdListener = nullptr;

private:
    void Load()
    {
        if (handle_ != nullptr) {
            return;
        }

        handle_ = dlopen(EVENTHANDLER_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            FFRT_LOGE("eventhandler lib handle is null.");
            return;
        }

        GetMainEventHandler = reinterpret_cast<GetMainEventHandlerType>(
            dlsym(handle_, "GetMainEventHandlerForFFRT"));
        if (GetMainEventHandler == nullptr) {
            FFRT_LOGE("get GetMainEventHandlerForFFRT symbol fail.");
            return;
        }

        GetCurrentEventHandler = reinterpret_cast<GetCurrentEventHandlerType>(
            dlsym(handle_, "GetCurrentEventHandlerForFFRT"));
        if (GetCurrentEventHandler == nullptr) {
            FFRT_LOGE("get GetCurrentEventHandlerForFFRT symbol fail.");
            return;
        }

        PostTask = reinterpret_cast<PostTaskType>(
            dlsym(handle_, "PostTaskByFFRT"));
        if (PostTask == nullptr) {
            FFRT_LOGE("get PostTaskByFFRT symbol fail.");
            return;
        }

        RemoveTask = reinterpret_cast<RemoveTaskType>(
            dlsym(handle_, "RemoveTaskForFFRT"));
        if (RemoveTask == nullptr) {
            FFRT_LOGE("get RemoveTaskForFFRT symbol fail.");
            return;
        }

        AddFdListener = reinterpret_cast<AddFdListenerType>(
            dlsym(handle_, "AddFdListenerByFFRT"));
        if (AddFdListener == nullptr) {
            FFRT_LOGE("get AddFdListenerByFFRT symbol fail.");
            return;
        }
         
        RemoveFdListener = reinterpret_cast<RemoveFdListenerType>(
            dlsym(handle_, "RemoveFdListenerByFFRT"));
        if (RemoveFdListener == nullptr) {
            FFRT_LOGE("get RemoveFdListenerByFFRT symbol fail.");
            return;
        }
    }

    void* handle_ = nullptr;
    std::mutex mutex_;
};
}  // namespace ffrt
#endif // __FFRT_EVENT_HANDLER_ADAPTER_H__