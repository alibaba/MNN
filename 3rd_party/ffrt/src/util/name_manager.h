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
#ifndef __NAME_MANAGER_H__
#define __NAME_MANAGER_H__

namespace ffrt {
#ifdef OHOS_STANDARD_SYSTEM
static const char* CPU_MONITOR_NAME = "OS_FFRT_Monitor";
static const char* WORKER_THREAD_NAME_PREFIX = "OS_FFRT_";
static const char* WORKER_THREAD_SYMBOL = "_";
static const char* DELAYED_WORKER_NAME = "OS_FFRT_Delay";
static const char* IO_POLLER_NAME = "OS_FFRT_IO";
#else
static const char* CPU_MONITOR_NAME = "ffrt_moniotor";
static const char* WORKER_THREAD_NAME_PREFIX = "ffrtwk/CPU-";
static const char* WORKER_THREAD_SYMBOL = "-";
static const char* DELAYED_WORKER_NAME = "delayed_worker";
static const char* IO_POLLER_NAME = "ffrt_io";
#endif
} // namespace ffrt
#endif // __NAME_MANAGER_H__
