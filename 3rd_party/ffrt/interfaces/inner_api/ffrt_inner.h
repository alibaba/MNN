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
#ifndef FFRT_API_FFRT_INNER_H
#define FFRT_API_FFRT_INNER_H
#include "ffrt.h"
#ifdef __cplusplus
#include "c/ffrt_dump.h"
#include "c/queue_ext.h"
#include "cpp/thread.h"
#include "cpp/future.h"
#include "cpp/task_ext.h"
#include "cpp/deadline.h"
#include "cpp/qos_convert.h"
#else
#include "c/task_ext.h"
#include "c/queue_ext.h"
#include "c/thread.h"
#include "c/executor_task.h"
#include "c/ffrt_dump.h"
#include "c/deadline.h"
#include "c/ffrt_cpu_boost.h"
#include "c/ffrt_ipc.h"
#include "c/init.h"
#endif
#endif
