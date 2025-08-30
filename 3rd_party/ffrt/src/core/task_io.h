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

#ifndef FFRT_TASK_IO_H
#define FFRT_TASK_IO_H

#include "internal_inc/types.h"
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "c/executor_task.h"

namespace ffrt {
typedef struct {
    ffrt_coroutine_ptr_t exec;
    ffrt_function_t destroy;
    void* data;
} ffrt_io_callable_t;
} /* namespace ffrt */
#endif