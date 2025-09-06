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
#include "tm/task_factory.h"
#include "tm/cpu_task.h"
#include "tm/queue_task.h"
#include "tm/io_task.h"
#include "tm/uv_task.h"

namespace ffrt {
template <typename T>
TaskFactory<T>& TaskFactory<T>::Instance()
{
    static TaskFactory<T> fac;
    return fac;
}

template class TaskFactory<CPUEUTask>;
template class TaskFactory<QueueTask>;
template class TaskFactory<IOTask>;
template class TaskFactory<UVTask>;
} // namespace ffrt