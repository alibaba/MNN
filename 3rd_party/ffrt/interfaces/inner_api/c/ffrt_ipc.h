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
#ifndef FFRT_IPC_H
#define FFRT_IPC_H
#include "type_def_ext.h"

/**
 * @brief ipc set legacy mode to ffrt.
 *
 * @param mode Indicates wheather use legacy mode.
 * @since 10
 */
FFRT_C_API void ffrt_this_task_set_legacy_mode(bool mode);

#endif