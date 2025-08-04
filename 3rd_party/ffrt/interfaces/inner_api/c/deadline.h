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
#ifndef FFRT_API_C_DEADLINE_H
#define FFRT_API_C_DEADLINE_H
#include <stdint.h>
#include "type_def_ext.h"

FFRT_C_API ffrt_interval_t ffrt_interval_create(uint64_t deadline_us, ffrt_qos_t qos);
FFRT_C_API int ffrt_interval_update(ffrt_interval_t it, uint64_t new_deadline_us);
FFRT_C_API int ffrt_interval_begin(ffrt_interval_t it);
FFRT_C_API int ffrt_interval_end(ffrt_interval_t it);
FFRT_C_API void ffrt_interval_destroy(ffrt_interval_t it);
FFRT_C_API int ffrt_interval_join(ffrt_interval_t it);
FFRT_C_API int ffrt_interval_leave(ffrt_interval_t it);
#endif