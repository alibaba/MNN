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
#ifndef FFRT_CPU_BOOST_C_API_H
#define FFRT_CPU_BOOST_C_API_H
#include "type_def_ext.h"

#define CPUBOOST_START_POINT  0
#define CPUBOOST_MAX_CNT      32

FFRT_C_API int ffrt_cpu_boost_start(int ctx_id);

FFRT_C_API int ffrt_cpu_boost_end(int ctx_id);

#endif // FFRT_CPU_BOOST_C_API_H