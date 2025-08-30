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

#ifndef FFRT_TASK_WRAPPER_H
#define FFRT_TASK_WRAPPER_H

#include "c/type_def.h"
#include "c/task.h"

typedef struct {
    ffrt_function_header_t header;
    ffrt_function_t func;
    ffrt_function_t after_func;
    void* arg;
} ffrt_function_wrapper_t;

static inline void ffrt_exec_function_wrapper(void* t)
{
    ffrt_function_wrapper_t* f = (ffrt_function_wrapper_t *)t;
    if (f->func) {
        f->func(f->arg);
    }
}

static inline void ffrt_destroy_function_wrapper(void* t)
{
    ffrt_function_wrapper_t* f = (ffrt_function_wrapper_t *)t;
    if (f->after_func) {
        f->after_func(f->arg);
    }
}

#define FFRT_STATIC_ASSERT(cond, msg) int x(int static_assertion_##msg[(cond) ? 1 : -1])
static inline ffrt_function_header_t *ffrt_create_function_wrapper(ffrt_function_t func, ffrt_function_t after_func,
    void* arg, ffrt_function_kind_t kind)
{
    FFRT_STATIC_ASSERT(sizeof(ffrt_function_wrapper_t) <= ffrt_auto_managed_function_storage_size,
        size_of_function_must_be_less_than_ffrt_auto_managed_function_storage_size);

    ffrt_function_wrapper_t* f = (ffrt_function_wrapper_t *)ffrt_alloc_auto_managed_function_storage_base(kind);
    f->header.exec = ffrt_exec_function_wrapper;
    f->header.destroy = ffrt_destroy_function_wrapper;
    f->func = func;
    f->after_func = after_func;
    f->arg = arg;
    return reinterpret_cast<ffrt_function_header_t *>(f);
}

#endif