/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

/**
 * @addtogroup FFRT
 * @{
 *
 * @brief Provides FFRT C APIs.
 *
 * @since 20
 */

/**
 * @file fiber.h
 *
 * @brief Declares the fiber interfaces in C.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 20
 */

#ifndef FFRT_API_C_FIBER_H
#define FFRT_API_C_FIBER_H

#include "type_def.h"

/**
 * @brief Initializes a fiber.
 *
 * This function initializes a fiber structure, preparing it for execution.
 *
 * @param fiber Indicates the pointer to the fiber structure to be initialized.
 * @param func Indicates the entry point function that the fiber will execute.
 * @param arg Indicates the argument to be passed to the entry point function.
 * @param stack Indicates the pointer to the memory region to be used as the fiber's stack.
 * @param stack_size Indicates the size of the stack in bytes.
 * @return Returns <b>ffrt_success</b> if the fiber is initialized;
           returns <b>ffrt_error</b> otherwise.
 * @since 20
 */
FFRT_C_API int ffrt_fiber_init(ffrt_fiber_t* fiber, void(*func)(void*), void* arg, void* stack, size_t stack_size);


/**
 * @brief Switch execution context between two fibers.
 *
 * Switches the execution context by saving the current context into the fiber specified
 * by @c from and restoring the context from the fiber specified by @c to.
 *
 * @param from Indicates the pointer to the fiber into which the current context will be saved.
 * @param to Indicates the pointer to the fiber from which the context will be restored.
 * @since 20
 */
FFRT_C_API void ffrt_fiber_switch(ffrt_fiber_t* from, ffrt_fiber_t* to);

#endif // FFRT_API_C_FIBER_H
/** @} */