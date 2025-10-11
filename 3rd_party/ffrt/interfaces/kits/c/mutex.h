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

/**
 * @addtogroup FFRT
 * @{
 *
 * @brief Provides FFRT C APIs.
 *
 * @since 10
 */

/**
 * @file mutex.h
 *
 * @brief Declares the mutex interfaces in C.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_C_MUTEX_H
#define FFRT_API_C_MUTEX_H

#include "type_def.h"

/**
 * @brief Initializes a mutex attribute.
 *
 * @param attr Indicates a pointer to the mutex attribute.
 * @return Returns <b>ffrt_success</b> if the mutex attribute is initialized;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 12
 */
FFRT_C_API int ffrt_mutexattr_init(ffrt_mutexattr_t* attr);

/**
 * @brief Sets the type of a mutex attribute.
 *
 * @param attr Indicates a pointer to the mutex attribute.
 * @param type Indicates a int to the mutex type.
 * @return Returns <b>ffrt_success</b> if the mutex attribute type is set successfully;
           returns <b>ffrt_error_inval</b> if <b>attr</b> is a null pointer or
           the mutex attribute type is not <b>ffrt_mutex_normal</b> or <b>ffrt_mutex_recursive</b>.
 * @since 12
 */
FFRT_C_API int ffrt_mutexattr_settype(ffrt_mutexattr_t* attr, int type);

/**
 * @brief Gets the type of a mutex attribute.
 *
 * @param attr Indicates a pointer to the mutex attribute.
 * @param type Indicates a pointer to the mutex type.
 * @return Returns <b>ffrt_success</b> if the mutex attribute type is get successfully;
           returns <b>ffrt_error_inval</b> if <b>attr</b> or <b>type</b> is a null pointer.
 * @since 12
 */
FFRT_C_API int ffrt_mutexattr_gettype(ffrt_mutexattr_t* attr, int* type);

/**
 * @brief Destroys a mutex attribute, the user needs to invoke this interface.
 *
 * @param attr Indicates a pointer to the mutex attribute.
 * @return Returns <b>ffrt_success</b> if the mutex attribute is destroyed;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 12
 */
FFRT_C_API int ffrt_mutexattr_destroy(ffrt_mutexattr_t* attr);

/**
 * @brief Initializes a mutex.
 *
 * @param mutex Indicates a pointer to the mutex.
 * @param attr Indicates a pointer to the mutex attribute.
 * @return Returns <b>ffrt_success</b> if the mutex is initialized;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_mutex_init(ffrt_mutex_t* mutex, const ffrt_mutexattr_t* attr);

/**
 * @brief Locks a mutex.
 *
 * @param mutex Indicates a pointer to the mutex.
 * @return Returns <b>ffrt_success</b> if the mutex is locked;
           returns <b>ffrt_error_inval</b> or blocks the calling thread otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_mutex_lock(ffrt_mutex_t* mutex);

/**
 * @brief Unlocks a mutex.
 *
 * @param mutex Indicates a pointer to the mutex.
 * @return Returns <b>ffrt_success</b> if the mutex is unlocked;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_mutex_unlock(ffrt_mutex_t* mutex);

/**
 * @brief Attempts to lock a mutex.
 *
 * @param mutex Indicates a pointer to the mutex.
 * @return Returns <b>ffrt_success</b> if the mutex is locked;
           returns <b>ffrt_error_inval</b> or <b>ffrt_error_busy</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_mutex_trylock(ffrt_mutex_t* mutex);

/**
 * @brief Destroys a mutex, the user needs to invoke this interface.
 *
 * @param mutex Indicates a pointer to the mutex.
 * @return Returns <b>ffrt_success</b> if the mutex is destroyed;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_mutex_destroy(ffrt_mutex_t* mutex);

#endif // FFRT_API_C_MUTEX_H
/** @} */