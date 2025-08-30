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
 * @since 18
 */

/**
 * @file shared_mutex.h
 *
 * @brief Declares the shared mutex interfaces in C.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 18
 */

#ifndef FFRT_API_C_SHARED_MUTEX_H
#define FFRT_API_C_SHARED_MUTEX_H

#include "type_def.h"

/**
 * @brief Initializes a rwlock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @param attr Indicates a pointer to the rwlock attribute.
 * @return Returns <b>ffrt_success</b> if the rwlock is initialized;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_init(ffrt_rwlock_t* rwlock, const ffrt_rwlockattr_t* attr);

/**
 * @brief Locks a write lock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @return Returns <b>ffrt_success</b> if the rwlock is locked;
           returns <b>ffrt_error_inval</b> or blocks the calling thread otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_wrlock(ffrt_rwlock_t* rwlock);

/**
 * @brief Attempts to lock a write lock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @return Returns <b>ffrt_success</b> if the rwlock is locked;
           returns <b>ffrt_error_inval</b> or <b>ffrt_error_busy</b> otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_trywrlock(ffrt_rwlock_t* rwlock);

/**
 * @brief Locks a read lock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @return Returns <b>ffrt_success</b> if the rwlock is locked;
           returns <b>ffrt_error_inval</b> or blocks the calling thread otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_rdlock(ffrt_rwlock_t* rwlock);

/**
 * @brief Attempts to lock a read lock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @return Returns <b>ffrt_success</b> if the rwlock is locked;
           returns <b>ffrt_error_inval</b> or <b>ffrt_error_busy</b> otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_tryrdlock(ffrt_rwlock_t* rwlock);

/**
 * @brief Unlocks a rwlock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @return Returns <b>ffrt_success</b> if the rwlock is unlocked;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_unlock(ffrt_rwlock_t* rwlock);

/**
 * @brief Destroys a rwlock.
 *
 * @param rwlock Indicates a pointer to the rwlock.
 * @return Returns <b>ffrt_success</b> if the rwlock is destroyed;
           returns <b>ffrt_error_inval</b> otherwise.
 * @since 18
 */
FFRT_C_API int ffrt_rwlock_destroy(ffrt_rwlock_t* rwlock);

#endif // FFRT_API_C_SHARED_MUTEX_H
/** @} */