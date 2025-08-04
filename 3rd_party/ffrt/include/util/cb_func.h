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

#ifndef FFRT_CB_FUNC_H_
#define FFRT_CB_FUNC_H_
#include <functional>
#include <vector>

template <typename T>
struct SingleInsCB {
    using Instance = std::function<T &()>;
};

template <typename T>
struct TaskAllocCB {
    using Alloc = std::function<T *()>;
    using Free = std::function<void (T *)>;
    using Free_ = std::function<void (T *)>;
    using GetUnfreedMem = std::function<std::vector<void *> ()>;
    using HasBeenFreed = std::function<bool (T *)>;
    using LockMem = std::function<void ()>;
    using UnlockMem = std::function<void ()>;
};

#endif /* FFRT_CB_FUNC_H_ */