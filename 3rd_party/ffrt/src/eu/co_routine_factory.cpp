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
#include "eu/co_routine_factory.h"
#include "util/slab.h"

namespace ffrt {

CoRoutine *CoRoutineAllocMem(std::size_t stack_size)
{
    return ffrt::QSimpleAllocator<CoRoutine>::AllocMem(stack_size);
}

void CoRoutineFreeMem(CoRoutine *co)
{
    ffrt::QSimpleAllocator<CoRoutine>::FreeMem(co);
}

void CoRoutineReleaseMem()
{
    ffrt::QSimpleAllocator<CoRoutine>::releaseMem();
}

void CoRoutineInstance(std::size_t size)
{
    QSimpleAllocator<CoRoutine>::Instance(size);
}

}