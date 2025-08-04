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
#include "sched/execute_ctx.h"
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>

pthread_key_t g_executeCtxTlsKey = 0;
pthread_once_t g_executeCtxKeyOnce = PTHREAD_ONCE_INIT;
namespace ffrt {
namespace {
void ExecuteCtxTlsDestructor(void* args)
{
    auto ctx = static_cast<ExecuteCtx*>(args);
    if (ctx) {
        delete ctx;
    }
}

void MakeExecuteCtxTlsKey()
{
    pthread_key_create(&g_executeCtxTlsKey, ExecuteCtxTlsDestructor);
}
}

ExecuteCtx::ExecuteCtx()
{
    task = nullptr;
    tid = syscall(SYS_gettid);
}

ExecuteCtx::~ExecuteCtx()
{
}

ExecuteCtx* ExecuteCtx::Cur(bool init)
{
    ExecuteCtx* ctx = nullptr;
    pthread_once(&g_executeCtxKeyOnce, MakeExecuteCtxTlsKey);

    void *curTls = pthread_getspecific(g_executeCtxTlsKey);
    if (curTls != nullptr) {
        ctx = reinterpret_cast<ExecuteCtx *>(curTls);
    } else if (init) {
        ctx = new ExecuteCtx();
        pthread_setspecific(g_executeCtxTlsKey, ctx);
    }
    return ctx;
}

} // namespace ffrt