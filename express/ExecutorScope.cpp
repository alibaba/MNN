//
//  ExecutorScope.cpp
//  MNN
//
//  Created by MNN on 2020/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <thread>
#include <mutex>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Scope.hpp>
#include <MNN/expr/ExecutorScope.hpp>

namespace MNN {
namespace Express {

typedef std::shared_ptr<Express::Executor> ExecutorRef;
#if TARGET_OS_IPHONE
#include <pthread.h>
static pthread_key_t gKey;
static std::once_flag gInitFlag;
#else
thread_local static std::once_flag gInitFlag;
thread_local static Scope<ExecutorRef>* g_executor_scope = nullptr;
#endif

static Scope<ExecutorRef>* _getGlobalScope() {
    std::call_once(gInitFlag,
                   [&]() {
#if TARGET_OS_IPHONE
        pthread_key_create(&gKey, NULL);
#else
        thread_local static Scope<ExecutorRef> initValue;
        g_executor_scope = &initValue;
#endif
    });
#if TARGET_OS_IPHONE
    Scope<ExecutorRef>* scope = static_cast<Scope<ExecutorRef>*>(pthread_getspecific(gKey));
    if (!scope) {
        scope = new Scope<ExecutorRef>;
        pthread_setspecific(gKey, static_cast<void*>(scope));
    }
    return scope;
#else
    return g_executor_scope;
#endif
}

ExecutorScope::ExecutorScope(const std::shared_ptr<Executor>& current) {
    _getGlobalScope()->EnterScope(current);
}

ExecutorScope::ExecutorScope(const std::string& scope_name,
                             const std::shared_ptr<Executor>& current) {
    _getGlobalScope()->EnterScope(scope_name, current);
}

ExecutorScope::~ExecutorScope() {
    _getGlobalScope()->ExitScope();
}

const std::shared_ptr<Executor> ExecutorScope::Current() {
    auto exe = _getGlobalScope()->Content();
    if (exe) {
        return exe;
    }
    return Executor::getGlobalExecutor();
}

}  // namespace Express
}  // namespace MNN
