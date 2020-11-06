//
//  ExecutorScope.cpp
//  MNN
//
//  Created by MNN on 2020/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <thread>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Scope.hpp>
#include <MNN/expr/ExecutorScope.hpp>

namespace MNN {
namespace Express {

typedef std::shared_ptr<Express::Executor> ExecutorRef;
#if !defined(__APPLE__)
thread_local static Scope<ExecutorRef> g_executor_scope;
#else
static Scope<ExecutorRef> g_executor_scope;
#endif

ExecutorScope::ExecutorScope(const std::shared_ptr<Executor>& current) {
    g_executor_scope.EnterScope(current);
}

ExecutorScope::ExecutorScope(const std::string& scope_name,
                             const std::shared_ptr<Executor>& current) {
    g_executor_scope.EnterScope(scope_name, current);
}

ExecutorScope::~ExecutorScope() {
    g_executor_scope.ExitScope();
}

const std::shared_ptr<Executor> ExecutorScope::Current() {
    if (g_executor_scope.ScopedLevel() > 0) {
        return g_executor_scope.Current().content;
    }
    return Executor::getGlobalExecutor();
}

}  // namespace Express
}  // namespace MNN
