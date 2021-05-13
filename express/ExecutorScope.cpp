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
#if !defined(__APPLE__)
thread_local static std::once_flag gInitFlag;
thread_local static Scope<ExecutorRef>* g_executor_scope = nullptr;
#else
static std::once_flag gInitFlag;
static Scope<ExecutorRef>* g_executor_scope = nullptr;
#endif

static Scope<ExecutorRef>* _getGlobalScope() {
    std::call_once(gInitFlag,
                   [&]() {
        g_executor_scope = new Scope<ExecutorRef>;
    });
    return g_executor_scope;
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
