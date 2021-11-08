//
//  ExecutorScope.hpp
//  MNN
//
//  Created by MNN on 2020/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_EXPR_EXECUTOR_SCOPE_HPP_
#define MNN_EXPR_EXECUTOR_SCOPE_HPP_

#include <MNN/expr/Executor.hpp>

namespace MNN {
namespace Express {

struct MNN_PUBLIC ExecutorScope final {
public:
    ExecutorScope() = delete;
    explicit ExecutorScope(const ExecutorScope&) = delete;
    explicit ExecutorScope(const std::shared_ptr<Executor>& current);

    explicit ExecutorScope(const std::string& scope_name,
                           const std::shared_ptr<Executor>& current);

    virtual ~ExecutorScope();

    static const std::shared_ptr<Executor> Current();
};

}  // namespace MNN
}  // namespace Express
#endif  // MNN_EXPR_EXECUTOR_SCOPE_HPP_
