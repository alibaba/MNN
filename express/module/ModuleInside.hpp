//
//  ModuleInside.hpp
//  MNN
//
//  Created by MNN on b'2025/01/13'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ModuleInside_hpp
#define ModuleInside_hpp
#include <unordered_map>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
namespace MNN {
namespace Express {
class Module::CloneContext {
public:
    CloneContext() = default;
    explicit CloneContext(const bool shareParams)
        : mShareParams(shareParams) {}
    virtual ~CloneContext() = default;

    const bool shareParams() const { return mShareParams; }

    EXPRP getOrClone(const EXPRP expr);
    VARP getOrClone(const VARP var);
    std::shared_ptr<Executor::RuntimeManager> pRuntimeManager;
private:
    bool mShareParams = false;
    std::unordered_map<const Expr*, EXPRP> mExprMap;
    std::unordered_map<const Variable*, VARP> mVarMap;
};
};
};
#endif
