//
//  Pass.hpp
//  MNNConverter
//
//  Created by MNN on b'2020/12/07'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_PASSES_PASS_HPP_
#define MNN_CONVERTER_PASSES_PASS_HPP_

#include "MNN/expr/Expr.hpp"
#include "MNN_generated.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

namespace MNN {
namespace passes {

typedef struct PassContext {
    bool is_training = false;
    bool verbose = true;
    NetSource source = NetSource_TENSORFLOW;
    // 
    Express::EXPRP node;
} PassContext;

class Pass;
class NestedPass;
class PassManager;

// The abstract base pass class.
class MNN_PUBLIC Pass {
public:
    Pass() = default;
    Pass(const std::string& pass_name) : pass_name_(pass_name) {}
    virtual ~Pass() = default;
  
    const std::string& name() const { return pass_name_; }
  
    enum class PassType : int {
        kInvalid = 0,
        kNested = 1,
        kRewrite = 2,
    };
    virtual PassType type() const = 0;
    virtual std::unique_ptr<Pass> Clone() const = 0;

    virtual bool Run(PassContext *context) = 0;

private:
    std::string pass_name_;
};

class MNN_PUBLIC PassManager {
public:
    PassManager() = delete;
    PassManager(PassContext *context) : context_(context) {}
    PassManager(const PassManager& other);
    PassManager& operator=(const PassManager&) = delete;

    virtual ~PassManager() = default;

    void AddPass(std::unique_ptr<Pass>&& pass);
    void AddPass(const std::string& pass_name);

    void AddNestedPass(std::unique_ptr<Pass>&& pass);
    void AddNestedPass(const std::string& pass_name);

    PassManager& AddNest();

    std::unique_ptr<NetT> Run(std::unique_ptr<NetT>& net);

private:
    friend class NestedPass;
    bool RunOnOperation(PassContext *context);

    std::unique_ptr<NetT> RunAllPasses(
         std::unique_ptr<MNN::NetT>& originNet,        // NOLINT
         const std::unordered_map<std::string, Express::VARP>& inputs);

    PassContext *context_;
    std::vector<std::unique_ptr<Pass>> passes_;
};

class MNN_PUBLIC NestedPass : public Pass {
public:
    NestedPass() = default;
    NestedPass(const std::string& pass_name, PassContext *context);
  
    PassType type() const override { return PassType::kNested; }
  
    PassManager& OwningPassManager() const;

    std::unique_ptr<Pass> Clone() const override {
        // TODO
        return nullptr;
    }
  
    bool Run(PassContext *context) override;

private:
    std::unique_ptr<PassManager> pass_manager_;
};

class MNN_PUBLIC RewritePass : public Pass {
public:
    using FuncType = std::function<bool(PassContext* context)>;

    RewritePass() = delete;
    virtual ~RewritePass() = default;
    RewritePass(const std::string& pass_name) : Pass(pass_name) {}
    RewritePass(const std::string& pass_name, FuncType verify_fn,
                FuncType rewrite_fn)
        : Pass(pass_name), verify_fn_(verify_fn), rewrite_fn_(rewrite_fn) {}

    void SetVerify(FuncType verify_fn) {
        verify_fn_ = verify_fn;
    }
    void SetRewrite(FuncType rewrite_fn) {
        rewrite_fn_ = rewrite_fn;
    }

    PassType type() const override { return PassType::kRewrite; }

    std::unique_ptr<Pass> Clone() const override {
        return std::unique_ptr<RewritePass>(
            new RewritePass(this->name(), verify_fn_, rewrite_fn_));
    }

    bool Run(PassContext *context) override;

private:
    bool VerifyAndRewrite(PassContext* context);

    std::function<bool(PassContext* context)> verify_fn_;
    std::function<bool(PassContext* context)> rewrite_fn_;
};

}  // namespace passes
}  // namespace MNN

#endif  // MNN_CONVERTER_PASSES_PASS_HPP_
