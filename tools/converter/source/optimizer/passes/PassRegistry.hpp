//
//  PassRegistry.hpp
//  MNNConverter
//
//  Created by MNN on b'2020/12/07'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_PASSES_PASS_REGISTRY_HPP_
#define MNN_CONVERTER_PASSES_PASS_REGISTRY_HPP_

#include <string>
#include "converter/source/optimizer/passes/Pass.hpp"

namespace MNN {
namespace passes {

class MNN_PUBLIC PassRegistry {
public:
    static Pass* GetPass(const std::string& name);
    static void AddPass(std::unique_ptr<Pass>&& pass);
};

class MNN_PUBLIC PassManagerRegistry {
public:
    static PassManager* GetPassManager(int index);
    static std::vector<PassManager*> GetAllPassManagers();

    static void AddPassManager(const PassManager& pm);
};

class MNN_PUBLIC RewritePassRegistry {
public:
    RewritePassRegistry(const std::string& name);

    using FuncType = std::function<bool(PassContext* context)>;

    void SetVerify(FuncType verify_fn);
    void SetRewrite(FuncType rewrite_fn);

private:
    std::string pass_name_;
};

class MNN_PUBLIC RewritePassRegistryHelper {
public:
    RewritePassRegistryHelper(const std::string& name)
        : registry_(new RewritePassRegistry(name)) {}

    virtual ~RewritePassRegistryHelper() = default;

    template <typename FuncType,       \
              typename std::enable_if< \
                      std::is_function<FuncType>::value> * = nullptr>
    RewritePassRegistryHelper&& Verify(FuncType verify_fn) {
        registry_->SetVerify(verify_fn);
        return std::move(*this);
    }

    template <typename FuncType,       \
              typename std::enable_if< \
                      std::is_function<FuncType>::value> * = nullptr>
    RewritePassRegistryHelper&& Rewrite(FuncType rewrite_fn) {
        registry_->SetRewrite(rewrite_fn);
        return std::move(*this);
    }

private:
    std::shared_ptr<RewritePassRegistry> registry_;
};

}  // namespace passes
}  // namespace MNN

// REGISTER_REWRITE_PASS(FuseLayerNorm) \
//     .Verify([](PassContext* context){ return false; })   \
//     .Rewrite([](PassContext* context) { return false; });
#define REGISTER_REWRITE_PASS(pass)                       \
    static auto _rewrite_pass_registry_helper_##pass##_ = \
        MNN::passes::RewritePassRegistryHelper(#pass)

#define REGISTER_PASS_MANAGER(pass_manager) \
    PassManagerRegistry::AddPassManager(pass_manager);

#endif  // MNN_CONVERTER_PASSES_PASS_REGISTRY_HPP_
