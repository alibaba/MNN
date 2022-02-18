//
//  PassRegistry.cpp
//  MNNConverter
//
//  Created by MNN on b'2020/12/07'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <unordered_map>

#include "MNN/MNNDefine.h"
#include "converter/source/optimizer/passes/PassRegistry.hpp"

namespace MNN {
namespace passes {

// All registered passes.
static std::unordered_map<std::string, std::unique_ptr<Pass>>*
    AllRegisteredPasses() {
    static std::unordered_map<std::string, \
                              std::unique_ptr<Pass>> g_registered_passes;
    return &g_registered_passes;
}
// All registered pass managers.
static std::vector<std::unique_ptr<PassManager>>* AllRegisteredPassManagers() {
    static std::vector<std::unique_ptr<PassManager>> g_registered_pass_managers;
    return &g_registered_pass_managers;
}

/*static*/ PassManager* PassManagerRegistry::GetPassManager(int index) {
    auto* g_registered_pass_managers = AllRegisteredPassManagers();
    MNN_CHECK(index < g_registered_pass_managers->size(),
              "The pass manager index is out of bounds.");
    return (*g_registered_pass_managers)[index].get();
}

/*static*/ std::vector<PassManager*> PassManagerRegistry::GetAllPassManagers() {
    std::vector<PassManager*> pass_managers;
    for (auto& pm : *(AllRegisteredPassManagers())) {
        pass_managers.push_back(pm.get());
    }
    return pass_managers;
}

/*static*/ void PassManagerRegistry::AddPassManager(const PassManager& pm) {
    auto* g_registered_pass_managers = AllRegisteredPassManagers();
    g_registered_pass_managers->emplace_back(new PassManager(pm));
}

/*static*/ void PassRegistry::AddPass(std::unique_ptr<Pass>&& pass) {
    auto* g_registered_passes = AllRegisteredPasses();
    g_registered_passes->emplace(pass->name(), std::move(pass));
}

/*static*/ Pass* PassRegistry::GetPass(const std::string& pass_name) {
    auto* g_registered_passes = AllRegisteredPasses();
    const auto& it = g_registered_passes->find(pass_name);
    if (it != g_registered_passes->end()) {
        return it->second.get();
    }
    return nullptr;
}

RewritePassRegistry::RewritePassRegistry(const std::string& pass_name)
    : pass_name_(pass_name) {
    std::unique_ptr<Pass> pass(new RewritePass(pass_name));
    PassRegistry::AddPass(std::move(pass));
}

RewritePass* GetRewritePassByName(const std::string& pass_name) {
    Pass* pass = PassRegistry::GetPass(pass_name);
    MNN_CHECK(pass, "Pass has not been setup.");
    if (pass->type() != Pass::PassType::kRewrite) {
        MNN_ERROR("Pass %s is registered but not rewrite pass.",
                  pass_name.c_str());
    }
    RewritePass *rewrite_pass = static_cast<RewritePass*>(pass);
    return rewrite_pass;
}

void RewritePassRegistry::SetVerify(RewritePassRegistry::FuncType verify_fn) {
    auto *rewrite_pass = GetRewritePassByName(pass_name_);
    rewrite_pass->SetVerify(verify_fn);
}

void RewritePassRegistry::SetRewrite(RewritePassRegistry::FuncType rewrite_fn) {
    auto *rewrite_pass = GetRewritePassByName(pass_name_);
    rewrite_pass->SetRewrite(rewrite_fn);
}

}  // namespace passes
}  // namespace MNN
