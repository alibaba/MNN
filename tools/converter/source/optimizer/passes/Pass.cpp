//
//  Pass.cpp
//  MNNConverter
//
//  Created by MNN on b'2020/12/07'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN/expr/Expr.hpp"
#include "converter/source/optimizer/passes/Pass.hpp"
#include "converter/source/optimizer/passes/PassRegistry.hpp"
#include "converter/source/optimizer/SubGraphComplete.hpp"
#include "converter/source/optimizer/Program.hpp"
#include "converter/source/optimizer/Global.hpp"

namespace MNN {
namespace passes {

NestedPass::NestedPass(const std::string& pass_name, PassContext *context)
    : Pass(pass_name), pass_manager_(new PassManager(context)) {}

PassManager& NestedPass::OwningPassManager() const {
    return *(pass_manager_.get());
}

bool NestedPass::Run(PassContext *context) {
    return pass_manager_->RunOnOperation(context);
}

PassManager::PassManager(const PassManager& other) {
    for (const auto& pass : other.passes_) {
        passes_.push_back(pass->Clone());
    }
}

void PassManager::AddPass(std::unique_ptr<Pass>&& pass) {
    passes_.push_back(std::move(pass));
}

void PassManager::AddPass(const std::string& pass_name) {
    std::unique_ptr<Pass> pass = PassRegistry::GetPass(pass_name)->Clone();
    AddPass(std::move(pass));
}

void PassManager::AddNestedPass(std::unique_ptr<Pass>&& pass) {
    PassManager& pm = AddNest();
    pm.AddPass(std::move(pass));
}

void PassManager::AddNestedPass(const std::string& pass_name) {
    std::unique_ptr<Pass> pass = PassRegistry::GetPass(pass_name)->Clone();
    AddNestedPass(std::move(pass));
}

PassManager& PassManager::AddNest() {
    std::unique_ptr<Pass> nest(new NestedPass("nested", context_));
    passes_.push_back(std::move(nest));
    return reinterpret_cast<NestedPass*>(  // NOLINT
        passes_.back().get())->OwningPassManager();
}

std::unique_ptr<NetT> PassManager::RunAllPasses(  // NOLINT
    std::unique_ptr<MNN::NetT>& originNet,        // NOLINT
    const std::unordered_map<std::string, Express::VARP>& inputs) {
    auto program = MNN::Express::Program::create(originNet.get(), true);
    program->input(inputs);

    auto operations = Express::Variable::getExecuteOrder(program->outputs());
    for (auto op : operations) {
        PassContext ctx(*context_);
        ctx.node = op;
        RunOnOperation(&ctx);
    }
    std::unique_ptr<MNN::NetT> newNet(new MNN::NetT);
    newNet->sourceType = originNet->sourceType;
    newNet->bizCode    = originNet->bizCode;
    newNet->outputName = originNet->outputName;
    Express::Variable::save(program->outputs(), newNet.get());
    return std::move(newNet);
}

std::unique_ptr<NetT> PassManager::Run(std::unique_ptr<NetT>& net) {
    std::vector<MNN::SubGraphProtoT*> subgraphs;
    for (auto& subgraph : net->subgraphs) {
        subgraphs.push_back(subgraph.get());
    }
    auto RunAllPassesImpl = [this](
        std::unique_ptr<MNN::NetT>& originNet,
        const std::unordered_map<std::string, Express::VARP>& inputs) {
        return this->RunAllPasses(originNet, inputs);
    };
    
    Express::OptimizeContext ctx;
    ctx.subgraphs = subgraphs;
    ctx.is_training = context_->is_training;
    ctx.verbose = context_->verbose;
    ctx.source = context_->source;
    ctx.completed_subgraphs = {};
    ctx.RunOptimize = RunAllPassesImpl;

    Global<Express::OptimizeContext>::Reset(&ctx);

    std::unordered_map<std::string, Express::VARP> inputs;
    std::unique_ptr<MNN::NetT> result = ctx.RunOptimize(net, inputs);
    for (auto* subgraph : ctx.completed_subgraphs) {
        result->subgraphs.emplace_back(subgraph);
    }
    return std::move(result);
}

bool PassManager::RunOnOperation(PassContext *context) {
    bool status = false;
    while (1) {
        bool iterable = false;
        for (auto& pass : passes_) {
            iterable = iterable || pass->Run(context);
        }
        if (!iterable) {
            break;
        } else {
            status = true;
        }
    }
    return status;
}

bool RewritePass::Run(PassContext *context) {
    return VerifyAndRewrite(context);
}

bool RewritePass::VerifyAndRewrite(PassContext* context) {
    if (!verify_fn_(context)) {
        return false;
    }
    return rewrite_fn_(context);
}

}  // namespace passes
}  // namespace MNN
