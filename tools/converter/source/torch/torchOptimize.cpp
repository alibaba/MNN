//
//  torchOptimize.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "torchOptimize.hpp"
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/erase_number_types.h>

namespace torch {
namespace jit {
void removeUselessOps(Block* block) {
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end; ++it) {
        for (auto b : it->blocks()) {
            removeUselessOps(b);
        }
        std::set<NodeKind> uselessKind = {
            // prime
            prim::Print,
            prim::RaiseException,
            prim::TimePoint,
            prim::annotate,
            // aten
            aten::warn,
        };
        if (uselessKind.count(it->kind())) {
            for (size_t i = 0; i < it->inputs().size();) {
                auto input = it->inputs().at(i);
                // only handling constants bc of potential side effects
                if (input->uses().size() == 1 &&
                    input->node()->kind() == prim::Constant) {
                    it->removeInput(i);
                    input->node()->destroy();
                } else {
                    ++i;
                }
            }
            it.destroyCurrent();
        } else if (it->kind() == prim::Loop) {
            if (it->outputs().empty()) {
                it.destroyCurrent();
            }
        } else if (it->kind() == aten::contiguous || it->kind().toUnqualString() == std::string("data")) {
            it->output()->replaceAllUsesWith(it->input(0));
            for (int i = it->inputs().size()-1; i >= 0; i--) {
                it->removeInput(i);
            }
            it.destroyCurrent();
        }
    }
}

/*
   We rewrite something like:
        x = {v0}
        x.append(v1)
        foo(x)
        x.append(v2)
        bar(x)
   to:
        x1 = {v0, v1}
        foo(x1)
        x2 = {v0, v1, v2}
        bar(x2)
   this is a strengthen version of RemoveListMutation
*/
void removeListAppend(Graph* graph, Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;

        for (Block* sub_block : node->blocks()) {
            removeListAppend(graph, sub_block);
        }

        if (!(node->kind() == aten::append && node->inputs().at(0)->node()->kind() == prim::ListConstruct)) {
            continue;
        }
        Value* mutated_value = node->inputs().at(0);
        Node* list_node = mutated_value->node();
        Node* new_list_node = graph->create(prim::ListConstruct, 1);
        for (Value* input : list_node->inputs()) {
            new_list_node->addInput(input);
        }
        new_list_node->addInput(node->inputs().at(1));
        new_list_node->copyMetadata(list_node);
        new_list_node->insertAfter(node);
        new_list_node->output()->setType(list_node->output()->type());
        mutated_value->replaceAllUsesAfterNodeWith(node, new_list_node->output());
        node->destroy();
    }
}
/*
   We rewrite something like:
        y = chunk(x)
        v1, v2, v3 = ListUnpack(y)
   to:
        v1, v2, v3 = chunk(x)
*/
void FuseListUnpack(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;

        for (Block* sub_block : node->blocks()) {
            FuseListUnpack(sub_block);
        }
        std::set<NodeKind> fusekind = {
            aten::split,
            aten::split_with_sizes,
            aten::split_with_sizes,
            aten::unsafe_split_with_sizes,
            aten::unbind,
            aten::chunk,
            aten::unsafe_chunk,
            aten::where,
        };
        if (fusekind.count(it->kind()) &&
            it->outputs().size() == 1 &&
            it->output()->uses().size() == 1) {
            const auto listunpack = it->output()->uses()[0].user;
            if (listunpack->kind() == prim::ListUnpack) {
                // it->i_(Symbol::fromQualString("attr::_outputs"),
                //         static_cast<int64_t>(listunpack->outputs().size()));
                for (auto i = 0; i < listunpack->outputs().size(); ++i) {
                    auto new_output = it->addOutput();
                    new_output->copyMetadata(listunpack->output(i));
                }
                listunpack->removeAllInputs();
                it->eraseOutput(0);
                listunpack->replaceAllUsesWith(*it);
                listunpack->destroy();
            }
        }
    }
}
std::shared_ptr<Graph> torchOptPass(const char* name) {
    // Deserialize the ScriptModule from a file, set to eval mode and freeze
    auto module = torch::jit::load(name);
    module.eval();
    module = torch::jit::freeze_module(module);
    auto graph = module.get_methods()[0].graph();
    Inline(*(graph.get()));
    // normalize, Example: aten::absolute -> aten::abs
    NormalizeOps(graph);
    // remove some ops, Example: prim::RaiseException
    removeUselessOps(graph->block());
    removeDropout(module);
    // Example: x = x + 1; -> x_1 = x + 1;
    RemoveInplaceOps(graph);
    // Example: x = {v0}; x.append(v1); -> x = {v0, v1};
    // RemoveListMutation(graph);
    removeListAppend(graph.get(), graph->block());
    //RemoveTensorMutation(graph);
    // elimate dead code
    EliminateDeadCode(graph, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
    // constant propagation
    ConstantPooling(graph);
    ConstantPropagation(graph);
    // fuse
    FuseGraph(graph);
    PeepholeOptimize(graph);
    FuseAddMM(graph);
    FoldConvBatchNorm(module);
    FuseListUnpack(graph->block());
    // graph->dump();
    return graph;
}
}
}
