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
#include "torchOpConverter.hpp"

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
        // useless op
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
        }
        if (it->kind() == prim::Loop) {
            if (it->outputs().empty()) {
                it.destroyCurrent();
            }
        }
        if (it->kind().toUnqualString() == std::string("data") ||
            it->kind() == prim::NumToTensor ||
            it->kind() == aten::ScalarImplicit ||
            it->kind() == aten::contiguous ||
            it->kind() == aten::dropout ||
            it->kind() == aten::dropout_ ||
            it->kind() == aten::feature_dropout ||
            it->kind() == aten::clone) {
            it->output()->replaceAllUsesWith(it->input(0));
            for (int i = it->inputs().size()-1; i >= 0; i--) {
                it->removeInput(i);
            }
            it.destroyCurrent();
        }
        if (it->kind() == aten::detach ||
            it->kind() == aten::list ||
            it->kind().toDisplayString() == std::string("aten::cpu")) {
            it->output()->replaceAllUsesWith(it->input(0));
            for (int i = it->inputs().size()-1; i >= 0; i--) {
                it->removeInput(i);
            }
            it.destroyCurrent();
        }
        if (it->kind() == aten::to) {
            auto dst = it->input(1);
            auto ivalue = toIValue(dst);
            if(!ivalue->isInt()) {
                it->output()->replaceAllUsesWith(it->input(0));
                for (int i = it->inputs().size()-1; i >= 0; i--) {
                    it->removeInput(i);
                }
                it.destroyCurrent();
            }
        }
        if (it->kind() == aten::slice) {
            auto start = it->input(2);
            auto end = it->input(3);
            // [0 : 1 : INT_MAX] can remove
            if (toIValue(start) && getValue<int64_t>(start) == 0 &&
                toIValue(end) && getValue<int64_t>(end) == 9223372036854775807) {
                if (it->inputs().size() > 4) {
                    auto stride = it->input(4);
                    if (toIValue(stride) && getValue<int64_t>(stride) != 1) {
                        continue;
                    }
                }
                it->output()->replaceAllUsesWith(it->input(0));
                for (int i = it->inputs().size()-1; i >= 0; i--) {
                    it->removeInput(i);
                }
                it.destroyCurrent();
            }
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
   We remove all ListConstruct op with only one input and not used by aten::cat, like below:
        %116 : Tensor?[] = prim::ListConstruct(%115)
        %alpha0.1 : Tensor = aten::index_put_(%alpha.1, %116, %x.1, %16)
   ListConstruct used by aten::cat will be reserved like below:
        %features.2 : Tensor[] = prim::ListConstruct(%input3.4)
        %concated_features.380 : Tensor = aten::cat(%features.2, %5)
   Attention: Runing this pass after removeListAppend
 */
void removeListConstructOps(Block* block) {
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end; ++it) {
        for (auto b : it->blocks()) {
            removeUselessOps(b);
        }
        if (it->kind() == prim::ListConstruct && it->inputs().size() == 1) {
            bool remove = true;
            for (auto use : it->output()->uses()) {
                if (use.user->kind() == aten::cat) {
                    remove = false;
                    break;
                }
            }
            if (remove) {
                it->output()->replaceAllUsesWith(it->input(0));
                it->removeInput(0);
                it.destroyCurrent();
            }
        }
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
/*
   We rewrite something like:
        x = ListConstruct(v1, v2, v3)
        y = stack(y, axis)
   to:
        y = stack(v1, v2, v3, axis)
*/
void FuseListStack(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;

        for (Block* sub_block : node->blocks()) {
            FuseListUnpack(sub_block);
        }
        std::set<NodeKind> fusekind = {
            aten::stack
        };
        if (it->kind() == aten::stack) {
            auto input = it->input(0)->node();
            if (input->kind() == prim::ListConstruct) {
                auto axis = it->input(1);
                it->removeAllInputs();
                for (int i = 0; i < input->inputs().size(); i++) {
                    it->addInput(input->input(i));
                }
                it->addInput(axis);
                input->destroy();
            }
        }
    }
}
/*
 We rewrite something like:
    %y : int, %z : int = prim::Loop(%6, %2, %y.1, %z.1) # <ipython-input-14-d0a2ead71c2a>:6:4
        block0(%i.1 : int, %y.11 : int, %z.11 : int):
            %y.5 : int = aten::add(%y.11, %i.1) # <ipython-input-14-d0a2ead71c2a>:7:8
            %z.5 : int = aten::mul(%z.11, %5) # <ipython-input-14-d0a2ead71c2a>:8:8
            -> (%2, %y.5, %z.5)
 to:
    %y : int, %z : int = prim::Loop(%6, %2, %y.1, %z.1) # <ipython-input-14-d0a2ead71c2a>:6:4
        block0(%i.1 : int, %y.11 : int, %z.11 : int):
            %y.5 : int = aten::add(%y.1, %i.1) # <ipython-input-14-d0a2ead71c2a>:7:8
            %z.5 : int = aten::mul(%z.1, %5) # <ipython-input-14-d0a2ead71c2a>:8:8
            -> (%2, %y.5, %z.5)
 */
void LoopBodyLegal(Graph* graph, Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;

        for (Block* sub_block : node->blocks()) {
            LoopBodyLegal(graph, sub_block);
        }
        if (node->kind() == prim::Loop) {
            auto body = node->blocks()[0];
            for (int i = body->inputs().size() - 1; i > 0; i--) {
                body->inputs().at(i)->replaceAllUsesWith(node->inputs().at(i + 1));
            }
        }
    }
}
/*
 inference input type, such as below:
    x: Tensor;
    y = aten::embedding(_, x);
 then x's scalar type is int
*/
void InputTypeInfer(Graph* graph) {
    // TODO: add more typeOps and propagateOps
    static std::map<NodeKind, std::vector<ScalarType>> opInputTypes {
        // aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
        { aten::embedding, { ScalarType::Float, ScalarType::Int } },
        // aten::matmul(Tensor self, Tensor other) -> Tensor
        { aten::matmul, { ScalarType::Float, ScalarType::Float } },
        // aten::linear(Tensor input, Tensor weight, Tensor bias) -> Tensor
        { aten::linear, { ScalarType::Float, ScalarType::Float, ScalarType::Float } },
        // aten::conv2d(Tensor input, Tensor weight, Tensor bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
        { aten::conv2d, { ScalarType::Float, ScalarType::Float, ScalarType::Float } },
    };
    static std::set<NodeKind> typePropagateOps {
        // shape change
        aten::slice, aten::view, aten::transpose, aten::permute,
        // compute
        aten::add, aten::sub, aten::mul, aten::div,
    };
    auto mergeType = [](ScalarType type, ScalarType newType) {
        if (type == newType || newType == c10::ScalarType::Undefined) {
            return type;
        }
        if (type == c10::ScalarType::Undefined) {
            return newType;
        }
        MNN_ASSERT(false);
        return c10::ScalarType::Undefined;
    };
    std::function<ScalarType(Value*)> getScalarType = [&](Value* input) -> ScalarType {
        auto inputType = ScalarType::Undefined;
        for (auto use : input->uses()) {
            int idx = -1;
            for (int i = 0; i < use.user->inputs().size(); i++) {
                if (use.user->input(i) == input) {
                    idx = i;
                }
            }
            auto newType = ScalarType::Undefined;
            if (typePropagateOps.find(use.user->kind()) != typePropagateOps.end()) {
                newType = getScalarType(use.user->output());
            } else {
                const auto iter = opInputTypes.find(use.user->kind());
                if (iter != opInputTypes.end() && idx >= 0 && idx < iter->second.size()) {
                    newType = iter->second[idx];
                }
            }
            inputType = mergeType(inputType, newType);
        }
        return inputType;
    };

    for (auto input : graph->inputs()) {
        auto type = input->type()->cast<TensorType>();
        if (!type) {
            continue;
        }
        auto scalarType = getScalarType(input);
        input->setType(type->withScalarType(scalarType));
    }
}

/*
Unpack outputs, such as below:
    return List(x, y); -> return x, y;
    return Dict('x', x); -> return x;
    return Tuple(Tuple(x, y), z); return x, y, z;
*/
void OutputsUnpack(Graph* graph) {
    std::function<void(Node* tuple, std::vector<Node*>&, std::vector<Value*>&)> flattenTuple =
    [&flattenTuple](Node* tuple, std::vector<Node*>& tuples, std::vector<Value*>& values) -> void
    {
        tuples.push_back(tuple);
        for (auto input : tuple->inputs()) {
            auto node = input->node();
            if (node->kind() == prim::TupleConstruct) {
                flattenTuple(node, tuples, values);
            } else {
                values.push_back(input);
            }
        }
    };
    for (int i = 0; i < graph->outputs().size(); i++) {
        auto node = graph->outputs()[i]->node();
        // unpack output
        switch (node->kind()) {
            case prim::TupleConstruct: {
                std::vector<Node*> tuples;
                std::vector<Value*> values;
                flattenTuple(node, tuples, values);
                for (auto realOutput : values) {
                    graph->registerOutput(realOutput);
                }
                graph->eraseOutput(i);
                for (auto tuple : tuples) {
                    if (!tuple->hasUses()) {
                        tuple->destroy();
                    }
                }
                break;
            }
            case prim::DictConstruct: {
                graph->registerOutput(node->input(1));
                graph->eraseOutput(i);
                node->destroy();
                break;
            }
            case prim::ListConstruct: {
                for (int i = 0; i < node->inputs().size(); i++) {
                    graph->registerOutput(node->input(i));
                }
                graph->eraseOutput(i);
                node->destroy();
                break;
            }
        }
    }
}

/*
distinguish overloaded function, such as below:
    torch.max(Tensor, Tensor) is compare
    torch.max(Tensor, int) is reduce
*/
void overloadDistinguish(Block* block) {
    auto symb = c10::Symbol::fromQualString("attr::mnn_tag");
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;

        for (Block* sub_block : node->blocks()) {
            overloadDistinguish(sub_block);
        }
        switch (node->kind()) {
            // min/max(Tensor, Tensor) is compare
            // min/max(Tensor, int) is reduce
            case aten::min:
            case aten::max:
            case aten::sum:
                if (node->inputs().size() > 1 &&
                    (node->input(1)->type()->kind() == c10::TypeKind::IntType ||
                     node->input(1)->type()->kind() == c10::TypeKind::ListType)) {
                    node->s_(symb, "reduce");
                } else {
                    node->s_(symb, "binary");
                }
                break;
            case aten::index:
                if (node->input(1)->node()->kind() == prim::ListConstruct) {
                    node->s_(symb, "stridedslice");
                }
                break;
            default:
                // do nothing
                break;
        }
    }
}
/*
fuse as_tensor, such as below:
    d = prim::dtype(b);
    c = aten::as_tensor(a, d);
 -> c = aten::type_as(a, b)
*/
void FuseAsTensor(Graph* graph, Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;
        for (Block* sub_block : node->blocks()) {
            FuseAsTensor(graph, sub_block);
        }
        if (node->kind() == prim::dtype) {
            for (auto use : node->output(0)->uses()) {
                auto as_tensor = use.user;
                Node* typeAs = graph->create(aten::type_as, 1);
                typeAs->addInput(as_tensor->input(0));
                typeAs->addInput(node->input(0));
                typeAs->output(0)->copyMetadata(as_tensor->output(0));
                as_tensor->replaceAllUsesWith(typeAs);
                as_tensor->removeAllInputs();
                as_tensor->destroy();
            }
        }
    }
}

/*
fuse uniform, such as below:
    d = aten::empty(shape);
    c = aten::uniform_(a, low, hight);
 -> c = aten::uniform_(shape, low, hight)
*/
void FuseUniform(Graph* graph, Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        auto* node = *it;
        it++;
        for (Block* sub_block : node->blocks()) {
            FuseUniform(graph, sub_block);
        }
        if (it->kind().toUnqualString() == std::string("uniform_")) {
            auto input = it->input(0)->node();
            if (input->kind() == aten::empty) {
                it->replaceInput(0, input->input(0));
                input->destroy();
            }
        }
    }
}

std::shared_ptr<Graph> torchOptPass(Module& module) {
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
    removeListConstructOps(graph->block());
    //RemoveTensorMutation(graph);
    // elimate dead code
    EliminateDeadCode(graph, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
    // constant propagation
    ConstantPooling(graph);
    ConstantPropagation(graph);
    // fuse
    FuseGraph(graph, true);
    PeepholeOptimize(graph);
    FuseAddMM(graph);
    // FoldConvBatchNorm(module);
    FuseListUnpack(graph->block());
    FuseListStack(graph->block());
    // distinguish overload function
    overloadDistinguish(graph->block());
    // legal loop body's var name
    LoopBodyLegal(graph.get(), graph->block());
    // infer input tensor's scalar type by op
    InputTypeInfer(graph.get());
    // split output tensor if wrap with list/tuple
    OutputsUnpack(graph.get());
    // dtype + as_tensor -> type_as
    FuseAsTensor(graph.get(), graph->block());
    // empty + uniform -> uniform
    FuseUniform(graph.get(), graph->block());
#ifdef MNN_DUMP_TORCHSCRIPT
    graph->dump();
#endif
    return graph;
}
}
}
