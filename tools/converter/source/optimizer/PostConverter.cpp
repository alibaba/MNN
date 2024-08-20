//
//  PostConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_set>

#include <MNN/expr/Optimizer.hpp>
#include <set>
#include <MNN/expr/ExecutorScope.hpp>
#include "PostConverter.hpp"
#include "PostTreatUtils.hpp"
#include "Program.hpp"
#include "SubGraphComplete.hpp"
#include "GenerateSubGraph.hpp"
#include "TemplateMerge.hpp"
#include "core/Backend.hpp"
#include "RuntimeAttr.hpp"

#include <MNN/expr/ExecutorScope.hpp>
//#define MNN_POST_CONVERTER_DEBUG

namespace MNN {
namespace Express {
static std::vector<int> NetInputIndices(const MNN::NetT* net) {
    std::vector<int> input_indices;
    for (const auto& op : net->oplists) {
        if (op->type == MNN::OpType_Input) {
            const auto& indices = op->outputIndexes;
            input_indices.insert(input_indices.end(), indices.begin(), indices.end());
        }
    }
    return std::move(input_indices);
}

SubGraphProtoT* FindSubGraphByName(const std::vector<SubGraphProtoT*>& subgraphs, const std::string& subgraph_name) {
    for (SubGraphProtoT* subgraph : subgraphs) {
        if (subgraph->name == subgraph_name) {
            return subgraph;
        }
    }
    return nullptr;
}

bool CompleteSubGraph(const std::unordered_map<std::string, VARP>& inputs, const SubGraphProtoT* subgraph) {
    auto* ctx = Global<OptimizeContext>::Get();
    auto config = Global<modelConfig>::Get();
    MNN_ASSERT(ctx != nullptr);
    // Disable verbose for subgraph.
    bool verbose = ctx->verbose;
    ctx->verbose = false;
    std::vector<std::string> outputNames;
    for (auto o : subgraph->outputs) {
        outputNames.emplace_back(subgraph->tensors[o]);
    }
    std::vector<std::string> inputNames;
    for (auto index : subgraph->inputs) {
        inputNames.emplace_back(subgraph->tensors[index]);
    }

    SubGraphProtoT* mutable_subgraph = // NOLINT
        FindSubGraphByName(ctx->subgraphs, subgraph->name);
    MNN_ASSERT(mutable_subgraph == subgraph);
    std::unique_ptr<MNN::NetT> subnet(new MNN::NetT);
    subnet->oplists    = std::move(mutable_subgraph->nodes);
    subnet->tensorName = mutable_subgraph->tensors;
    subnet->sourceType = ctx->source;
    subnet->outputName = outputNames;
    bool gDebug = false;
    if (gDebug) {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, subnet.get()));
        std::ofstream output("temp.before_opt.mnn", std::ofstream::binary);
        output.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    }
    config->inSubGraph = true;
    std::unique_ptr<MNN::NetT> new_subnet = ctx->RunOptimize(subnet, inputs);
    config->inSubGraph = false;
    if (gDebug) {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, new_subnet.get()));
        std::ofstream output("temp.after_opt.mnn", std::ofstream::binary);
        output.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    }
    mutable_subgraph->nodes               = std::move(subnet->oplists);

    MNN::SubGraphProtoT* new_subgraph(new MNN::SubGraphProtoT);
    new_subgraph->name    = mutable_subgraph->name;
    if (ctx->source != NetSource_ONNX) {
        new_subgraph->inputs  = NetInputIndices(new_subnet.get());
    } else {
        new_subgraph->inputs.resize(inputNames.size());
        for (int i=0; i<inputNames.size(); ++i) {
            for (int j=0; j<new_subnet->tensorName.size(); ++j) {
                if (new_subnet->tensorName[j] == inputNames[i]) {
                    new_subgraph->inputs[i] = j;
                    break;
                }
            }
        }
    }
    new_subgraph->outputs.clear();
    outputNames = new_subnet->outputName;
    for (auto& output : outputNames) {
        bool find = false;
        for (int i = 0; i < new_subnet->tensorName.size(); ++i) {
            if (new_subnet->tensorName[i] == output) {
                find = true;
                new_subgraph->outputs.emplace_back(i);
                break;
            }
        }
        if (!find) {
            MNN_ERROR("Can't find output for %s\n", output.c_str());
        }
    }
    MNN_ASSERT(new_subgraph->outputs.size() == outputNames.size());
    new_subgraph->nodes   = std::move(new_subnet->oplists);
    new_subgraph->tensors = new_subnet->tensorName;

    MNN_ASSERT(!FindSubGraphByName(ctx->completed_subgraphs, new_subgraph->name));
    ctx->completed_subgraphs.push_back(new_subgraph);

    // Recovery verbose.
    ctx->verbose = verbose;
    return true;
}


void RunNetPass(const std::vector<std::string>& passes, std::unique_ptr<MNN::NetT>& originNet) {
    for (auto pass : passes) {
        auto convert = PostConverter::get(pass);
        if (nullptr == convert) {
            LOG(INFO) << "Can't find pass of " << pass << "\n";
            continue;
        }
        bool valid = convert->onExecute(originNet);
        if (!valid) {
            LOG(INFO) << "Run " << pass << "Error\n";
        }
    }
}

std::unique_ptr<MNN::NetT> RunExtraPass(std::unique_ptr<MNN::NetT>& originNet,
                                        const std::unordered_map<std::string, VARP>& inputs) {
    auto program = MNN::Express::Program::create(originNet.get(), true, true);
    program->input(inputs, true);

    std::string pass = "TFExtra";
    switch (originNet->sourceType) {
        case MNN::NetSource_TFLITE:
            pass = "TFliteExtra";
            break;
        case MNN::NetSource_TENSORFLOW:
            pass = "TFExtra";
            break;
        case MNN::NetSource_CAFFE:
            pass = "CaffeExtra";
            break;
        case MNN::NetSource_ONNX:
            pass = "OnnxExtra";
            break;
        case MNN::NetSource_TORCH:
            pass = "TorchExtra";
            break;
        default:
            break;
    }
    auto& merge = MNN::Express::TemplateMerge::getInstance(pass);
    merge.onExecute(program->outputs());
    originNet->oplists.clear();
    originNet->tensorName.clear();

    std::unique_ptr<MNN::NetT> newNet(new MNN::NetT);
    newNet->sourceType = originNet->sourceType;
    newNet->bizCode    = originNet->bizCode;
    newNet->outputName = originNet->outputName;
    program->save(newNet.get());
    return std::move(newNet);
}

std::unique_ptr<MNN::NetT> RunMergePass(std::unique_ptr<MNN::NetT>& originNet,
                                        const std::unordered_map<std::string, VARP>& inputs, PassPriority priority) {
    auto program = MNN::Express::Program::create(originNet.get(), true, true);
    auto boundary = program->input(inputs, true);

    std::string pass = "Merge";
    auto& merge      = MNN::Express::TemplateMerge::getInstance(pass);
    std::map<std::string, VARP> updateVars;
    merge.onExecute(program->outputs(), priority, updateVars, boundary);

    auto Update = [&](std::shared_ptr<Program> program, const std::vector<std::string>& tensorName) {
        program->updateVars(updateVars, tensorName);
    };

    Update(program, originNet->tensorName);

    originNet->oplists.clear();
    originNet->tensorName.clear();

    std::unique_ptr<MNN::NetT> newNet(new MNN::NetT);
    newNet->sourceType = originNet->sourceType;
    newNet->bizCode    = originNet->bizCode;
    newNet->outputName = originNet->outputName;
    program->save(newNet.get());

    RunNetPass({"RemoveUnusefulOp"}, newNet);
    return std::move(newNet);
}

std::unique_ptr<MNN::NetT> optimizeNetImpl(std::unique_ptr<MNN::NetT>& originNet,
                                           const std::unordered_map<std::string, VARP>& inputs) {
    auto current = ExecutorScope::Current();
    current->lazyEval = true;
    current->setLazyComputeMode(Executor::LAZY_FULL);
    current->getAttr()->externalFile = ".__convert_external_data.bin";

    auto* ctx = Global<OptimizeContext>::Get();
    MNN_ASSERT(ctx != nullptr);

    if (ctx->is_training) {
        LOG(INFO) << "convert model for training, reserve BatchNorm and Dropout";
    }
    if (originNet->oplists.size() <= 0) {
        return nullptr;
    }
    std::vector<std::string> postConvertPass;
    postConvertPass = {
        // Separate Tensor for inplace op
        "RemoveInplace",

        // Remove Unuseful Op such as NoOp, Identity, Seq2Out,
        "RemoveUnusefulOp",

        // Remove Dropout, if `forTraining` flag is set, Dropout will be reserved
        "RemoveDropout",

        // Remove Dup op
        "FuseDupOp",
        
        // Remove Invalid Cast
        "RemoveInvalidCast",

        // Turn InnerProduct from Caffe / Onnx to Convolution
        "TransformInnerProduct",

        // Turn Im2Seq from Caffe to Reshape
        "TransformIm2Seq",

        // Turn Caffe's ShuffleChannel to compose op
        "TransformShuffleChannel",
        
        "MoveUnaryOpBeforeReshape",

    };
    if (ctx->is_training) {
        std::vector<std::string>::iterator iter = postConvertPass.begin();
        while (iter != postConvertPass.end()) {
            if (*iter == "RemoveDropout") {
                iter = postConvertPass.erase(iter);
            } 
            else {
                iter++;
            }
        }
    }
    RunNetPass(postConvertPass, originNet);
    std::vector<std::string> midOptPass = {
        // Remove Dup op
        "FuseDupOp",
        // Remove Invalid Cast
        "RemoveInvalidCast"
    };
    std::vector<std::unique_ptr<TensorDescribeT>> tensorDescribe;
    if (originNet->extraTensorDescribe.size() > 0) {
        tensorDescribe = std::move(originNet->extraTensorDescribe);
    }
    
    std::unique_ptr<MNN::NetT> newNet;
    newNet = std::move(RunExtraPass(originNet, inputs));
    RunNetPass(midOptPass, newNet);
    newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_FRONT));
    newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_HIGH));

    std::vector<std::string> afterProgramConvert = {
        // Turn BatchNormal to Scale When inference, if `forTraining` flag is set, BN will be reserved
        "TransformBatchNormal",

        // expand ShapeN to N Shapes
        "ResolveTfShapeN",

        // WARNNING: should merge BN and Scale before Relu and Relu6

        // Merge BN info Convolution, if `forTraining` flag is set, BN will be reserved
        "MergeBNToConvolution",

        // Merge Scale info Convolution
        "MergeScaleToConvolution",

        // Merge Relu Convolution
        "MergeReluToConvolution",

        // Merge Relu6 Convolution
        "MergeRelu6ToConvolution",

        // Merge Relu BinaryOp
        "MergeReluToBinaryOp",

    };
    if (ctx->is_training) {
        std::vector<std::string>::iterator iter = afterProgramConvert.begin();
        while (iter != afterProgramConvert.end()) {
            if (*iter == "TransformBatchNormal" || *iter == "MergeBNToConvolution") {
                iter = afterProgramConvert.erase(iter);
            }
            else {
                iter++;
            }
        }
    }
    RunNetPass(afterProgramConvert, newNet);

    newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_MIDDLE));

    afterProgramConvert = {
        "RemoveCopy",
        // Add tensor dimension format convert for NC4HW4 - NHWC / NC4HW4 - NCHW
        "AddTensorFormatConverter",

        // Turn group convolution to Slice - Convolution - Concat
        "TransformGroupConvolution",
        "TransformGroupConvolution3D",

        // Remove output tensor convert
        "RemoveOutputTensorConvert",
    };
    RunNetPass(afterProgramConvert, newNet);

    // Maybe eliminate the redundant quantize and dequantize ops, then remove
    // the unuseful `Identity`.
    newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_LOW));

    // Maybe eliminate the redundant tensor format ops, then remove the unuseful
    // `Identity`.
    newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_LOW));
    newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_FINAL));

    if (tensorDescribe.size() > 0) {
        newNet->extraTensorDescribe = std::move(tensorDescribe);
    }
    RunNetPass({"ReIndexTensor"}, newNet);
    RunNetPass({"ReIndexOnnxIfAlias"}, newNet);

    return std::move(newNet);
}

bool fuseConstIntoSubgraph(MNN::NetT* net, const std::vector<MNN::SubGraphProtoT*>& subgraphs) {
    if (subgraphs.empty()) {
        return false;
    }
    // Create Map for subGraphs
    // Key, protot, refcount
    std::map<std::string, std::pair<MNN::SubGraphProtoT*, int>> subGraphMaps;
    std::set<MNN::SubGraphProtoT*> modifiedSubGraph;
    for (auto s : subgraphs) {
        subGraphMaps.insert(std::make_pair(s->name, std::make_pair(s, 0)));
    }
    for (int i = 0; i < net->oplists.size(); ++i) {
        auto& op = net->oplists[i];
        if (op->type == MNN::OpType_While) {
            auto param = op->main.AsWhileParam();
            subGraphMaps[param->body_graph].second++;
            subGraphMaps[param->cond_graph].second++;
            continue;
        }
        if (op->type == MNN::OpType_If) {
            auto param = op->main.AsIfParam();
            subGraphMaps[param->else_graph].second++;
            subGraphMaps[param->then_graph].second++;
            continue;
        }
    }

    // Try Merge Const into subgraph
    // Search all const op
    std::vector<int> constOpIndexes(net->tensorName.size(), -1);
    for (int i = 0; i < net->oplists.size(); ++i) {
        auto& op = net->oplists[i];
        if (op->type == MNN::OpType_Const) {
            constOpIndexes[op->outputIndexes[0]] = i;
        }
    }

    // Try Merge for while
    std::set<int> removeConstOpIndexes;
    for (int opIndex = 0; opIndex < net->oplists.size(); ++opIndex) {
        auto& op = net->oplists[opIndex];
        if (op->type != MNN::OpType_While) {
            continue;
        }
        auto param = op->main.AsWhileParam();
        if (param->cond_graph.empty()) {
            // If cond_graph is empty, it come from onnx's loop
            // TODO: Support Loop from onnx
            continue;
        }
        auto body  = subGraphMaps[param->body_graph];
        auto cond  = subGraphMaps[param->cond_graph];
        // Don't support for shared subgrah's optimize
        if (body.second > 1 || cond.second > 1) {
            continue;
        }
        MNN_ASSERT(op->inputIndexes.size() == param->aliases_inputs.size());

        // Merge into subgraph
        std::set<int> removeInputs;
        std::set<int> bodyInputRemove;
        std::set<int> condInputRemove;
        auto mergeToSubGraph = [](MNN::SubGraphProtoT* subGraph, std::set<int>& inputRemove, const MNN::OpT* constOp,
                                  const std::string& inputName) {
            // Merge Const Index to Body
            for (auto& inputIndex : subGraph->inputs) {
                if (subGraph->tensors[inputIndex] == inputName) {
                    inputRemove.insert(inputIndex);
                    for (int v = 0; v < subGraph->nodes.size(); ++v) {
                        auto& subOp = subGraph->nodes[v];
                        if (subOp->type != MNN::OpType_Input) {
                            continue;
                        }
                        if (subOp->outputIndexes[0] == inputIndex) {
                            auto src              = constOp->main.AsBlob();
                            subOp->type           = MNN::OpType_Const;
                            subOp->main.type      = MNN::OpParameter_Blob;
                            subOp->main.value     = new MNN::BlobT;
                            *subOp->main.AsBlob() = *src;
                            break;
                        }
                    }
                    break;
                }
            }
            return true;
        };
        for (int subI = 0; subI < op->inputIndexes.size(); ++subI) {
            auto index      = op->inputIndexes[subI];
            auto constIndex = constOpIndexes[index];
            if (constIndex < 0) {
                continue;
            }
            // Don't support for graph shared input
            if (param->aliases_inputs[subI]->data.size() != 1) {
                continue;
            }
            auto inputName = param->aliases_inputs[subI]->data[0];
            // Don't support for const init and update next
            bool isUpdate = false;
            for (auto& update : param->aliases_updates) {
                for (auto updateName : update->data) {
                    if (updateName == inputName) {
                        isUpdate = true;
                        break;
                    }
                }
                if (isUpdate) {
                    break;
                }
            }
            if (isUpdate) {
                continue;
            }
            // Count Refcount for const tensor
            int refCount = 0;
            for (int sub = constIndex + 1; sub < net->oplists.size(); ++sub) {
                auto& subOp = net->oplists[sub];
                for (auto subIndex : subOp->inputIndexes) {
                    if (subIndex == index) {
                        refCount++;
                        break;
                    }
                }
            }
            if (refCount > 1) {
                // The const input is shared with other op
                continue;
            }
            auto& constOp = net->oplists[constIndex];
            //FUNC_PRINT_ALL(constOp->name.c_str(), s);
            MNN_ASSERT(constOp->main.type == MNN::OpParameter_Blob);

            removeConstOpIndexes.insert(constIndex);
            mergeToSubGraph(body.first, bodyInputRemove, constOp.get(), inputName);
            mergeToSubGraph(cond.first, condInputRemove, constOp.get(), inputName);
            removeInputs.insert(subI);

            modifiedSubGraph.insert(body.first);
            modifiedSubGraph.insert(cond.first);

            // Release no needed Const Memory
            constOp->main.Reset();
        }
        auto removeSubGraphInputs = [](MNN::SubGraphProtoT* subGraph, const std::set<int>& inputRemove) {
            auto originInput = std::move(subGraph->inputs);
            subGraph->inputs.clear();
            for (auto index : originInput) {
                if (inputRemove.find(index) == inputRemove.end()) {
                    subGraph->inputs.emplace_back(index);
                }
            }
        };
        removeSubGraphInputs(body.first, bodyInputRemove);
        removeSubGraphInputs(cond.first, condInputRemove);

        // Remove no use input for while op
        auto originIndexes = std::move(op->inputIndexes);
        auto aliInputs     = std::move(param->aliases_inputs);
        for (int subI = 0; subI < originIndexes.size(); ++subI) {
            if (removeInputs.find(subI) == removeInputs.end()) {
                op->inputIndexes.emplace_back(originIndexes[subI]);
                param->aliases_inputs.emplace_back(std::move(aliInputs[subI]));
            }
        }
    }
    if (removeConstOpIndexes.empty()) {
        return false;
    }
    auto originOpLists = std::move(net->oplists);
    for (int i = 0; i < originOpLists.size(); ++i) {
        if (removeConstOpIndexes.find(i) == removeConstOpIndexes.end()) {
            net->oplists.emplace_back(std::move(originOpLists[i]));
        }
    }
    // Try Optimize Subgraph for more const op get
    auto* ctx = Global<OptimizeContext>::Get();
    std::unordered_map<std::string, VARP> empty;
    for (auto mutable_subgraph : modifiedSubGraph) {
        std::unique_ptr<MNN::NetT> subnet(new MNN::NetT);
        subnet->oplists    = std::move(mutable_subgraph->nodes);
        subnet->tensorName = std::move(mutable_subgraph->tensors);
        subnet->sourceType = ctx->source;
        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;
        for (auto v: mutable_subgraph->inputs) {
            inputNames.emplace_back(subnet->tensorName[v]);
        }
        for (auto v: mutable_subgraph->outputs) {
            outputNames.emplace_back(subnet->tensorName[v]);
        }
#ifdef MNN_POST_CONVERTER_DEBUG
        for (auto& v : outputNames) {
            FUNC_PRINT_ALL(v.c_str(), s);
        }
        FUNC_PRINT_ALL(mutable_subgraph->name.c_str(), s);
#endif
        subnet->outputName = outputNames;

        std::unique_ptr<MNN::NetT> new_subnet = optimizeNetImpl(subnet, empty);
        mutable_subgraph->nodes               = std::move(subnet->oplists);

        MNN::SubGraphProtoT* new_subgraph = mutable_subgraph;
        for (int i = 0; i < inputNames.size(); ++i) {
            auto& name = inputNames[i];
            for (int v = 0; v < new_subnet->tensorName.size(); ++v) {
                if (new_subnet->tensorName[v] == name) {
                    mutable_subgraph->inputs[i] = v;
                    break;
                }
            }
        }
        for (int i = 0; i < outputNames.size(); ++i) {
            auto& name = outputNames[i];
            for (int v = 0; v < new_subnet->tensorName.size(); ++v) {
                if (new_subnet->tensorName[v] == name) {
                    mutable_subgraph->outputs[i] = v;
                    break;
                }
            }
        }
        mutable_subgraph->nodes   = std::move(new_subnet->oplists);
        mutable_subgraph->tensors = std::move(new_subnet->tensorName);
    }
    return true;
}

} // namespace Express
} // namespace MNN

using namespace MNN;
using namespace MNN::Express;
std::unique_ptr<MNN::NetT> optimizeNet(std::unique_ptr<MNN::NetT>& originNet, bool forTraining, modelConfig& config) {
    Global<modelConfig>::Reset(&config);
    std::unique_ptr<std::ofstream, void(*)(std::ofstream*)> externalFile(
        new std::ofstream(".__convert_external_data.bin", std::ios::binary),
        [](std::ofstream* fs){
            fs->close();
            delete fs;
    });
    if (externalFile.get() && externalFile->is_open() && externalFile->good()) {
        config.externalFile = externalFile.get();
    } else {
        config.externalFile = nullptr;
    }
    if (originNet->sourceType == NetSource_TENSORFLOW) {
        GenerateSubGraph(originNet);
    }
    std::vector<MNN::SubGraphProtoT*> subgraphs;
    for (auto& subgraph : originNet->subgraphs) {
        subgraphs.push_back(subgraph.get());
    }
    OptimizeContext ctx;
    ctx.subgraphs = subgraphs;
    ctx.is_training = forTraining;
    ctx.verbose = true;
    ctx.source = originNet->sourceType;
    ctx.completed_subgraphs = {};
    ctx.RunOptimize = optimizeNetImpl;

    Global<OptimizeContext>::Reset(&ctx);
    std::unordered_map<std::string, VARP> inputs, empty;
    // subgraph may depend on vars of outter subgraph or root net, getting vars of them need Program::create.
    // But program (create from unoptimize net) may have OpType_Extra op, causing vars can't do getInfo/readMap correctly,
    // then subgraph depend on it may convert failed (nullptr) or wrong (error shape)
    // RunOptimize won't use subgraph, so we can do it before other subgraph optimize safely
    std::unique_ptr<MNN::NetT> net = ctx.RunOptimize(originNet, empty);
    auto program = Program::create(net.get(), true, true);
    auto addVars = [&](std::shared_ptr<Program> program, const std::vector<std::string>& tensorName) {
        for (const auto& iter : program->vars()) {
            if (iter.first < tensorName.size() && iter.first >= 0) {
                auto name = tensorName[iter.first];
                if (inputs.find(name) == inputs.end()) {
                    inputs[name] = iter.second;
                }
            }
        }
    };
    addVars(program, net->tensorName);
    // Reversing subgraph so we iterate them by topo order (like tree traversal), so every var used by subgraph be prepared
    std::reverse(ctx.subgraphs.begin(), ctx.subgraphs.end());
    for (int idx = 0; idx < ctx.subgraphs.size(); ++idx) {
        // complete it first so OpType_Extra be removed
        CompleteSubGraph(inputs, ctx.subgraphs[idx]);
        auto new_graph = ctx.completed_subgraphs[idx];
        auto subProgram = Program::create(new_graph, true, true);
        subProgram->input(inputs, true);
        // add vars of subgraph, so inner subgraph can use them
        addVars(subProgram, new_graph->tensors);
    }
    ctx.first_run = false;
    ctx.subgraphs = std::move(ctx.completed_subgraphs);
    // from inner to upper, make some optimize for subgraph is visable to outer graph and root
    std::reverse(ctx.subgraphs.begin(), ctx.subgraphs.end());
    for (auto subgraph : ctx.subgraphs) {
        CompleteSubGraph(inputs, subgraph);
    }
    net = ctx.RunOptimize(net, empty);
    
    fuseConstIntoSubgraph(net.get(), ctx.completed_subgraphs);
    for (auto* subgraph : ctx.completed_subgraphs) {
        net->subgraphs.emplace_back(subgraph);
    }
    return std::move(net);
}
