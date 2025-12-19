#include <set>
#include <map>
#include <queue>
#include <fstream>
#include <sstream>
#include "flatbuffers/flexbuffers.h"
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/MNNFileUtils.h"
#include "shape/SizeComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Schedule.hpp"
#include "rapidjson/document.h"
#include <rapidjson/prettywriter.h>

using namespace MNN;
static bool gNeedOffline = false;
static int gMaxKVSize = 0;
static std::string gNPUName = "CoreML";
static std::string gOfflieSrc;
static std::string gOfflieDst;
static std::string gGraphName = "graph";
static std::string gCacheDir = "res";
static MNNForwardType gNPUType = MNN_FORWARD_NN;
static bool initConstTensorsNoAlloc(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net) {
    bool valid    = true;
    tensors.resize(net->tensorName()->size());
    // Set up const
    for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
        auto op = net->oplists()->GetAs<Op>(opIndex);
        if (OpType_Const == op->type() || OpType_TrainableParam == op->type()) {
            MNN_ASSERT(nullptr != op->outputIndexes());
            auto index = op->outputIndexes()->data()[0];
            tensors[index].reset(new Tensor);
            TensorUtils::getDescribe(tensors[index].get())->index = index;
            auto output    = tensors[index].get();
            if (op->type() == OpType_TrainableParam) {
                TensorUtils::getDescribe(output)->usage = Tensor::InsideDescribe::TRAINABLE;
            }
            TensorUtils::getDescribe(output)->usage = Tensor::InsideDescribe::CONSTANT;
            TensorUtils::getDescribe(output)->isMutable = false;
        }
    }
    return true;
}

static bool needComputeOp(const Op* op) {
    if (op->type() != OpType_Input && op->type() != OpType_Const && op->type() != OpType_TrainableParam) {
        return true;
    }
    return false;
}
struct SubModuleInfo {
    std::vector<int> opList;
    std::vector<int> inputs;;
    std::vector<int> outputs;
    std::vector<uint8_t> tensorMask;
    bool isBreak = false;
};
struct SubModuleIO {
    std::vector<MNN::Express::VARP> inputs;
    std::vector<MNN::Express::VARP> outputs;
    std::vector<std::vector<int>> kvcache;
};
static void _computeTensorMask(SubModuleInfo& m, const Net* net) {
    /**Compute All SubModule's inputs and outputs*/
    // 0: not use, 1: input, 2: output, 3: mid, 4: valid output
    m.tensorMask = std::vector<uint8_t>(net->tensorName()->size(), 0);
    auto& tensorMask = m.tensorMask;
    for (auto opIndex : m.opList) {
        auto op = net->oplists()->GetAs<Op>(opIndex);
        if (nullptr != op->inputIndexes()) {
            for (int v=0; v<op->inputIndexes()->size(); ++v) {
                auto index = op->inputIndexes()->data()[v];
                tensorMask[index] = tensorMask[index] | 1;
            }
        }
        if (nullptr != op->outputIndexes()) {
            for (int v=0; v<op->outputIndexes()->size(); ++v) {
                auto index = op->outputIndexes()->data()[v];
                tensorMask[index] = tensorMask[index] | 2;
            }
        }
    }
}
static bool _npuSupportOp(const Op* op) {
    if (gMaxKVSize > 0) {
        return true;
    }
    if (op->type() == OpType_Attention) {
        auto attn = op->main_as_AttentionParam();
        if (nullptr != attn && attn->kv_cache()) {
            return false;
        }
    }
    return true;
}

static bool isBreakOp(const Op* op) {
    bool isWhileControlflow = false;
    if (op->type() == OpType_While && op->main_as_WhileParam() != nullptr) {
        isWhileControlflow = true;
    }
    if (op->type() == OpType_If || isWhileControlflow || op->type() == OpType_Where || op->type() == OpType_Segment || op->type() == OpType_Unique || op->type() == OpType_NonMaxSuppressionV2) {
        return true;
    }
    if (!_npuSupportOp(op)) {
        return true;
    }
    return false;
}

static std::vector<int> _collectNeededOps(const MNN::Net* net, const std::set<int>& inputIndexes, const std::set<int>& outputIndexes) {
    // 0: not set, 1: output, 2:input
    std::vector<int> tensorMask(net->tensorName()->size());
    ::memset(tensorMask.data(), 0, tensorMask.size() * sizeof(int));

    // 0: use, 1: no use
    std::vector<int> opMask(net->oplists()->size());
    ::memset(opMask.data(), 0, opMask.size() * sizeof(int));

    // Set Initial Status
    for (auto v : outputIndexes) {
        tensorMask[v] = 1;
    }
    for (auto v : inputIndexes) {
        // If both input/output, set as input
        tensorMask[v] = 2;
    }
    bool change = false;
    do {
        change = false;
        for (int i=0; i<opMask.size(); ++i) {
            if (opMask[i] > 0) {
                continue;
            }
            auto op = net->oplists()->GetAs<Op>(i);
            if (nullptr != op->outputIndexes()) {
                for (int j=0; j<op->outputIndexes()->size(); ++j) {
                    auto index = op->outputIndexes()->data()[j];
                    if (tensorMask[index] == 1) {
                        opMask[i] = 1;
                        change = true;
                    }
                }
            }
            if (nullptr != op->inputIndexes() && opMask[i]) {
                for (int j=0; j<op->inputIndexes()->size(); ++j) {
                    auto index = op->inputIndexes()->data()[j];
                    if (tensorMask[index] != 2) {
                        tensorMask[index] = 1;
                    }
                }
            }
        }
    } while (change);

    std::vector<int> ops;
    for (int i=0; i<opMask.size(); ++i) {
        if (opMask[i] > 0) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (needComputeOp(op)) {
                ops.emplace_back(i);
                continue;
            }
        }
    }
    return ops;
}

static std::vector<int> _findBreakIndex(const SubModuleInfo& info, const Net* net, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    // 0: not used, 1: const, 2: output
    std::vector<uint8_t> constMask(sharedConst->allTensors.size(), 0);
    for (int i=0; i<sharedConst->allTensors.size(); ++i) {
        if(sharedConst->allTensors[i].get() != nullptr) {
            constMask[i] = 1;
        }
    }
    for (int v = 0; v < info.opList.size(); ++v) {
        auto op = net->oplists()->GetAs<Op>(info.opList[v]);
        if (nullptr == op->outputIndexes()) {
            continue;
        }
        bool isConst = true;
        if (nullptr != op->inputIndexes()) {
            for (int i=0; i<op->inputIndexes()->size(); ++i) {
                auto index = op->inputIndexes()->data()[i];
                if (constMask[index]) {
                    continue;
                }
                if (OpCommonUtils::opNeedContent(op, i)) {
                    isConst = false;
                    break;
                }
            }
        }
        if (isConst) {
            for (int i=0; i<op->outputIndexes()->size(); ++i) {
                auto index = op->outputIndexes()->data()[i];
                constMask[index] = 1;
            }
        }
    }
    std::vector<int> res;
    // Check Break Index
    for (int v = 0; v < info.opList.size(); ++v) {
        auto op = net->oplists()->GetAs<Op>(info.opList[v]);
        if (nullptr == op->outputIndexes() || nullptr == op->inputIndexes()) {
            continue;
        }
        int inputNum = op->inputIndexes()->size();
        auto dims = SizeComputer::needInputContent(op, inputNum);
        for (auto index : dims) {
            if (index < inputNum) {
                if (constMask[op->inputIndexes()->data()[index]] != 1) {
                    res.emplace_back(v);
                    break;
                }
            }
        }
    }
    return res;
}
static std::vector<SubModuleInfo> _splitSubModuleForShapeConst(const std::vector<SubModuleInfo>& origin, const Net* net, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    std::vector<SubModuleInfo> res;
    for (auto& m : origin) {
        if (m.isBreak) {
            res.emplace_back(std::move(m));
            continue;
        }
        auto breakIndexes = _findBreakIndex(m, net, sharedConst);
        if (breakIndexes.size() > 0) {
            int current = 0;
            for (auto breakIndex : breakIndexes) {
                // Split
                if (breakIndex > current) {
                    SubModuleInfo m0;
                    m0.opList.insert(m0.opList.begin(), m.opList.begin() + current, m.opList.begin() + breakIndex);
                    res.emplace_back(std::move(m0));
                }
                SubModuleInfo m1;
                m1.opList = {m.opList[breakIndex]};
                res.emplace_back(std::move(m1));
                current = breakIndex + 1;
            }
            if (current < m.opList.size()) {
                SubModuleInfo m2;
                m2.opList.insert(m2.opList.begin(), m.opList.begin() + current, m.opList.end());
                res.emplace_back(std::move(m2));
            }
        } else {
            res.emplace_back(std::move(m));
        }
    }
    return res;
}

static std::vector<SubModuleInfo> _createSubModuleInfo(const Net* net, const std::set<int>& inputIndexes, const std::set<int>& outputIndexes, const std::set<int>& noComputeIndexes, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    std::vector<SubModuleInfo> submodule;
    auto selectOps = _collectNeededOps(net, inputIndexes, outputIndexes);

    // Separate the graph to serveral submodule
    SubModuleInfo current;
    for (int si=0; si<selectOps.size(); ++si) {
        auto i = selectOps[si];
        auto op = net->oplists()->GetAs<Op>(i);
        if (isBreakOp(op)) {
            // TODO: Don't need split segment
            if (current.opList.size() > 0) {
                // Not empty
                submodule.emplace_back(std::move(current));
            }
            SubModuleInfo controlOp;
            controlOp.opList = {i};
            controlOp.isBreak = true;
            if (nullptr != op->inputIndexes()) {
                controlOp.inputs.resize(op->inputIndexes()->size());
                ::memcpy(controlOp.inputs.data(), op->inputIndexes()->data(), controlOp.inputs.size() * sizeof(int));
            }
            if (nullptr != op->outputIndexes()) {
                controlOp.outputs.resize(op->outputIndexes()->size());
                ::memcpy(controlOp.outputs.data(), op->outputIndexes()->data(), controlOp.outputs.size() * sizeof(int));
            }
            submodule.emplace_back(std::move(controlOp));
            continue;
        }
        current.opList.emplace_back(i);
    }
    if (!current.opList.empty()) {
        submodule.emplace_back(std::move(current));
    }
    submodule = _splitSubModuleForShapeConst(submodule, net, sharedConst);
    for (int moduleIndex=0; moduleIndex < submodule.size(); ++moduleIndex) {
        auto& m = submodule[moduleIndex];
        // Compute input / output
        if (!m.isBreak) {
            _computeTensorMask(m, net);
            for (int i=0; i<m.tensorMask.size(); ++i) {
                if (0 == m.tensorMask[i]) {
                    continue;
                }
                if (1 == m.tensorMask[i]) {
                    if (noComputeIndexes.find(i) != noComputeIndexes.end()) {
                        continue;
                    }
                    m.inputs.emplace_back(i);
                    continue;
                }
                if (2 == m.tensorMask[i]) {
                    m.outputs.emplace_back(i);
                    continue;
                }
                if (3 == m.tensorMask[i]) {
                    if (outputIndexes.find(i) != outputIndexes.end()) {
                        m.outputs.emplace_back(i);
                    }
                }
            }
        }
        // Check if the module's input is valid
        for (int i=0; i<m.inputs.size(); ++i) {
            auto index = m.inputs[i];
            if (inputIndexes.find(index) != inputIndexes.end()) {
                continue;
            }
            if (noComputeIndexes.find(index) != noComputeIndexes.end()) {
                continue;
            }
            bool find = false;
            for (int sub=0; sub < moduleIndex; ++sub) {
                for (auto out : submodule[sub].outputs) {
                    if (out == index) {
                        find = true;
                        break;
                    }
                }
                if (find) {
                    break;
                }
            }
            if (find) {
                continue;
            }
            // Find from module
            for (int sub=0; sub < moduleIndex; ++sub) {
                if (submodule[sub].tensorMask.empty()) {
                    continue;
                }
                if (submodule[sub].tensorMask[index] == 2) {
                    find = true;
                    break;
                }
                if (submodule[sub].tensorMask[index] == 3) {
                    submodule[sub].outputs.emplace_back(index);
                    submodule[sub].tensorMask[index] = 2;
                    find = true;
                    break;
                }
            }
            if (!find) {
                if (net->tensorName() != nullptr) {
                    MNN_PRINT("%d tensor [ %s ] is input but not found\n", index, net->tensorName()->GetAsString(index)->c_str());
                }
            }
            MNN_ASSERT(find);
        }
    }
    for (auto& m : submodule) {
        m.tensorMask.clear();
    }
    // sort input and output
    for (auto& m : submodule) {
        std::sort(m.inputs.begin(), m.inputs.end());
        std::sort(m.outputs.begin(), m.outputs.end());
    }
    return submodule;
}
static std::set<std::string> _getAttentionName(const void* buffer, size_t bufferSize) {
    std::set<std::string> attentionNames;
    auto net = flatbuffers::GetRoot<Net>(buffer);
    if (nullptr == net->oplists()) {
        return attentionNames;
    }
    for (int i=0; i<net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() == OpType_Attention) {
            if (nullptr != op->main_as_AttentionParam()) {
                if (op->main_as_AttentionParam()->kv_cache()) {
                    attentionNames.insert(op->name()->str());
                }
            }
        }
    }
    return attentionNames;
}
static SubModuleIO _getSubModuleIO(std::vector<MNN::Express::VARP> inputs, const SubModuleInfo& info, const void* buffer, size_t bufferSize, std::string srcpath) {
    // Deep clone output to let the module release
    SubModuleIO io;
    std::vector<std::string> inputNames(info.inputs.size());
    std::vector<std::string> outputNames(info.outputs.size());
    auto net = flatbuffers::GetRoot<Net>(buffer);
    for (int i=0; i<info.inputs.size(); ++i) {
        auto index = info.inputs[i];
        inputNames[i] = net->tensorName()->GetAsString(index)->str();
    }
    for (int i=0; i<info.outputs.size(); ++i) {
        auto index = info.outputs[i];
        outputNames[i] = net->tensorName()->GetAsString(index)->str();
    }
    auto attentionNames = _getAttentionName(buffer, bufferSize);
    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((srcpath + ".weight").c_str());
    rtmgr->setMode(MNN::Interpreter::Session_Debug);
    std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, outputNames, (const uint8_t*)buffer, bufferSize, rtmgr), MNN::Express::Module::destroy);
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() != "Attention") {
            return true;
        }
        if (attentionNames.find(opName) != attentionNames.end()) {
            auto query = ntensors[0];
            auto key = ntensors[1];
            auto value = ntensors[2];
            int seq_len = query->length(1);
            auto numHead = query->length(2);
            auto headDim = query->length(3);
            auto kvNumHead = key->length(2);
            std::vector<int> kvDims = {kvNumHead, 1, 1, headDim};
            io.kvcache.emplace_back(kvDims);
        }
        return true;
    };
    MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const MNN::OperatorInfo* info) {
        return true;
    };
    MNN::Express::ExecutorScope::Current()->setCallBack(std::move(beforeCallBack), std::move(callBack));
    auto outputs = m->onForward(inputs);
    io.inputs = inputs;
    io.outputs.resize(outputs.size());
    for (int i=0; i<outputs.size(); ++i) {
        io.outputs[i] = MNN::Express::_Clone(outputs[i], true);
    }
    return io;
}

static std::unique_ptr<MNN::OpT> _compileSubModule(const SubModuleIO& io, SubModuleInfo& info, const void* buffer, size_t bufferSize, const std::string& path, std::string srcpath, const std::string& targetNpuPath, float& cpuTotal, float& npuTotal, int shapeIndex, std::string graphicName) {
    std::vector<std::string> inputNames(info.inputs.size());
    std::vector<std::string> outputNames(info.outputs.size());
    auto net = flatbuffers::GetRoot<Net>(buffer);
    for (int i=0; i<info.inputs.size(); ++i) {
        auto index = info.inputs[i];
        inputNames[i] = net->tensorName()->GetAsString(index)->str();
    }
    for (int i=0; i<info.outputs.size(); ++i) {
        auto index = info.outputs[i];
        outputNames[i] = net->tensorName()->GetAsString(index)->str();
    }
    /** Get Output shapes*/
    std::vector<MNN::Express::Variable::Info> outputInfos(io.outputs.size());
    for (int i=0; i<outputInfos.size(); ++i) {
        outputInfos[i] = *io.outputs[i]->getInfo();
    }

    /** Make ML Model*/
    do {
        MNN::ScheduleConfig config;
        config.type = gNPUType;
        std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
        rtmgr->setExternalFile((srcpath + ".weight").c_str());
        rtmgr->setCache(path.c_str());
        rtmgr->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, gMaxKVSize);
        MNN::Express::Module::Config mdconfig;
        mdconfig.shapeMutable = false;
        std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, outputNames, (const uint8_t*)buffer, bufferSize, rtmgr, &mdconfig), MNN::Express::Module::destroy);
        auto predict = m->onForward(io.inputs);
        if (gNeedOffline) {
            break;
        }
        if (predict.size() != io.outputs.size()) {
            MNN_ERROR("Failed to compile: %s\n", path.c_str());
            info.isBreak = true;
            return nullptr;
        }
        for (int i=0; i<predict.size(); ++i) {
            auto error = io.outputs[i]-predict[i];
            error = MNN::Express::_ReduceMax(MNN::Express::_Abs(MNN::Express::_Cast<float>(error)));
            auto maxValue = MNN::Express::_ReduceMax(MNN::Express::_Abs(MNN::Express::_Cast<float>(io.outputs[i])))->readMap<float>()[0];
            if (maxValue < 0.01f) {
                maxValue = 0.01f;
            }
            auto errorf = error->readMap<float>()[0];
            if (errorf / maxValue > 0.1f) {
                MNN_ERROR("error = %f, max = %f for %s\n", errorf, maxValue, path.c_str());
                info.isBreak = true;
                return nullptr;
            }
        }
        // Compare Speed
        int testTime = 20;
        MNN_PRINT("Start to Test speed for %d times\n", testTime);
        MNN::Timer timer;
        for (int i=0; i<testTime; ++i) {
            predict = m->onForward(io.inputs);
            ((MNN::Tensor*)predict[0]->getTensor())->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        }
        auto npuCost = timer.durationInUs();
        MNN::ScheduleConfig configcpu;
        configcpu.numThread = 4;
        std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgrCPU(MNN::Express::Executor::RuntimeManager::createRuntimeManager(configcpu));
        rtmgrCPU->setExternalFile((srcpath + ".weight").c_str());
        m.reset(MNN::Express::Module::load(inputNames, outputNames, (const uint8_t*)buffer, bufferSize, rtmgrCPU, &mdconfig), MNN::Express::Module::destroy);
        predict = m->onForward(io.inputs);
        timer.reset();
        for (int i=0; i<testTime; ++i) {
            predict = m->onForward(io.inputs);
            ((MNN::Tensor*)predict[0]->getTensor())->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        }
        auto cpuCost = timer.durationInUs();
        float npuF = (float)npuCost/ 1000.0f / testTime;
        float cpuF = (float)cpuCost / 1000.0f / testTime;

        MNN_PRINT("%s, Speed Compare: NPU: %f ms : CPU: %f ms\n", path.c_str(), npuF, cpuF);
        cpuTotal += cpuF;
        npuTotal += npuF;
    } while (false);
    
    /** Fuse to Op*/
    std::unique_ptr<MNN::OpT> op(new OpT);
    op->inputIndexes = info.inputs;
    op->outputIndexes = info.outputs;
    op->name = targetNpuPath;
    op->main.Reset();
    op->type = MNN::OpType_Plugin;
    op->main.type = MNN::OpParameter_Plugin;
    op->main.value = new MNN::PluginT;
    auto extra = op->main.AsPlugin();
    extra->type = gNPUName;
    std::unique_ptr<MNN::AttributeT> attr(new MNN::AttributeT);
    attr->key = "path";
    attr->s = targetNpuPath;
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);
    attr->key = "inputs";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(inputNames.size());
    for (int i=0; i<inputNames.size(); ++i) {
        // CoreML Backend will name tensor as t + index
        attr->list->s[i] = std::string("t") + std::to_string(info.inputs[i]);
    }
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new AttributeT);
    attr->key = "allGraphName";
    attr->list.reset(new ListValueT);
    attr->list->s = {graphicName};
    extra->attr.emplace_back(std::move(attr));
    if (io.kvcache.size() > 0) {
        attr.reset(new MNN::AttributeT);
        attr->key = "state";
        attr->tensor.reset(new BlobT);
        attr->tensor->dataType = DataType_DT_UINT8;
        flexbuffers::Builder builder;
        auto start = builder.StartMap();
        builder.Int("number", io.kvcache.size() * 2);
        builder.Int("max_length", gMaxKVSize);
        builder.Int("axis", 2);
        auto shapeStart = builder.StartVector("shape");

        // Add State
        for (int i=0; i<io.kvcache.size(); ++i) {
            // Each KV has two state
            for (int j=0; j<2; ++j) {
                auto vecStart = builder.StartVector();
                for (int v=0; v<io.kvcache[i].size(); ++v) {
                    builder.Add(io.kvcache[i][v]);
                }
                builder.EndVector(vecStart, false, false);
            }
        }
        builder.EndVector(shapeStart, false, false);

        builder.EndMap(start);
        builder.Finish();
        attr->tensor->uint8s = builder.GetBuffer();
        if (false) {
            // Try Read
            auto ref = flexbuffers::GetRoot(attr->tensor->uint8s.data(), attr->tensor->uint8s.size());
            auto refMap = ref.AsMap();
            auto keys = refMap.Keys();
            int readNumber = 0;
            int maxLength = 0;
            std::vector<std::vector<int>> stateShape;
            for (int i=0; i<keys.size(); ++i) {
                auto key = keys[i].AsKey();
                if (std::string(key) == "number") {
                    readNumber = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "max_length") {
                    maxLength = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "shape") {
                    auto shapeVectors = refMap.Values()[i].AsVector();
                    for (int u=0; u<shapeVectors.size(); ++u) {
                        auto shapeV = shapeVectors[u].AsVector();
                        std::vector<int> shapes;
                        for (int v=0; v<shapeV.size(); ++v) {
                            shapes.emplace_back(shapeV[v].AsInt32());
                        }
                        stateShape.emplace_back(shapes);
                    }
                    continue;
                }
            }
            FUNC_PRINT(1);
        }
        extra->attr.emplace_back(std::move(attr));
    }

    attr.reset(new MNN::AttributeT);
    attr->key = "outputs";
    attr->list.reset(new ListValueT);
    attr->list->s.resize(outputNames.size());
    for (int i=0; i<outputNames.size(); ++i) {
        // CoreML Backend will name tensor as t + index
        attr->list->s[i] = std::string("t") + std::to_string(info.outputs[i]);
    }
    extra->attr.emplace_back(std::move(attr));
    attr.reset(new MNN::AttributeT);
    attr->key = "allInputShape";
    attr->list.reset(new ListValueT);
    std::string inputsShapeStr = "";
    for (int i = 0; i < io.inputs.size(); i++) {
        for (int j = 0; j < io.inputs[i]->getInfo()->dim.size(); j++) {
            attr->list->i.emplace_back(io.inputs[i]->getInfo()->dim[j]);
        }
    }
    extra->attr.emplace_back(std::move(attr));

    for (int i=0; i<outputInfos.size(); ++i) {
        attr.reset(new MNN::AttributeT);
        attr->key = "o_" + std::to_string(shapeIndex) + "_" + std::to_string(i);
        attr->tensor.reset(new BlobT);
        attr->tensor->dataType = OpCommonUtils::convertDataType( outputInfos[i].type);
        attr->tensor->dims = outputInfos[i].dim;
        switch(outputInfos[i].order) {
            case MNN::Express::NHWC:
                attr->tensor->dataFormat = MNN_DATA_FORMAT_NHWC;
                break;
            case MNN::Express::NCHW:
                attr->tensor->dataFormat = MNN_DATA_FORMAT_NCHW;
                break;
            case MNN::Express::NC4HW4:
                attr->tensor->dataFormat = MNN_DATA_FORMAT_NC4HW4;
                break;
            default:
                attr->tensor->dataFormat = MNN_DATA_FORMAT_NCHW;
                break;
        }
        extra->attr.emplace_back(std::move(attr));
    }
    return op;
}
static bool _fuse(MNN::NetT* net, MNN::NetT* srcNet) {
    std::map<std::string, MNN::OpT*> dstPlugin;
    for (auto& op : net->oplists) {
        if (op->type == OpType_Plugin) {
            dstPlugin.insert(std::make_pair(op->name, op.get()));
        }
    }
    for (auto& op : srcNet->oplists) {
        if (op->type != OpType_Plugin) {
            continue;
        }
        auto iter = dstPlugin.find(op->name);
        if (iter == dstPlugin.end()) {
            MNN_ERROR("Can't find plugin: %s\n", op->name.c_str());
            continue;
        }
        auto dst = iter->second->main.AsPlugin();
        auto src = op->main.AsPlugin();
        std::map<std::string, AttributeT*> dstKeys;
        for (auto& dstAttr : dst->attr) {
            dstKeys.insert(std::make_pair(dstAttr->key, dstAttr.get()));
        }
        for (auto&& srcAttr : src->attr) {
            if (srcAttr->key == "inputs" || srcAttr->key == "outputs") {
                // Don't fuse same one
                continue;
            }
            auto dstIter = dstKeys.find(srcAttr->key);
            if (dstIter == dstKeys.end()) {
                dst->attr.emplace_back(std::move(srcAttr));
                continue;
            }
            if (dstIter->second->list != nullptr && srcAttr->list != nullptr) {
                dstIter->second->list->s.insert(dstIter->second->list->s.end(), srcAttr->list->s.begin(), srcAttr->list->s.end());
                dstIter->second->list->i.insert(dstIter->second->list->i.end(), srcAttr->list->i.begin(), srcAttr->list->i.end());
            }
        }
    }
    return true;
}
static bool _reOrderOp(MNN::NetT* net) {
    auto oplist = std::move(net->oplists);
    std::set<int> validInputs;
    do {
        bool empty = true;
        for (int i=0; i<oplist.size(); ++i) {
            if (nullptr == oplist[i]) {
                continue;
            }
            bool valid = true;
            for (auto index : oplist[i]->inputIndexes) {
                if (validInputs.find(index) == validInputs.end()) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                for (auto index : oplist[i]->outputIndexes) {
                    validInputs.insert(index);
                }
                net->oplists.emplace_back(std::move(oplist[i]));
                oplist[i] = nullptr;
            } else {
                empty = false;
            }
        }
        if (empty) {
            break;
        }
    } while (true);
    return true;
}

static bool _reIndexTensor(MNN::NetT* net) {
    auto& mNet = net;
    std::map<std::string, int> tensorNameIdx;
    std::map<int, int> usefulTensorIndexMap;
    std::vector<std::string> usefulTensorName;
    // extraTensorDescribe reindex
    for (int i = 0; i < mNet->tensorName.size(); i++) {
        tensorNameIdx.insert(std::make_pair(mNet->tensorName[i], i));
    }
    for (int i = 0; i < mNet->extraTensorDescribe.size(); i++) {
        auto name = mNet->extraTensorDescribe[i]->name;
        auto iter = tensorNameIdx.find(name);
        if (iter == tensorNameIdx.end()) {
            mNet->extraTensorDescribe[i]->index = -1;
        } else {
            mNet->extraTensorDescribe[i]->index = iter->second;
        }
    }

    std::vector<bool> tensorValid(mNet->tensorName.size(), false);
    for (auto& op : mNet->oplists) {
        for (auto index : op->inputIndexes) {
            if (index < 0) {
                continue; // optional input, ignore it
            }
            tensorValid[index] = true;
        }
        for (auto index : op->outputIndexes) {
            tensorValid[index] = true;
        }
    }

    for (int i = 0; i < tensorValid.size(); ++i) {
        if (tensorValid[i]) {
            usefulTensorIndexMap.insert(std::make_pair(i, usefulTensorName.size()));
            usefulTensorName.push_back(mNet->tensorName[i]);
        }
    }

    // Re index
    for (auto& op : mNet->oplists) {
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            if (op->inputIndexes[i] < 0) {
                continue;
            }
            auto iter = usefulTensorIndexMap.find(op->inputIndexes[i]);
            op->inputIndexes[i] = iter->second;
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            auto iter = usefulTensorIndexMap.find(op->outputIndexes[i]);
            op->outputIndexes[i] = iter->second;
        }
    }

    mNet->tensorName = usefulTensorName;
    for (auto iter = mNet->extraTensorDescribe.begin(); iter != mNet->extraTensorDescribe.end();) {
        auto index = (*iter)->index;
        if (usefulTensorIndexMap.find(index) == usefulTensorIndexMap.end()) {
            iter = mNet->extraTensorDescribe.erase(iter);
            continue;
        }
        (*iter)->index = usefulTensorIndexMap.find(index)->second;
        iter++;
    }
    // Check dup name and modify
    std::set<std::string> names;
    std::set<std::string> tensorNames;
    for (int i = 0; i < mNet->oplists.size(); ++i) {
        auto& op    = mNet->oplists[i];
        auto opName = op->name;
        if (opName.empty() || names.find(opName) != names.end()) {
            std::ostringstream defaultName;
            defaultName << EnumNameOpType(op->type);
            defaultName << i;
            op->name = defaultName.str();
#ifdef DEBUG
            MNN_PRINT("%d op name is empty or dup, set to %s\n", i, op->name.c_str());
#endif
            opName = op->name;
        }
        names.insert(opName);
        for (auto output : op->outputIndexes) {
            auto origin = net->tensorName[output];
            if (origin.empty() || tensorNames.find(origin) != tensorNames.end()) {
                std::ostringstream defaultName;
                defaultName << output;
                origin                  = defaultName.str();
                net->tensorName[output] = origin;
            }
            tensorNames.insert(origin);
        }
    }
    return true;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./compilefornpu src.mnn dst.mnn npu.json\n");
        return 0;
    }
    const char* srcMNN = argv[1];
    const char* dstMNN = argv[2];
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::vector<MNN::Express::VARP>> inputs;
    std::set<std::string> skipOps;

    if (argc >= 4) {
        rapidjson::Document document;
        std::ifstream fileNames(argv[3]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        gNPUName = document["type"].GetString();
        if (gNPUName == "QNN") {
            MNN_PRINT("Convert for QNN, QualComn's NPU\n");
            gNPUType = MNN_CONVERT_QNN;
            gNeedOffline = true;
            gOfflieSrc = "";
            gOfflieDst = "bin";
        } else if (gNPUName == "MLDA") {
            MNN_PRINT("Convert for MLDA, MTK's NPU\n");
            gNPUType = MNN_CONVERT_NEUROPILOT;
            gNeedOffline = true;
            gOfflieSrc = "tflite";
            gOfflieDst = "dla";
        } else if (gNPUName == "CoreML") {
            MNN_PRINT("Convert for CoreML, Apple's framework\n");
            gNPUType = MNN_CONVERT_COREML;
            gNeedOffline = true;
            gOfflieSrc = "";
            gOfflieDst = "";
        } else {
            MNN_PRINT("Use Native NPU compute\n");
        }
        if (document.HasMember("cache")) {
            gCacheDir = document["cache"].GetString();
            FUNC_PRINT_ALL(gCacheDir.c_str(), s);
            MNNCreateDir(gCacheDir.c_str());
        }
        if (document.HasMember("graph_name")) {
            gGraphName = document["graph_name"].GetString();
        }
        if (document.HasMember("skips")) {
            auto skips = document["skips"].GetArray();
            for (auto iter = skips.Begin(); iter != skips.End(); iter++) {
                skipOps.insert(iter->GetString());
            }
        }
        if (document.HasMember("KVCACHE_SIZE_LIMIT")) {
            gMaxKVSize = document["KVCACHE_SIZE_LIMIT"].GetInt();
        }
        if (document.HasMember("testdir")) {
            auto testdir = document["testdir"].GetArray();
            for (auto iter = testdir.Begin(); iter != testdir.End(); iter++) {
                std::string dirname = iter->GetString();
                auto subinputs = MNN::Express::Variable::load((dirname + "/input.mnn").c_str());
                inputs.emplace_back(subinputs);
                inputNames.clear();
                for (int i=0; i<subinputs.size(); ++i) {
                    inputNames.emplace_back(subinputs[i]->name());
                }
                auto outputs = MNN::Express::Variable::load((dirname + "/output.mnn").c_str());
                outputNames.clear();
                for (int i=0; i<outputs.size(); ++i) {
                    outputNames.emplace_back(outputs[i]->name());
                }
            }
        }
    }
    if (outputNames.empty()) {
        std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, outputNames, srcMNN), MNN::Express::Module::destroy);
        auto minfo = m->getInfo();
        outputNames = minfo->outputNames;
        inputNames = minfo->inputNames;
        std::vector<MNN::Express::VARP> subinputs;
        subinputs.resize(minfo->inputs.size());
        for (int i=0; i<minfo->inputs.size(); ++i) {
            auto& info = minfo->inputs[i];
            auto varp = MNN::Express::_Input(info.dim, info.order, info.type);
            varp->writeMap<void>();
            subinputs[i] = varp;
        }
        inputs = {subinputs};
    }
    // Registor size computor
    MNN::Express::Executor::getGlobalExecutor();
    // Get Net struct
    std::shared_ptr<MNN::Interpreter> netC(MNN::Interpreter::createFromFile(srcMNN), MNN::Interpreter::destroy);
    auto bufferPair = netC->getModelBuffer();
    std::shared_ptr<Schedule::ScheduleInfo> sharedConst;
    auto buffer = bufferPair.first;
    auto length = bufferPair.second;
    auto net = GetNet(buffer);
    std::map<std::string, int> tensorIndexMap;
    for (int i=0; i<net->tensorName()->size(); ++i) {
        auto tname = net->tensorName()->GetAsString(i)->str();
        tensorIndexMap.insert(std::make_pair(tname, i));
    }
    // Extra Const Tensors
    sharedConst.reset(new Schedule::ScheduleInfo);
    std::vector<std::shared_ptr<Tensor>> allTensors;
    sharedConst->allTensors.resize(net->tensorName()->size());
    initConstTensorsNoAlloc(sharedConst->allTensors, net);
    std::set<int> noneedComputeIndexes;
    for (int i=0; i<sharedConst->allTensors.size(); ++i) {
        if (sharedConst->allTensors[i].get() != nullptr) {
            noneedComputeIndexes.insert(i);
        }
    }

    std::set<int> inputIndexes;
    std::set<int> outputIndexes;
    std::map<std::string, int> outputIndexesMap;
    for (int i=0; i<net->tensorName()->size(); ++i) {
        auto tname = net->tensorName()->GetAsString(i)->str();
        for (int j=0; j<inputNames.size(); ++j) {
            if (tname == inputNames[j]) {
                inputIndexes.emplace(i);
                break;
            }
        }
        for (int j=0; j<outputNames.size(); ++j) {
            if (tname == outputNames[j]) {
                outputIndexes.emplace(i);
                outputIndexesMap.insert(std::make_pair(tname, i));
                break;
            }
        }
    }
    if (outputIndexesMap.size() != outputNames.size()) {
        MNN_ERROR("PipelineModule:: Can't find enough output from the model, finded is:\n");
        for (auto& iter : outputIndexesMap) {
            MNN_ERROR("[ %s ] ", iter.first.c_str());
        }
        MNN_ERROR("\n");
    }
    auto firstInputIndex = inputIndexes;
    std::set<int> firstOutputIndex;
    for (int i=0; i<net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (skipOps.find(op->name()->str()) != skipOps.end()) {
            MNN_PRINT("Skip %s op\n", op->name()->c_str());
            auto outputSize = op->outputIndexes()->size();
            for (int v=0; v<outputSize; ++v) {
                firstOutputIndex.insert(op->outputIndexes()->data()[v]);
            }
        }
    }
    std::vector<bool> keepOp(net->oplists()->size(), false);
    {
        auto subModulesInfo = _createSubModuleInfo(net, inputIndexes, outputIndexes, noneedComputeIndexes, sharedConst);
        for (int moduleIndex=0; moduleIndex<subModulesInfo.size(); ++moduleIndex) {
            auto moduleInfo = subModulesInfo[moduleIndex];
            for (auto& index : moduleInfo.opList) {
                keepOp[index] = true;
            }
        }
    }
    if (!firstOutputIndex.empty()) {
        // Get New Inputs
        std::vector<std::string> firstOutputNames;

        // Compute New Input
        for (auto output : firstOutputIndex) {
            inputIndexes.insert(output);
            firstOutputNames.emplace_back(net->tensorName()->GetAsString(output)->str());
        }
        std::vector<std::string> newInputNames;
        for (auto index : inputIndexes) {
            newInputNames.emplace_back(net->tensorName()->GetAsString(index)->str());
        }
        std::shared_ptr<MNN::Express::Module> m(MNN::Express::Module::load(inputNames, firstOutputNames, (const uint8_t*)bufferPair.first, bufferPair.second), MNN::Express::Module::destroy);
        for (int i=0; i<inputs.size(); ++i) {
            std::map<std::string, MNN::Express::VARP> vars;
            for (int v=0; v<inputNames.size(); ++v) {
                vars.insert(std::make_pair(inputNames[v], inputs[i][v]));
            }
            auto outputs = m->onForward(inputs[i]);
            for (int v=0; v<firstOutputNames.size(); ++v) {
                vars.insert(std::make_pair(firstOutputNames[v], outputs[v]));
            }
            inputs[i].clear();
            for (int v=0; v<newInputNames.size(); ++v) {
                inputs[i].emplace_back(vars[newInputNames[v]]);
            }
        }
        inputNames = newInputNames;
    }
    // Split Module
    auto subModulesInfo = _createSubModuleInfo(net, inputIndexes, outputIndexes, noneedComputeIndexes, sharedConst);
    // TODO: Insert pass to split submodule to npu and not npu
    std::map<std::string, std::vector<std::string>> merges;
    std::vector<std::shared_ptr<NetT>> allNets;
    for (int inputIndex=0; inputIndex < inputs.size(); ++inputIndex) {
        std::map<int, MNN::Express::VARP> stackes;
        // Compute module's io
        for (int i=0; i<net->tensorName()->size(); ++i) {
            auto tname = net->tensorName()->GetAsString(i)->str();
            for (int j=0; j<inputNames.size(); ++j) {
                if (tname == inputNames[j]) {
                    stackes.insert(std::make_pair(i, inputs[inputIndex][j]));
                    break;
                }
            }
        }
        std::vector<SubModuleIO> moduleIO(subModulesInfo.size());
        for (int i=0; i<subModulesInfo.size(); ++i) {
            auto& current = subModulesInfo[i];
            std::vector<MNN::Express::VARP> subInputs;
            for (auto index : current.inputs) {
                subInputs.emplace_back(stackes[index]);
            }
            moduleIO[i] = _getSubModuleIO(subInputs, current, bufferPair.first, bufferPair.second, srcMNN);
            for (int j=0; j<current.outputs.size(); ++j) {
                stackes.insert(std::make_pair(current.outputs[j], moduleIO[i].outputs[j]));
            }
        }
        for (int i=0; i<subModulesInfo.size(); ++i) {
            if (subModulesInfo[i].isBreak) {
                continue;
            }
            bool hasConvolution = false;
            for (auto opIndex : subModulesInfo[i].opList) {
                auto op = net->oplists()->GetAs<Op>(opIndex);
                if (op->type() == OpType_Convolution) {
                    hasConvolution = true;
                    break;
                }
            }
            if (!hasConvolution) {
                subModulesInfo[i].isBreak = true;
            }
        }
        // Compile NPU Module
        std::vector<std::unique_ptr<OpT>> npuOps(subModulesInfo.size());
        int npuIndex = 0;
        float npuTotal = 0.0f;
        float cpuTotal = 0.0f;

        for (int i=0; i<subModulesInfo.size(); ++i) {
            if (!subModulesInfo[i].isBreak) {
                auto path = gCacheDir + "/" + gGraphName + std::to_string(npuIndex);
                if (!gOfflieDst.empty()) {
                    path += ("." + gOfflieDst);
                }
                std::string srcPath;
                std::string graphicName;
                if (inputIndex == 0) {
                    srcPath = gCacheDir + "/" + gGraphName +  std::to_string(npuIndex);
                    graphicName = gGraphName +  std::to_string(npuIndex);
                } else {
                    srcPath = gCacheDir + "/" + gGraphName + std::to_string(inputIndex) + "_" +  std::to_string(npuIndex);
                    graphicName = gGraphName + std::to_string(inputIndex) + "_" +  std::to_string(npuIndex);
                }
                if (!gOfflieSrc.empty()) {
                    srcPath += ("." + gOfflieSrc);
                }
                if (merges.find(path) != merges.end()) {
                    merges[path].emplace_back(srcPath);
                } else {
                    merges.insert(std::make_pair(path, std::vector<std::string>{srcPath}));
                }
                npuOps[i] = std::move(_compileSubModule(moduleIO[i], subModulesInfo[i], bufferPair.first, bufferPair.second, srcPath, srcMNN, path, cpuTotal, npuTotal, inputIndex, graphicName));
                npuIndex++;
            }
        }
        MNN_PRINT("Total Speed Compare: NPU: %f ms : CPU: %f ms\n", npuTotal, cpuTotal);
        // Merge to dst
        std::shared_ptr<MNN::NetT> dstNet(flatbuffers::GetRoot<Net>(bufferPair.first)->UnPack());
        for (int i=0; i<keepOp.size(); ++i) {
            if (dstNet->oplists[i]->inputIndexes.empty()) {
                continue;
            }
            if (!keepOp[i]) {
                dstNet->oplists[i].reset();
            }
        }
        for (int moduleIndex=0; moduleIndex<subModulesInfo.size(); ++moduleIndex) {
            auto moduleInfo = subModulesInfo[moduleIndex];
            if (moduleInfo.isBreak) {
                continue;
            }
            for (auto& index : moduleInfo.opList) {
                dstNet->oplists[index].reset();
            }
            dstNet->oplists[moduleInfo.opList[0]] = std::move(npuOps[moduleIndex]);
        }
        auto oplist = std::move(dstNet->oplists);
        for (auto& op : oplist) {
            if (nullptr != op.get()) {
                dstNet->oplists.emplace_back(std::move(op));
            }
        }
        _reIndexTensor(dstNet.get());
        _reOrderOp(dstNet.get());
        allNets.emplace_back(std::move(dstNet));
    }
    // Fuse And Store
    auto dstNet = allNets[0].get();
    for (int i=1; i<allNets.size(); ++i) {
        _fuse(dstNet, allNets[i].get());
        allNets[i].reset();
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Net::Pack(builder, dstNet));
    std::ofstream outputOs(dstMNN, std::ios::binary);
    outputOs.write((const char*)builder.GetBufferPointer(), builder.GetSize());

    // Write Merge Info
    rapidjson::Document resDocument;
    resDocument.SetObject();
    rapidjson::Value mergeMessages;
    mergeMessages.SetObject();
    for (auto& iter : merges) {
        rapidjson::Value mergeSrc;
        mergeSrc.SetArray();
        for (auto& v : iter.second) {
            rapidjson::Value vt;
            vt.SetString(v.c_str(), resDocument.GetAllocator());
            mergeSrc.GetArray().PushBack(vt, resDocument.GetAllocator());
        }
        rapidjson::Value key;
        key.SetString(iter.first.c_str(), resDocument.GetAllocator());
        mergeMessages.AddMember(key, mergeSrc, resDocument.GetAllocator());
    }
    {
        rapidjson::Value type;
        type.SetString(gNPUName.c_str(), resDocument.GetAllocator());
        resDocument.AddMember("type", type, resDocument.GetAllocator());
    }
    resDocument.AddMember("merge", mergeMessages, resDocument.GetAllocator());
    {
        rapidjson::Value cachedir;
        cachedir.SetString(gCacheDir.c_str(), resDocument.GetAllocator());
        resDocument.AddMember("cache", cachedir, resDocument.GetAllocator());
    }
    rapidjson::StringBuffer buf;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> bufwriter(buf);
    resDocument.Accept(bufwriter);
    MNN_PRINT("Write config to npu_postreat.json\n");
    std::ofstream os("npu_postreat.json");
    os << buf.GetString();

    return 0;
}
