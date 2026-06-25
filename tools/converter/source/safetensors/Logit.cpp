#include <cstring>
#include <vector>
#include <MNN/MNNDefine.h>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include "MNN_generated.h"
#include <flatbuffers/flatbuffers.h>

#include "SafetensorConverter.hpp"
#include "Logit.hpp"
#include "SafetensorModelRegistry.hpp"
#include "SafetensorUtils.hpp"
#include "WorkflowJson.hpp"

using namespace MNN::Express;
using namespace MNN::Express::SafeTensorUtils;

namespace MNN {
namespace SafeTensors {

static inline void _setNameIfEmpty(const VARP& v, const std::string& name) {
    if (nullptr != v.get() && v->name().empty()) {
        v->setName(name);
    }
}

static VARP _linear2d(VARP x4d, VARP weightOI, VARP bias = nullptr) {
    auto wInfo = weightOI->getInfo();
    if (nullptr == wInfo || wInfo->dim.size() < 2) {
        return nullptr;
    }

    const int outDim = wInfo->dim[0];
    const int inDim = wInfo->dim[1];
    if (inDim <= 0 || outDim <= 0) {
        return nullptr;
    }

    std::vector<float> weightData(weightOI->getInfo()->size);
    ::memcpy(weightData.data(), weightOI->readMap<float>(), weightData.size() * sizeof(float));
    std::vector<float> biasData(outDim, 0.0f);
    if (nullptr != bias) {
        ::memcpy(biasData.data(), bias->readMap<float>(), outDim * sizeof(float));
    }

    return _Conv(std::move(weightData), std::move(biasData), x4d, {inDim, outDim}, {1, 1});
}

// Deep-copy `src` into `dst` via flatbuffers Pack/UnPack and strip Convolution2D
// weight payloads — the runtime reuses the original logit's quantized weights.
static void _cloneLogitNet(const MNN::NetT* src, MNN::NetT* dst) {
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(MNN::CreateNet(fbb, src));
    std::unique_ptr<MNN::NetT> cloned(flatbuffers::GetRoot<MNN::Net>(fbb.GetBufferPointer())->UnPack());
    *dst = std::move(*cloned);
    for (auto& op : dst->oplists) {
        if (op && op->main.type == OpParameter_Convolution2D) {
            auto conv = op->main.AsConvolution2D();
            conv->weight.clear();
            conv->bias.clear();
            conv->quanParameter.reset();
            conv->external.clear();
        }
    }
}

// Locate logits tensor index — prefer outputName mapping, fall back to the last
// op with an output index.
static int _findLogitsIndex(const MNN::NetT* net) {
    if (!net->outputName.empty()) {
        const auto& outName = net->outputName[0];
        for (int i = 0; i < (int)net->tensorName.size(); ++i) {
            if (net->tensorName[i] == outName) return i;
        }
    }
    for (int i = (int)net->oplists.size() - 1; i >= 0; --i) {
        if (net->oplists[i] && !net->oplists[i]->outputIndexes.empty()) {
            return net->oplists[i]->outputIndexes[0];
        }
    }
    return -1;
}

static int _addTensor(MNN::NetT* net, const std::string& name) {
    int idx = (int)net->tensorName.size();
    net->tensorName.push_back(name);
    return idx;
}

static int _appendConstInt(MNN::NetT* net, const std::string& opName, const std::string& tensorName, int value) {
    int idx = _addTensor(net, tensorName);
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Const;
    op->main.type = OpParameter_Blob;
    op->main.value = new BlobT;
    auto blob = op->main.AsBlob();
    blob->dataFormat = MNN_DATA_FORMAT_NCHW;
    blob->dataType = DataType_DT_INT32;
    blob->dims = {1};
    blob->int32s = {value};
    op->name = opName;
    op->outputIndexes = {idx};
    net->oplists.emplace_back(std::move(op));
    return idx;
}

static int _appendSoftmaxOp(MNN::NetT* net, int inputIdx, const std::string& opName, const std::string& tensorName, int axis = -1) {
    int idx = _addTensor(net, tensorName);
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Softmax;
    op->main.type = OpParameter_Axis;
    op->main.value = new AxisT;
    op->main.AsAxis()->axis = axis;
    op->name = opName;
    op->inputIndexes = {inputIdx};
    op->outputIndexes = {idx};
    net->oplists.emplace_back(std::move(op));
    return idx;
}

// Returns {valuesIdx, indicesIdx}. Default largest=true (no main parameter).
static std::pair<int, int> _appendTopK2Op(MNN::NetT* net, int inputIdx, int kIdx,
                                          const std::string& opName,
                                          const std::string& valuesName,
                                          const std::string& indicesName) {
    int valuesIdx = _addTensor(net, valuesName);
    int indicesIdx = _addTensor(net, indicesName);
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_TopKV2;
    op->name = opName;
    op->inputIndexes = {inputIdx, kIdx};
    op->outputIndexes = {valuesIdx, indicesIdx};
    net->oplists.emplace_back(std::move(op));
    return {valuesIdx, indicesIdx};
}

static int _appendUnaryOp(MNN::NetT* net, int inputIdx, UnaryOpOperation kind,
                          const std::string& opName, const std::string& tensorName) {
    int idx = _addTensor(net, tensorName);
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_UnaryOp;
    op->main.type = OpParameter_UnaryOp;
    op->main.value = new UnaryOpT;
    op->main.AsUnaryOp()->opType = kind;
    op->name = opName;
    op->inputIndexes = {inputIdx};
    op->outputIndexes = {idx};
    net->oplists.emplace_back(std::move(op));
    return idx;
}

// Clone `logit` into `dst` and resolve the logits tensor index. Returns -1 on
// failure (also logs the error tagged with `fnTag`).
static int _cloneAndFindLogits(const MNN::NetT* logit, MNN::NetT* dst, const char* fnTag) {
    if (nullptr == logit || nullptr == dst) {
        return -1;
    }
    _cloneLogitNet(logit, dst);
    int idx = _findLogitsIndex(dst);
    if (idx < 0) {
        MNN_ERROR("%s: 未找到 logits 输出\n", fnTag);
    }
    return idx;
}

void LogitConvert(const Converter* converter, MNN::NetT* dst, const LogitConfig& config) {
    if (nullptr == converter || nullptr == dst) {
        return;
    }

    auto weight = converter->loadTensor(config.wteWeightName);
    if (nullptr == weight.get() && config.wteWeightName.size() > 7 && config.wteWeightName.substr(0, 7) == "module.") {
        weight = converter->loadTensor(config.wteWeightName.substr(7));
    }
    if (nullptr == weight.get() || nullptr == weight->getInfo() || weight->getInfo()->dim.size() < 2) {
        MNN_ERROR("LogitConvert: missing/invalid %s\n", config.wteWeightName.c_str());
        return;
    }

    const int d0 = weight->getInfo()->dim[0];
    const int d1 = weight->getInfo()->dim[1];
    if (d0 <= 0 || d1 <= 0) {
        MNN_ERROR("LogitConvert: invalid wte weight shape\n");
        return;
    }

    int hiddenSize = config.hiddenSize;
    int vocabSize = 0;
    bool needTranspose = false;

    if (hiddenSize > 0) {
        if (d1 == hiddenSize) {
            vocabSize = d0;
        } else if (d0 == hiddenSize) {
            vocabSize = d1;
            needTranspose = true;
        } else {
            // Fallback: assume [vocab, hidden]
            hiddenSize = d1;
            vocabSize = d0;
        }
    } else {
        // Fallback heuristics: vocab is usually larger than hidden
        if (d0 >= d1) {
            vocabSize = d0;
            hiddenSize = d1;
        } else {
            vocabSize = d1;
            hiddenSize = d0;
            needTranspose = true;
        }
    }

    if (needTranspose) {
        weight = _Transpose(weight, {1, 0}); // -> [vocab, hidden]
    }

    // Input hidden state: [B, S, H]
    auto hiddenState = _Input({1, -1, hiddenSize}, NCHW, halide_type_of<float>());
    hiddenState->setName(config.inputName);

    auto shapeHidden = _Shape(hiddenState, true);
    auto one = _Unsqueeze(_Scalar<int32_t>(1), {0});
    auto batchVar = _Slice(shapeHidden, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);
    auto seqVar = _Slice(shapeHidden, _Unsqueeze(_Scalar<int32_t>(1), {0}), one);

    auto hidden2d = _Reshape(hiddenState, {-1, hiddenSize, 1, 1});

    // Optional bias
    VARP bias = nullptr;
    auto prefix = config.wteWeightName;
    const std::string suffix = ".weight";
    if (prefix.size() > suffix.size() && prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) == 0) {
        prefix.resize(prefix.size() - suffix.size());
    }
    auto biasName = prefix + ".bias";
    if (converter->hasTensor(biasName)) {
        bias = converter->loadTensor(biasName);
    } else if (biasName.size() > 7 && biasName.substr(0, 7) == "module." && converter->hasTensor(biasName.substr(7))) {
        bias = converter->loadTensor(biasName.substr(7));
    }

    // Quantized path if weight_qscale exists
    VARP logits2d = nullptr;
    auto wScaleName = config.wteWeightName + "_qscale";
    std::string textScaleName = config.wteWeightName;
    if (textScaleName.find(".weight") != std::string::npos) {
        textScaleName.replace(textScaleName.find(".weight"), 7, ".text_embedding.weight_qscale");
    }

    auto loadScale = [&](const std::string& name) -> VARP {
        if (converter->hasTensor(name)) {
            return converter->loadTensor(name);
        }
        if (name.size() > 7 && name.substr(0, 7) == "module." && converter->hasTensor(name.substr(7))) {
            return converter->loadTensor(name.substr(7));
        }
        return nullptr;
    };

    VARP wScale = loadScale(wScaleName);
    if (nullptr == wScale.get()) {
        wScale = loadScale(textScaleName);
    }

    if (wScale.get() != nullptr) {
        logits2d = _QConvolution1x1(hiddenSize, hidden2d, nullptr, nullptr, weight, wScale, nullptr, bias, vocabSize);
    } else {
        // Float path
        if (weight->getInfo()->type.code != halide_type_float) {
            MNN_ERROR("LogitConvert: wte weight is not float and no qscale found\n");
            return;
        }
        logits2d = _linear2d(hidden2d, weight, bias);
    }

    if (nullptr == logits2d.get()) {
        MNN_ERROR("LogitConvert: build logits failed\n");
        return;
    }
    _setNameIfEmpty(logits2d, prefix + ".out2d");

    auto vocabVar = _Unsqueeze(_Scalar<int32_t>(vocabSize), {0});
    auto logits3d = _Reshape(logits2d, _Concat({batchVar, seqVar, vocabVar}, 0));
    logits3d->setName(config.outputName);

    Variable::save({logits3d}, dst);
    dst->sourceType = NetSource_ONNX;
    dst->outputName = {config.outputName};

}

void MakeTieEmbedding(const Converter* converter, const MNN::NetT* src, MNN::NetT* dst) {
    if (nullptr == converter || nullptr == src || nullptr == dst) {
        return;
    }

    std::string sharedName;
    int ic = 0;
    int oc = 0;

    for (auto& op : src->oplists) {
        if (nullptr == op) {
            continue;
        }
        if (op->type != OpType_Convolution) {
            continue;
        }
        auto conv = op->main.AsConvolution2D();
        if (nullptr == conv || nullptr == conv->common) {
            continue;
        }
        if (conv->common->inputCount <= 0 || conv->common->outputCount <= 0) {
            continue;
        }
        // Keep the last conv as shared weight provider (align with makeSharedGather.py).
        sharedName = op->name;
        ic = conv->common->inputCount;
        oc = conv->common->outputCount;
    }

    if (sharedName.empty() || ic <= 0 || oc <= 0) {
        MNN_ERROR("MakeTieEmbedding: can't find valid convolution in src\n");
        return;
    }

    // Indices input.
    auto input = _Input({-1}, NCHW, halide_type_of<int>());
    input->setName("x");

    // GatherV2 with OpParameter_Input main is a special form that lets runtime reuse
    // the quantized weights from base model's convolution execution.
    std::unique_ptr<OpT> gather(new OpT);
    gather->type = OpType_GatherV2;
    gather->main.type = OpParameter_Input;
    gather->main.value = new InputT;
    gather->main.AsInput()->dims = {oc, ic};
    gather->main.AsInput()->dtype = DataType_DT_FLOAT;
    gather->main.AsInput()->dformat = MNN_DATA_FORMAT_NCHW;

    auto gatherExpr = Expr::create(gather.get(), {input});
    gatherExpr->setName(sharedName);

    auto output = Variable::create(gatherExpr);
    output->setName(sharedName);

    Variable::save({output}, dst);
    dst->sourceType = NetSource_ONNX;
    dst->outputName = {sharedName};
}


// Clone `logit` into `dst` and append a TopKV2 producing top-K indices.
void MakeTopKV(const Converter* /*converter*/, const MNN::NetT* logit, MNN::NetT* dst, int K) {
    int logitsIdx = _cloneAndFindLogits(logit, dst, "MakeTopKV");
    if (logitsIdx < 0) return;

    int kIdx = _appendConstInt(dst, "const_topk_k", "topk_k", K);
    auto vi = _appendTopK2Op(dst, logitsIdx, kIdx, "TopKV2", "topk_values", "topk_indices");
    dst->outputName = {dst->tensorName[vi.second]};
}

// Clone `logit` into `dst` and append a Softmax (axis=-1) as the new output.
void MakeSoftmax(const Converter* /*converter*/, const MNN::NetT* logit, MNN::NetT* dst) {
    int logitsIdx = _cloneAndFindLogits(logit, dst, "MakeSoftmax");
    if (logitsIdx < 0) return;

    int smIdx = _appendSoftmaxOp(dst, logitsIdx, "LogitSoftmax", "logit_softmax");
    dst->outputName = {dst->tensorName[smIdx]};
}

// Clone `logit` into `dst`, append Softmax → TopKV2 → Log(values) for beam search.
// Outputs {log(values), indices}.
void MakeBeamTopKV(const Converter* /*converter*/, const MNN::NetT* logit, MNN::NetT* dst, int K) {
    int logitsIdx = _cloneAndFindLogits(logit, dst, "MakeBeamTopKV");
    if (logitsIdx < 0) return;

    int smIdx = _appendSoftmaxOp(dst, logitsIdx, "BeamSoftmax", "beam_softmax");
    int kIdx = _appendConstInt(dst, "const_beam_topk_k", "beam_topk_k", K);
    auto vi = _appendTopK2Op(dst, smIdx, kIdx, "BeamTopKV2", "beam_topk_values", "beam_topk_indices");
    int logIdx = _appendUnaryOp(dst, vi.first, UnaryOpOperation_LOG, "BeamTopKV2_Log", "beam_topk_log_values");
    dst->outputName = {dst->tensorName[logIdx], dst->tensorName[vi.second]};
}

namespace {

// Parse the K parameter, accepting either an int array or a single int (legacy).
// Always returns a non-empty list (defaults to {1}).
static std::vector<int> _parseKList(const rapidjson::Value& block) {
    std::vector<int> kList;
    if (auto kArr = WorkflowJson::getArray(block, "K")) {
        for (auto& kv : kArr->GetArray()) {
            if (kv.IsInt() && kv.GetInt() > 0) kList.push_back(kv.GetInt());
        }
    } else {
        int K = WorkflowJson::getInt(block, "K", 1);
        if (K > 0) kList.push_back(K);
    }
    if (kList.empty()) kList.push_back(1);
    return kList;
}

// Build and write a per-K logit variant for each entry in `kList`. External
// data is force-disabled for variants since they reuse the base logit's
// quantized weights.
template <typename Builder>
static void _saveLogitVariants(const Converter* converter, MNN::NetT* logitNet,
                               modelConfig& modelPath, const std::string& path,
                               const std::vector<int>& kList,
                               const std::string& filePrefix, Builder build) {
    auto originExternal = modelPath.saveExternalData;
    modelPath.saveExternalData = false;
    for (int K : kList) {
        auto net = std::unique_ptr<MNN::NetT>(new MNN::NetT);
        build(converter, logitNet, net.get(), K);
        modelPath.MNNModel = path + filePrefix + std::to_string(K) + ".mnn";
        optimizeAndWrite(modelPath, net);
    }
    modelPath.saveExternalData = originExternal;
}

static bool _convertLogitModel(const Converter* converter, const rapidjson::Value* model, modelConfig& modelPath) {
    if (nullptr == converter) {
        return false;
    }

    LogitConfig config;
    auto path = modelPath.MNNModel;

    auto logitNet = std::unique_ptr<MNN::NetT>(new MNN::NetT);
    std::unique_ptr<MNN::NetT> embeddingNet;

    auto ensureLogits = [&]() {
        if (logitNet->oplists.empty()) {
            LogitConvert(converter, logitNet.get(), config);
        }
    };

    if (nullptr != model && model->IsObject()) {
        auto blocks = WorkflowJson::getArray(*model, "blocks");
        if (nullptr != blocks) {
            for (auto& block : blocks->GetArray()) {
                if (!block.IsObject()) {
                    continue;
                }

                auto prefix = WorkflowJson::getString(block, "prefix");
                if (!prefix.empty()) {
                    auto wteWeightName = prefix;
                    if (wteWeightName.find(".weight") == std::string::npos) {
                        auto candidate = wteWeightName + ".weight";
                        if (converter->hasTensor(candidate)) {
                            wteWeightName = candidate;
                        }
                    }
                    if (converter->hasTensor(wteWeightName)) {
                        config.wteWeightName = wteWeightName;
                    }
                }

                auto type = WorkflowJson::getString(block, "type");
                if (type == "InnerProduct") {
                    LogitConvert(converter, logitNet.get(), config);
                    continue;
                }
                if (type == "TieEmbedding") {
                    embeddingNet = std::unique_ptr<MNN::NetT>(new MNN::NetT);
                    MakeTieEmbedding(converter, logitNet.get(), embeddingNet.get());
                    continue;
                }
                if (type == "TopKV") {
                    ensureLogits();
                    _saveLogitVariants(converter, logitNet.get(), modelPath, path,
                                       _parseKList(block), "logit_topkv_", MakeTopKV);
                    continue;
                }
                if (type == "Softmax") {
                    ensureLogits();
                    auto originExternal = modelPath.saveExternalData;
                    modelPath.saveExternalData = false;
                    auto softmaxNet = std::unique_ptr<MNN::NetT>(new MNN::NetT);
                    MakeSoftmax(converter, logitNet.get(), softmaxNet.get());
                    modelPath.MNNModel = path + "logit_softmax.mnn";
                    optimizeAndWrite(modelPath, softmaxNet);
                    modelPath.saveExternalData = originExternal;
                    continue;
                }
                if (type == "BeamTopKV") {
                    ensureLogits();
                    _saveLogitVariants(converter, logitNet.get(), modelPath, path,
                                       _parseKList(block), "logit_beam_", MakeBeamTopKV);
                    continue;
                }
            }
        }
    }

    modelPath.MNNModel = path + "logit.mnn";
    optimizeAndWrite(modelPath, logitNet);

    if (nullptr != embeddingNet.get()) {
        modelPath.MNNModel = path + "embed.mnn";
        optimizeAndWrite(modelPath, embeddingNet);
    }
    return true;
}

REGISTER_SAFETENSOR_MODEL_BUILDER("logit", _convertLogitModel);

} // namespace

} // namespace SafeTensors
} // namespace MNN
