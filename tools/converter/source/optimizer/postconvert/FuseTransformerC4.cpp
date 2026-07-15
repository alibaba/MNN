//
//  FuseTransformerC4.cpp
//  MNNConverter
//
//  Created by MNN on 2026/06/23.
//

#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace MNN;

namespace {

static bool sameConsumers(const std::unordered_map<int, std::vector<int>>& consumers, int tensor, int opIdx) {
    auto iter = consumers.find(tensor);
    return iter != consumers.end() && iter->second.size() == 1 && iter->second[0] == opIdx;
}

static std::unique_ptr<OpT> cloneOp(const OpT* op) {
    if (op == nullptr) {
        return nullptr;
    }
    flatbuffers::FlatBufferBuilder builder(1024);
    builder.Finish(Op::Pack(builder, op));
    return std::unique_ptr<OpT>(flatbuffers::GetRoot<Op>(builder.GetBufferPointer())->UnPack());
}

static std::vector<std::unique_ptr<OpT>> cloneOps(const std::vector<std::unique_ptr<OpT>>& ops) {
    std::vector<std::unique_ptr<OpT>> copy;
    copy.reserve(ops.size());
    for (auto& op : ops) {
        copy.emplace_back(cloneOp(op.get()));
    }
    return copy;
}

struct PreConvertMatch {
    bool valid;
    int reshapeIdx;
    int convertIdx;
    int convertOut;
    int inputTensor;
    int hiddenSize;
    std::vector<int> convUsers;
    PreConvertMatch() : valid(false), reshapeIdx(-1), convertIdx(-1), convertOut(-1), inputTensor(-1), hiddenSize(0) {}
};

struct PostConvertMatch {
    bool valid;
    int convIdx;
    int convertIdx;
    int reshapeIdx;
    int convOut;
    int reshapeOut;
    PostConvertMatch() : valid(false), convIdx(-1), convertIdx(-1), reshapeIdx(-1), convOut(-1), reshapeOut(-1) {}
};

struct RopeInputMatch {
    bool valid;
    int input;
    int channel;
    std::vector<int> removeIndexes;
    RopeInputMatch() : valid(false), input(-1), channel(0) {}
};

struct ProjectionExpressionMatch {
    bool valid;
    PostConvertMatch projection;
    std::vector<int> c4Ops;
    std::map<int, int> c4Constants;
    ProjectionExpressionMatch() : valid(false) {}
};

struct TensorReplacement {
    int oldTensor;
    int newTensor;
    TensorReplacement(int oldValue, int newValue) : oldTensor(oldValue), newTensor(newValue) {}
};

struct LinearAttentionC4Plan {
    bool valid;
    bool shortConv;
    int attentionIdx;
    int outputConvIdx;
    int outputPreConvertOut;
    int outputInputC4;
    int finalReshapeIdx;
    int finalValueDim;
    std::vector<TensorReplacement> replacements;
    std::vector<int> c4Ops;
    std::map<int, int> c4Constants;
    std::set<int> removeIndexes;
    LinearAttentionC4Plan()
        : valid(false),
          shortConv(false),
          attentionIdx(-1),
          outputConvIdx(-1),
          outputPreConvertOut(-1),
          outputInputC4(-1),
          finalReshapeIdx(-1),
          finalValueDim(0) {}
};

class TransformerC4Graph {
public:
    TransformerC4Graph(std::vector<std::unique_ptr<OpT>>& ops, std::vector<std::string>& tensors,
                       const std::set<int>& graphOutputs)
        : mOps(ops), mTensors(tensors), mGraphOutputs(graphOutputs) {}

    bool run() {
        auto optimizedOps = cloneOps(mOps);
        auto optimizedTensors = mTensors;
        TransformerC4Graph optimizedGraph(optimizedOps, optimizedTensors, mGraphOutputs);
        if (optimizedGraph.runFusePipeline() && optimizedGraph.validateOptimizedGraph()) {
            mOps.swap(optimizedOps);
            mTensors.swap(optimizedTensors);
            return true;
        }
        if (!hasRoPE()) {
            return false;
        }

        auto fallbackOps = cloneOps(mOps);
        auto fallbackTensors = mTensors;
        TransformerC4Graph fallbackGraph(fallbackOps, fallbackTensors, mGraphOutputs);
        fallbackGraph.ensureRoPEInputsC4();
        if (!fallbackGraph.validateOptimizedGraph()) {
            MNN_ERROR("FuseTransformerC4: unable to build a valid C4 RoPE fallback graph.\n");
            return false;
        }
        MNN_PRINT("FuseTransformerC4: use the C4 RoPE fallback graph.\n");
        mOps.swap(fallbackOps);
        mTensors.swap(fallbackTensors);
        return true;
    }

private:
    bool runFusePipeline() {
        bool changed = false;
        changed |= fuseAttentionOutputC4();
        changed |= fuseMulSilu();
        changed |= fuseMlpOutputC4();
        changed |= fuseRoPEInputC4();
        changed |= fuseAttentionValueC4();
        changed |= ensureRoPEInputsC4();
        changed |= fuseHiddenStateC4Regions();
        changed |= fuseBinaryLayerNormC4();
        return changed;
    }

    std::vector<std::unique_ptr<OpT>>& mOps;
    std::vector<std::string>& mTensors;
    const std::set<int> mGraphOutputs;
    std::unordered_map<int, int> mProducer;
    std::unordered_map<int, std::vector<int>> mConsumers;

    bool hasRoPE() const {
        for (const auto& op : mOps) {
            if (op != nullptr && op->type == OpType_RoPE) {
                return true;
            }
        }
        return false;
    }

    void rebuildMaps() {
        mProducer.clear();
        mConsumers.clear();
        for (int i = 0; i < (int)mOps.size(); ++i) {
            auto op = mOps[i].get();
            if (op == nullptr) {
                continue;
            }
            for (auto output : op->outputIndexes) {
                mProducer[output] = i;
            }
            for (auto input : op->inputIndexes) {
                mConsumers[input].push_back(i);
            }
        }
    }

    int producerOf(int tensor) const {
        auto iter = mProducer.find(tensor);
        if (iter == mProducer.end()) {
            return -1;
        }
        return iter->second;
    }

    bool replaceInput(int opIdx, int oldTensor, int newTensor) {
        if (opIdx < 0 || opIdx >= (int)mOps.size()) {
            return false;
        }
        bool changed = false;
        auto& inputs = mOps[opIdx]->inputIndexes;
        for (int i = 0; i < (int)inputs.size(); ++i) {
            if (inputs[i] == oldTensor) {
                inputs[i] = newTensor;
                changed = true;
            }
        }
        return changed;
    }

    int buildTensor(const std::string& name) {
        for (int i = 0; i < (int)mTensors.size(); ++i) {
            if (mTensors[i] == name) {
                return i;
            }
        }
        int index = (int)mTensors.size();
        mTensors.push_back(name);
        return index;
    }

    int buildUniqueTensor(const std::string& prefix) {
        int suffix = (int)mTensors.size();
        while (true) {
            auto name = prefix + "_" + std::to_string(suffix++);
            bool exists = false;
            for (int i = 0; i < (int)mTensors.size(); ++i) {
                if (mTensors[i] == name) {
                    exists = true;
                    break;
                }
            }
            if (exists) {
                continue;
            }
            int index = (int)mTensors.size();
            mTensors.push_back(name);
            return index;
        }
    }

    void removeOps(const std::set<int>& removeIndexes) {
        if (removeIndexes.empty()) {
            return;
        }
        std::vector<std::unique_ptr<OpT>> newOps;
        newOps.reserve(mOps.size() - removeIndexes.size());
        for (int i = 0; i < (int)mOps.size(); ++i) {
            if (removeIndexes.find(i) == removeIndexes.end()) {
                newOps.emplace_back(std::move(mOps[i]));
            }
        }
        mOps.swap(newOps);
    }

    static bool isReshape(OpT* op) {
        return op != nullptr && op->type == OpType_Reshape && op->main.type == OpParameter_Reshape &&
               op->main.AsReshape() != nullptr;
    }

    static bool isConvert(OpT* op, MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest) {
        if (op == nullptr || op->type != OpType_ConvertTensor || op->main.type != OpParameter_TensorConvertInfo ||
            op->main.AsTensorConvertInfo() == nullptr) {
            return false;
        }
        auto convert = op->main.AsTensorConvertInfo();
        return convert->source == source && convert->dest == dest;
    }

    static bool isConvolution(OpT* op) {
        return op != nullptr && op->type == OpType_Convolution && op->main.type == OpParameter_Convolution2D &&
               op->main.AsConvolution2D() != nullptr;
    }

    static bool isLayerNorm(OpT* op) {
        return op != nullptr && op->type == OpType_LayerNorm && op->main.type == OpParameter_LayerNorm &&
               op->main.AsLayerNorm() != nullptr;
    }

    static bool isBinaryOp(OpT* op, BinaryOpOperation opType) {
        return op != nullptr && op->type == OpType_BinaryOp && op->main.type == OpParameter_BinaryOp &&
               op->main.AsBinaryOp() != nullptr && op->main.AsBinaryOp()->opType == opType;
    }

    static int convolutionOutputCount(OpT* op) {
        if (!isConvolution(op) || op->main.AsConvolution2D()->common == nullptr) {
            return 0;
        }
        return op->main.AsConvolution2D()->common->outputCount;
    }

    static bool dimsEqual(const std::vector<int>& dims, const std::vector<int>& expected) { return dims == expected; }

    int singleConsumer(int tensor) const {
        auto iter = mConsumers.find(tensor);
        if (iter == mConsumers.end() || iter->second.size() != 1) {
            return -1;
        }
        return iter->second[0];
    }

    std::vector<int> constInt32s(int tensorIdx) const {
        std::vector<int> values;
        int opIdx = producerOf(tensorIdx);
        if (opIdx < 0) {
            return values;
        }
        auto op = mOps[opIdx].get();
        if (op == nullptr || op->type != OpType_Const || op->main.type != OpParameter_Blob ||
            op->main.AsBlob() == nullptr) {
            return values;
        }
        auto blob = op->main.AsBlob();
        if (blob->int32s.empty()) {
            return values;
        }
        values.reserve(blob->int32s.size());
        for (auto value : blob->int32s) {
            values.push_back(value);
        }
        return values;
    }

    int findConstIntTensor(int value) const {
        for (auto& opPtr : mOps) {
            auto op = opPtr.get();
            if (op == nullptr || op->outputIndexes.size() != 1) {
                continue;
            }
            auto values = constInt32s(op->outputIndexes[0]);
            if (values.size() == 1 && values[0] == value) {
                return op->outputIndexes[0];
            }
        }
        return -1;
    }

    RoPEParamT* ropeParam(OpT* op) const {
        if (op == nullptr || op->type != OpType_RoPE || op->main.type != OpParameter_RoPEParam) {
            return nullptr;
        }
        return op->main.AsRoPEParam();
    }

    RoPEParamT* attentionRoPEParam(OpT* attention) const {
        if (attention == nullptr || attention->inputIndexes.size() < 2) {
            return nullptr;
        }
        int ropeIdx = producerOf(attention->inputIndexes[0]);
        if (ropeIdx < 0 || ropeIdx != producerOf(attention->inputIndexes[1])) {
            return nullptr;
        }
        return ropeParam(mOps[ropeIdx].get());
    }

    std::unique_ptr<OpT> makeReshape(const std::string& name, int input, int output, const std::vector<int>& dims) {
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_Reshape;
        op->name = name;
        op->inputIndexes = {input};
        op->outputIndexes = {output};
        op->main.type = OpParameter_Reshape;
        op->main.value = new ReshapeT;
        op->main.AsReshape()->dims = dims;
        op->main.AsReshape()->dimType = MNN_DATA_FORMAT_NCHW;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
        return op;
    }

    std::unique_ptr<OpT> makeConvert(const std::string& name, int input, int output, MNN_DATA_FORMAT source,
                                     MNN_DATA_FORMAT dest) {
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_ConvertTensor;
        op->name = name;
        op->inputIndexes = {input};
        op->outputIndexes = {output};
        op->main.type = OpParameter_TensorConvertInfo;
        op->main.value = new TensorConvertInfoT;
        op->main.AsTensorConvertInfo()->source = source;
        op->main.AsTensorConvertInfo()->dest = dest;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
        return op;
    }

    bool fuseAttentionOutputC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto op = mOps[idx].get();
            if (op->type != OpType_Attention || op->main.type != OpParameter_AttentionParam ||
                op->main.AsAttentionParam() == nullptr || op->main.AsAttentionParam()->output_c4 ||
                op->outputIndexes.size() != 1) {
                continue;
            }
            int attentionOut = op->outputIndexes[0];
            auto reshapeUsers = mConsumers[attentionOut];
            if (reshapeUsers.size() == 1) {
                int reshapeIdx = reshapeUsers[0];
                auto reshape = mOps[reshapeIdx].get();
                if (isReshape(reshape) && reshape->outputIndexes.size() == 1) {
                    auto dims = reshape->main.AsReshape()->dims;
                    auto convertUsers = mConsumers[reshape->outputIndexes[0]];
                    if (dims.size() == 4 && dims[2] == 1 && dims[3] == 1 && convertUsers.size() == 1) {
                        int convertIdx = convertUsers[0];
                        auto convert = mOps[convertIdx].get();
                        if (isConvert(convert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) &&
                            convert->outputIndexes.size() == 1) {
                            auto convUsers = mConsumers[convert->outputIndexes[0]];
                            if (convUsers.size() == 1 && isConvolution(mOps[convUsers[0]].get())) {
                                op->main.AsAttentionParam()->output_c4 = true;
                                op->outputIndexes = convert->outputIndexes;
                                removeIndexes.insert(reshapeIdx);
                                removeIndexes.insert(convertIdx);
                                continue;
                            }
                        }
                    }
                }
            }

            int mulIdx = singleConsumer(attentionOut);
            if (mulIdx < 0 || !isBinaryOp(mOps[mulIdx].get(), BinaryOpOperation_MUL)) {
                continue;
            }
            auto mul = mOps[mulIdx].get();
            if (mul->inputIndexes.size() != 2 || mul->outputIndexes.size() != 1) {
                continue;
            }
            int gateTensor = mul->inputIndexes[0] == attentionOut ? mul->inputIndexes[1] : mul->inputIndexes[0];
            auto ropeConfig = attentionRoPEParam(op);
            int outputChannels = ropeConfig == nullptr ? 0 : ropeConfig->num_head * ropeConfig->head_dim;
            if (outputChannels <= 0 || outputChannels % 4 != 0) {
                continue;
            }
            auto gateExpression = matchProjectionExpression(gateTensor, mulIdx, outputChannels);
            PreConvertMatch outputPre;
            int outputConvIdx = -1;
            if (!gateExpression.valid ||
                !findOutputProjection(mul->outputIndexes[0], outputChannels, &outputPre, &outputConvIdx)) {
                continue;
            }

            for (int opIdx = 0; opIdx < (int)mOps.size(); ++opIdx) {
                replaceInput(opIdx, gateExpression.projection.reshapeOut, gateExpression.projection.convOut);
            }
            for (int opIdx : gateExpression.c4Ops) {
                mOps[opIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
            for (auto& constant : gateExpression.c4Constants) {
                auto constOp = mOps[constant.first].get();
                if (constOp == nullptr || constOp->main.type != OpParameter_Blob || constOp->main.AsBlob() == nullptr) {
                    continue;
                }
                auto blob = constOp->main.AsBlob();
                blob->dims = {1, constant.second, 1, 1};
                blob->dataFormat = MNN_DATA_FORMAT_NC4HW4;
                constOp->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
            replaceInput(outputConvIdx, outputPre.convertOut, mul->outputIndexes[0]);
            op->main.AsAttentionParam()->output_c4 = true;
            op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            mul->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            removeIndexes.insert(gateExpression.projection.convertIdx);
            removeIndexes.insert(gateExpression.projection.reshapeIdx);
            removeIndexes.insert(outputPre.reshapeIdx);
            removeIndexes.insert(outputPre.convertIdx);
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
    }

    bool fuseMulSilu() {
        rebuildMaps();
        std::unordered_map<int, int> consumerCount;
        for (auto& iter : mConsumers) {
            consumerCount[iter.first] = (int)iter.second.size();
        }
        std::set<int> removeIndexes;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto op = mOps[idx].get();
            if (op->type != OpType_BinaryOp || op->main.type != OpParameter_BinaryOp ||
                op->main.AsBinaryOp()->opType != BinaryOpOperation_MUL || op->inputIndexes.size() != 2) {
                continue;
            }
            int siluPos = -1;
            OpT* siluOp = nullptr;
            for (int i = 0; i < 2; ++i) {
                int producerIdx = producerOf(op->inputIndexes[i]);
                if (producerIdx < 0 || consumerCount[op->inputIndexes[i]] != 1) {
                    continue;
                }
                auto candidate = mOps[producerIdx].get();
                if (candidate->type == OpType_UnaryOp && candidate->main.type == OpParameter_UnaryOp &&
                    candidate->main.AsUnaryOp()->opType == UnaryOpOperation_SILU) {
                    siluPos = i;
                    siluOp = candidate;
                    break;
                }
            }
            if (siluOp == nullptr || siluOp->inputIndexes.empty() || siluOp->outputIndexes.empty()) {
                continue;
            }
            int gateInput = siluOp->inputIndexes[0];
            int upInput = op->inputIndexes[1 - siluPos];
            int gateProducerIdx = producerOf(gateInput);
            int upProducerIdx = producerOf(upInput);
            if (gateProducerIdx < 0 || upProducerIdx < 0) {
                continue;
            }
            auto gateProducer = mOps[gateProducerIdx].get();
            auto upProducer = mOps[upProducerIdx].get();
            if (!isReshape(gateProducer) || !isReshape(upProducer) ||
                gateProducer->main.AsReshape()->dims != upProducer->main.AsReshape()->dims) {
                continue;
            }
            op->inputIndexes = {upInput, gateInput};
            op->main.AsBinaryOp()->opType = BinaryOpOperation_MUL_SILU;
            int siluIdx = producerOf(siluOp->outputIndexes[0]);
            if (siluIdx >= 0) {
                removeIndexes.insert(siluIdx);
            }
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
    }

    PostConvertMatch matchLinearPost(int tensorIdx) const {
        PostConvertMatch match;
        int reshapeIdx = producerOf(tensorIdx);
        if (reshapeIdx < 0) {
            return match;
        }
        auto reshape = mOps[reshapeIdx].get();
        if (!isReshape(reshape) || reshape->inputIndexes.size() != 1 || reshape->main.AsReshape()->dims.size() != 3) {
            return match;
        }
        int convertIdx = producerOf(reshape->inputIndexes[0]);
        if (convertIdx < 0) {
            return match;
        }
        auto convert = mOps[convertIdx].get();
        if (!isConvert(convert, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW) || convert->inputIndexes.size() != 1) {
            return match;
        }
        int convIdx = producerOf(convert->inputIndexes[0]);
        if (convIdx < 0 || mOps[convIdx]->type != OpType_Convolution) {
            return match;
        }
        match.valid = true;
        match.convIdx = convIdx;
        match.convertIdx = convertIdx;
        match.reshapeIdx = reshapeIdx;
        match.convOut = convert->inputIndexes[0];
        match.reshapeOut = tensorIdx;
        return match;
    }

    bool fuseMlpOutputC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto op = mOps[idx].get();
            if (op->type != OpType_BinaryOp || op->main.type != OpParameter_BinaryOp ||
                op->main.AsBinaryOp()->opType != BinaryOpOperation_MUL_SILU || op->inputIndexes.size() != 2 ||
                op->outputIndexes.size() != 1) {
                continue;
            }
            auto upPost = matchLinearPost(op->inputIndexes[0]);
            auto gatePost = matchLinearPost(op->inputIndexes[1]);
            if (!upPost.valid || !gatePost.valid) {
                continue;
            }
            auto downUsers = mConsumers[op->outputIndexes[0]];
            if (downUsers.size() != 1) {
                continue;
            }
            int downReshapeIdx = downUsers[0];
            auto downReshape = mOps[downReshapeIdx].get();
            if (!isReshape(downReshape) || downReshape->outputIndexes.size() != 1) {
                continue;
            }
            auto downConvertUsers = mConsumers[downReshape->outputIndexes[0]];
            if (downConvertUsers.size() != 1) {
                continue;
            }
            int downConvertIdx = downConvertUsers[0];
            auto downConvert = mOps[downConvertIdx].get();
            if (!isConvert(downConvert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) ||
                downConvert->outputIndexes.size() != 1) {
                continue;
            }
            auto downConvUsers = mConsumers[downConvert->outputIndexes[0]];
            if (downConvUsers.size() != 1 || mOps[downConvUsers[0]]->type != OpType_Convolution) {
                continue;
            }
            op->inputIndexes = {mOps[upPost.convIdx]->outputIndexes[0], mOps[gatePost.convIdx]->outputIndexes[0]};
            op->outputIndexes = downConvert->outputIndexes;
            removeIndexes.insert(upPost.convertIdx);
            removeIndexes.insert(upPost.reshapeIdx);
            removeIndexes.insert(gatePost.convertIdx);
            removeIndexes.insert(gatePost.reshapeIdx);
            removeIndexes.insert(downReshapeIdx);
            removeIndexes.insert(downConvertIdx);
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
    }

    bool fuseBinaryLayerNormC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        std::set<int> fusedLayerNorms;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto op = mOps[idx].get();
            if (!isBinaryOp(op, BinaryOpOperation_ADD) || op->inputIndexes.size() != 2 ||
                op->outputIndexes.size() != 1 || op->defaultDimentionFormat != MNN_DATA_FORMAT_NC4HW4) {
                continue;
            }
            int layerNormIdx = -1;
            auto users = mConsumers[op->outputIndexes[0]];
            for (int i = 0; i < (int)users.size(); ++i) {
                int userIdx = users[i];
                if (fusedLayerNorms.find(userIdx) != fusedLayerNorms.end()) {
                    continue;
                }
                auto user = mOps[userIdx].get();
                if (!isLayerNorm(user) || user->inputIndexes != op->outputIndexes || user->outputIndexes.size() != 1) {
                    continue;
                }
                if (user->defaultDimentionFormat != MNN_DATA_FORMAT_NC4HW4) {
                    continue;
                }
                layerNormIdx = userIdx;
                break;
            }
            if (layerNormIdx < 0 || layerNormIdx <= idx) {
                continue;
            }
            bool hasEarlierUser = false;
            for (int userIdx : users) {
                if (userIdx != layerNormIdx && userIdx < layerNormIdx) {
                    hasEarlierUser = true;
                    break;
                }
            }
            if (hasEarlierUser) {
                continue;
            }
            auto layerNorm = mOps[layerNormIdx].get();
            int normOutput = layerNorm->outputIndexes[0];
            layerNorm->inputIndexes = op->inputIndexes;
            layerNorm->outputIndexes = {op->outputIndexes[0], normOutput};
            layerNorm->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            removeIndexes.insert(idx);
            fusedLayerNorms.insert(layerNormIdx);
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
    }

    RopeInputMatch matchRoPEInput(int tensorIdx, int ropeIdx) const {
        RopeInputMatch match;
        std::vector<int> removeChain;
        int current = tensorIdx;
        while (true) {
            int opIdx = producerOf(current);
            if (opIdx < 0) {
                return match;
            }
            auto op = mOps[opIdx].get();
            if (!isReshape(op)) {
                break;
            }
            if (removeChain.empty()) {
                if (!sameConsumers(mConsumers, current, ropeIdx)) {
                    return match;
                }
            } else if (mConsumers.find(current) == mConsumers.end() || mConsumers.at(current).size() != 1) {
                return match;
            }
            if (op->inputIndexes.empty()) {
                return match;
            }
            removeChain.push_back(opIdx);
            current = op->inputIndexes[0];
        }
        if (removeChain.empty()) {
            return match;
        }
        int convertIdx = producerOf(current);
        if (convertIdx < 0) {
            return match;
        }
        auto convert = mOps[convertIdx].get();
        if (!isConvert(convert, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW) ||
            mConsumers.find(current) == mConsumers.end() || mConsumers.at(current).size() != 1 ||
            convert->inputIndexes.size() != 1 || convert->outputIndexes.size() != 1) {
            return match;
        }
        int convIdx = producerOf(convert->inputIndexes[0]);
        if (convIdx < 0) {
            return match;
        }
        auto conv = mOps[convIdx].get();
        if (conv->type != OpType_Convolution || conv->outputIndexes.size() != 1 ||
            !sameConsumers(mConsumers, conv->outputIndexes[0], convertIdx)) {
            return match;
        }
        int outputCount = convolutionOutputCount(conv);
        auto postReshape = mOps[removeChain.back()].get();
        auto postDims = postReshape->main.AsReshape()->dims;
        if (outputCount <= 0 || postDims.size() != 3 || postDims[0] != 1 || postDims[2] != outputCount) {
            return match;
        }
        match.valid = true;
        match.input = conv->outputIndexes[0];
        match.channel = outputCount;
        match.removeIndexes = removeChain;
        match.removeIndexes.push_back(convertIdx);
        return match;
    }

    RopeInputMatch matchAttentionValue(int tensorIdx, int attentionIdx) const {
        RopeInputMatch match;
        std::vector<int> removeChain;
        int current = tensorIdx;
        while (true) {
            int opIdx = producerOf(current);
            if (opIdx < 0) {
                return match;
            }
            auto op = mOps[opIdx].get();
            if (!isReshape(op)) {
                break;
            }
            if (removeChain.empty()) {
                if (!sameConsumers(mConsumers, current, attentionIdx)) {
                    return match;
                }
            } else if (mConsumers.find(current) == mConsumers.end() || mConsumers.at(current).size() != 1) {
                return match;
            }
            if (op->inputIndexes.empty()) {
                return match;
            }
            removeChain.push_back(opIdx);
            current = op->inputIndexes[0];
        }
        if (removeChain.empty()) {
            return match;
        }
        int convertIdx = producerOf(current);
        if (convertIdx < 0) {
            return match;
        }
        auto convert = mOps[convertIdx].get();
        if (!isConvert(convert, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW) ||
            mConsumers.find(current) == mConsumers.end() || mConsumers.at(current).size() != 1 ||
            convert->inputIndexes.size() != 1 || convert->outputIndexes.size() != 1) {
            return match;
        }
        int convIdx = producerOf(convert->inputIndexes[0]);
        if (convIdx < 0) {
            return match;
        }
        auto conv = mOps[convIdx].get();
        if (conv->type != OpType_Convolution || conv->outputIndexes.size() != 1 ||
            !sameConsumers(mConsumers, conv->outputIndexes[0], convertIdx)) {
            return match;
        }
        int outputCount = convolutionOutputCount(conv);
        auto postReshape = mOps[removeChain.back()].get();
        auto postDims = postReshape->main.AsReshape()->dims;
        if (outputCount <= 0 || postDims.size() != 3 || postDims[0] != 1 || postDims[2] != outputCount) {
            return match;
        }
        match.valid = true;
        match.input = conv->outputIndexes[0];
        match.channel = outputCount;
        match.removeIndexes = removeChain;
        match.removeIndexes.push_back(convertIdx);
        return match;
    }

    bool fuseRoPEInputC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto rope = mOps[idx].get();
            auto ropeConfig = ropeParam(rope);
            if (ropeConfig == nullptr || rope->inputIndexes.size() < 2) {
                continue;
            }
            auto qMatch = matchRoPEInput(rope->inputIndexes[0], idx);
            auto kMatch = matchRoPEInput(rope->inputIndexes[1], idx);
            int headDim = ropeConfig->head_dim > 0 ? ropeConfig->head_dim : ropeConfig->rope_cut_head_dim;
            if (!qMatch.valid || !kMatch.valid || headDim <= 0 || qMatch.channel % headDim != 0 ||
                kMatch.channel % headDim != 0) {
                continue;
            }
            int numHead = qMatch.channel / headDim;
            int kvNumHead = kMatch.channel / headDim;
            if ((ropeConfig->num_head > 0 && ropeConfig->num_head != numHead) ||
                (ropeConfig->kv_num_head > 0 && ropeConfig->kv_num_head != kvNumHead)) {
                continue;
            }

            rope->inputIndexes[0] = qMatch.input;
            rope->inputIndexes[1] = kMatch.input;
            ropeConfig->num_head = numHead;
            ropeConfig->kv_num_head = kvNumHead;
            ropeConfig->head_dim = headDim;
            removeIndexes.insert(qMatch.removeIndexes.begin(), qMatch.removeIndexes.end());
            removeIndexes.insert(kMatch.removeIndexes.begin(), kMatch.removeIndexes.end());
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
    }

    bool fuseAttentionValueC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto attention = mOps[idx].get();
            if (attention->type != OpType_Attention || attention->inputIndexes.size() < 3) {
                continue;
            }
            auto match = matchAttentionValue(attention->inputIndexes[2], idx);
            if (!match.valid) {
                continue;
            }
            attention->inputIndexes[2] = match.input;
            removeIndexes.insert(match.removeIndexes.begin(), match.removeIndexes.end());
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
    }

    bool ensureRoPEInputsC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        std::map<int, std::vector<std::unique_ptr<OpT>>> insertBefore;
        bool changed = false;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto rope = mOps[idx].get();
            auto param = ropeParam(rope);
            if (param == nullptr || rope->inputIndexes.size() < 4 || rope->outputIndexes.size() != 2 ||
                param->num_head <= 0 || param->kv_num_head <= 0 || param->head_dim <= 0) {
                continue;
            }
            const int qChannels = param->num_head * param->head_dim;
            const int kChannels = param->kv_num_head * param->head_dim;
            auto qMatch = matchRoPEInput(rope->inputIndexes[0], idx);
            auto kMatch = matchRoPEInput(rope->inputIndexes[1], idx);
            auto prepareInput = [&](int inputPosition, int channels, const RopeInputMatch& match,
                                    const std::string& prefix) {
                if (match.valid && match.channel == channels) {
                    rope->inputIndexes[inputPosition] = match.input;
                    removeIndexes.insert(match.removeIndexes.begin(), match.removeIndexes.end());
                    return;
                }
                int originalInput = rope->inputIndexes[inputPosition];
                if (isPackedRoPEInput(originalInput, channels)) {
                    return;
                }
                int reshaped = buildUniqueTensor(prefix + "_reshape");
                int packed = buildUniqueTensor(prefix + "_convert");
                insertBefore[idx].push_back(
                    makeReshape(mTensors[reshaped], originalInput, reshaped, {-1, channels, 1, 1}));
                insertBefore[idx].push_back(makeConvert(mTensors[packed], reshaped, packed, MNN_DATA_FORMAT_NCHW,
                                                        MNN_DATA_FORMAT_NC4HW4));
                rope->inputIndexes[inputPosition] = packed;
            };
            prepareInput(0, qChannels, qMatch, "FuseTransformerC4_rope_q_input");
            prepareInput(1, kChannels, kMatch, "FuseTransformerC4_rope_k_input");

            changed = true;
        }
        if (!changed) {
            return false;
        }

        std::vector<std::unique_ptr<OpT>> newOps;
        newOps.reserve(mOps.size() + insertBefore.size() * 4);
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto beforeIter = insertBefore.find(idx);
            if (beforeIter != insertBefore.end()) {
                for (auto& op : beforeIter->second) {
                    newOps.emplace_back(std::move(op));
                }
            }
            if (removeIndexes.find(idx) == removeIndexes.end()) {
                newOps.emplace_back(std::move(mOps[idx]));
            }
        }
        mOps.swap(newOps);
        return true;
    }

    bool isPackedRoPEInput(int tensor, int channels) const {
        int producerIdx = producerOf(tensor);
        if (producerIdx < 0) {
            return false;
        }
        auto producer = mOps[producerIdx].get();
        if (isConvolution(producer)) {
            return convolutionOutputCount(producer) == channels;
        }
        if (!isConvert(producer, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) ||
            producer->inputIndexes.size() != 1) {
            return false;
        }
        int reshapeIdx = producerOf(producer->inputIndexes[0]);
        if (reshapeIdx < 0 || !isReshape(mOps[reshapeIdx].get())) {
            return false;
        }
        const auto& dims = mOps[reshapeIdx]->main.AsReshape()->dims;
        return dimsEqual(dims, {-1, channels, 1, 1});
    }

    PreConvertMatch matchPreConvertFromConv(int convIdx) const {
        PreConvertMatch match;
        if (convIdx < 0 || convIdx >= (int)mOps.size()) {
            return match;
        }
        auto conv = mOps[convIdx].get();
        if (!isConvolution(conv) || conv->inputIndexes.empty()) {
            return match;
        }
        int convertIdx = producerOf(conv->inputIndexes[0]);
        if (convertIdx < 0) {
            return match;
        }
        auto convert = mOps[convertIdx].get();
        if (!isConvert(convert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) || convert->inputIndexes.size() != 1 ||
            convert->outputIndexes.size() != 1 || convert->outputIndexes[0] != conv->inputIndexes[0]) {
            return match;
        }
        int reshapeIdx = producerOf(convert->inputIndexes[0]);
        if (reshapeIdx < 0) {
            return match;
        }
        auto reshape = mOps[reshapeIdx].get();
        if (!isReshape(reshape) || reshape->inputIndexes.size() != 1 || reshape->outputIndexes.size() != 1 ||
            reshape->outputIndexes[0] != convert->inputIndexes[0]) {
            return match;
        }
        const auto& dims = reshape->main.AsReshape()->dims;
        if (dims.size() != 4 || dims[0] != -1 || dims[1] <= 0 || dims[2] != 1 || dims[3] != 1 || dims[1] % 4 != 0) {
            return match;
        }
        int convertOut = convert->outputIndexes[0];
        if (!sameConsumers(mConsumers, reshape->outputIndexes[0], convertIdx)) {
            return match;
        }
        auto consumers = mConsumers.find(convertOut);
        if (consumers != mConsumers.end()) {
            for (int userIdx : consumers->second) {
                if (isConvolution(mOps[userIdx].get())) {
                    match.convUsers.push_back(userIdx);
                }
            }
        }
        if (match.convUsers.empty()) {
            return match;
        }
        match.valid = true;
        match.reshapeIdx = reshapeIdx;
        match.convertIdx = convertIdx;
        match.convertOut = convertOut;
        match.inputTensor = reshape->inputIndexes[0];
        match.hiddenSize = dims[1];
        return match;
    }

    std::vector<PreConvertMatch> findPreConvertMatchesFromInput(int inputTensor, int hiddenSize) const {
        std::vector<PreConvertMatch> matches;
        std::set<int> seenConvert;
        auto consumers = mConsumers.find(inputTensor);
        if (consumers == mConsumers.end()) {
            return matches;
        }
        for (int reshapeIdx : consumers->second) {
            auto reshape = mOps[reshapeIdx].get();
            if (!isReshape(reshape) || reshape->inputIndexes.size() != 1 || reshape->outputIndexes.size() != 1) {
                continue;
            }
            const auto& dims = reshape->main.AsReshape()->dims;
            if (!dimsEqual(dims, std::vector<int>{-1, hiddenSize, 1, 1})) {
                continue;
            }
            int convertIdx = singleConsumer(reshape->outputIndexes[0]);
            if (convertIdx < 0 || seenConvert.find(convertIdx) != seenConvert.end()) {
                continue;
            }
            auto convert = mOps[convertIdx].get();
            if (!isConvert(convert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) ||
                convert->inputIndexes.size() != 1 || convert->outputIndexes.size() != 1) {
                continue;
            }
            PreConvertMatch match;
            match.reshapeIdx = reshapeIdx;
            match.convertIdx = convertIdx;
            match.convertOut = convert->outputIndexes[0];
            match.inputTensor = inputTensor;
            match.hiddenSize = hiddenSize;
            auto convUsers = mConsumers.find(match.convertOut);
            if (convUsers == mConsumers.end()) {
                continue;
            }
            for (int userIdx : convUsers->second) {
                if (isConvolution(mOps[userIdx].get())) {
                    match.convUsers.push_back(userIdx);
                }
            }
            if (match.convUsers.empty()) {
                continue;
            }
            match.valid = true;
            seenConvert.insert(convertIdx);
            matches.push_back(match);
        }
        return matches;
    }

    bool findPreConvertForConv(const std::vector<PreConvertMatch>& matches, int convIdx, PreConvertMatch* match) const {
        for (auto& candidate : matches) {
            for (int userIdx : candidate.convUsers) {
                if (userIdx == convIdx) {
                    if (match != nullptr) {
                        *match = candidate;
                    }
                    return true;
                }
            }
        }
        return false;
    }

    PostConvertMatch matchPostConvertFromConv(int convIdx, int hiddenSize) const {
        PostConvertMatch match;
        if (convIdx < 0 || convIdx >= (int)mOps.size()) {
            return match;
        }
        auto conv = mOps[convIdx].get();
        if (!isConvolution(conv) || conv->outputIndexes.size() != 1) {
            return match;
        }
        int convOut = conv->outputIndexes[0];
        int convertIdx = singleConsumer(convOut);
        if (convertIdx < 0) {
            return match;
        }
        auto convert = mOps[convertIdx].get();
        if (!isConvert(convert, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW) || convert->inputIndexes.size() != 1 ||
            convert->inputIndexes[0] != convOut || convert->outputIndexes.size() != 1) {
            return match;
        }
        int reshapeIdx = singleConsumer(convert->outputIndexes[0]);
        if (reshapeIdx < 0) {
            return match;
        }
        auto reshape = mOps[reshapeIdx].get();
        if (!isReshape(reshape) || reshape->inputIndexes.size() != 1 ||
            reshape->inputIndexes[0] != convert->outputIndexes[0] || reshape->outputIndexes.size() != 1 ||
            !dimsEqual(reshape->main.AsReshape()->dims, std::vector<int>{1, -1, hiddenSize})) {
            return match;
        }
        match.valid = true;
        match.convIdx = convIdx;
        match.convertIdx = convertIdx;
        match.reshapeIdx = reshapeIdx;
        match.convOut = convOut;
        match.reshapeOut = reshape->outputIndexes[0];
        return match;
    }

    int blobElementCount(const BlobT* blob) const {
        if (blob == nullptr) {
            return 0;
        }
        if (blob->dims.empty()) {
            return 1;
        }
        int count = 1;
        for (int dim : blob->dims) {
            if (dim <= 0) {
                return 0;
            }
            count *= dim;
        }
        return count;
    }

    bool matchC4Constant(int tensor, int consumerIdx, int expectedChannels, ProjectionExpressionMatch* match) const {
        int constIdx = producerOf(tensor);
        if (constIdx < 0) {
            return false;
        }
        auto op = mOps[constIdx].get();
        if (op == nullptr || op->type != OpType_Const || op->main.type != OpParameter_Blob ||
            op->main.AsBlob() == nullptr) {
            return false;
        }
        int elementCount = blobElementCount(op->main.AsBlob());
        if (elementCount == 1) {
            return true;
        }
        if (elementCount != expectedChannels || match == nullptr) {
            return false;
        }
        if (consumerIdx < 0 || consumerIdx >= (int)mOps.size()) {
            return false;
        }
        auto consumer = mOps[consumerIdx].get();
        if (consumer == nullptr || consumer->type != OpType_BinaryOp || consumer->main.type != OpParameter_BinaryOp ||
            consumer->main.AsBinaryOp() == nullptr || !sameConsumers(mConsumers, tensor, consumerIdx)) {
            return false;
        }
        match->c4Constants[constIdx] = expectedChannels;
        return true;
    }

    ProjectionExpressionMatch matchProjectionExpression(int tensor, int consumerIdx, int expectedChannels) const {
        ProjectionExpressionMatch result;
        if (expectedChannels <= 0) {
            return result;
        }

        std::function<bool(int, int, ProjectionExpressionMatch*)> walk;
        walk = [&](int currentTensor, int currentConsumer, ProjectionExpressionMatch* current) -> bool {
            if (current == nullptr || !sameConsumers(mConsumers, currentTensor, currentConsumer)) {
                return false;
            }
            auto post = matchLinearPost(currentTensor);
            if (post.valid) {
                if (convolutionOutputCount(mOps[post.convIdx].get()) != expectedChannels) {
                    return false;
                }
                current->projection = post;
                current->valid = true;
                return true;
            }

            int producerIdx = producerOf(currentTensor);
            if (producerIdx < 0) {
                return false;
            }
            auto producer = mOps[producerIdx].get();
            if (producer == nullptr || producer->outputIndexes.size() != 1 ||
                producer->outputIndexes[0] != currentTensor) {
                return false;
            }
            if (producer->type == OpType_UnaryOp || producer->type == OpType_Cast) {
                if (producer->inputIndexes.size() != 1 || !walk(producer->inputIndexes[0], producerIdx, current)) {
                    return false;
                }
                current->c4Ops.push_back(producerIdx);
                return true;
            }
            if (isReshape(producer)) {
                auto reshape = producer->main.AsReshape();
                if (producer->inputIndexes.empty() || reshape->dimType == MNN_DATA_FORMAT_NHWC ||
                    !walk(producer->inputIndexes[0], producerIdx, current)) {
                    return false;
                }
                current->c4Ops.push_back(producerIdx);
                return true;
            }
            if (producer->type != OpType_BinaryOp || producer->main.type != OpParameter_BinaryOp ||
                producer->inputIndexes.size() != 2) {
                return false;
            }

            ProjectionExpressionMatch first;
            if (walk(producer->inputIndexes[0], producerIdx, &first) &&
                matchC4Constant(producer->inputIndexes[1], producerIdx, expectedChannels, &first)) {
                first.c4Ops.push_back(producerIdx);
                *current = first;
                return true;
            }
            ProjectionExpressionMatch second;
            if (walk(producer->inputIndexes[1], producerIdx, &second) &&
                matchC4Constant(producer->inputIndexes[0], producerIdx, expectedChannels, &second)) {
                second.c4Ops.push_back(producerIdx);
                *current = second;
                return true;
            }
            return false;
        };

        walk(tensor, consumerIdx, &result);
        return result;
    }

    bool findOutputProjection(int inputTensor, int inputChannels, PreConvertMatch* preMatch, int* convIdx) const {
        auto matches = findPreConvertMatchesFromInput(inputTensor, inputChannels);
        int matchedConv = -1;
        PreConvertMatch matchedPre;
        for (auto& candidate : matches) {
            for (int candidateConv : candidate.convUsers) {
                if (matchedConv >= 0 && matchedConv != candidateConv) {
                    return false;
                }
                matchedConv = candidateConv;
                matchedPre = candidate;
            }
        }
        if (matchedConv < 0) {
            return false;
        }
        if (preMatch != nullptr) {
            *preMatch = matchedPre;
        }
        if (convIdx != nullptr) {
            *convIdx = matchedConv;
        }
        return true;
    }

    int findAddConsumerWithInputs(int tensor, int otherTensor) const {
        auto consumers = mConsumers.find(tensor);
        if (consumers == mConsumers.end()) {
            return -1;
        }
        int matched = -1;
        for (int userIdx : consumers->second) {
            auto user = mOps[userIdx].get();
            if (!isBinaryOp(user, BinaryOpOperation_ADD) || user->inputIndexes.size() != 2 ||
                user->outputIndexes.size() != 1) {
                continue;
            }
            bool hasTensor = user->inputIndexes[0] == tensor || user->inputIndexes[1] == tensor;
            bool hasOther = user->inputIndexes[0] == otherTensor || user->inputIndexes[1] == otherTensor;
            if (!hasTensor || !hasOther) {
                continue;
            }
            if (matched >= 0) {
                return -1;
            }
            matched = userIdx;
        }
        return matched;
    }

    int findLayerNormConsumer(int tensor) const {
        auto consumers = mConsumers.find(tensor);
        if (consumers == mConsumers.end()) {
            return -1;
        }
        int matched = -1;
        for (int userIdx : consumers->second) {
            auto user = mOps[userIdx].get();
            if (!isLayerNorm(user) || user->inputIndexes.size() != 1 || user->inputIndexes[0] != tensor ||
                user->outputIndexes.size() != 1) {
                continue;
            }
            if (matched >= 0) {
                return -1;
            }
            matched = userIdx;
        }
        return matched;
    }

    int findLayerNormConsumerWithInputs(int tensor, int otherTensor) const {
        auto consumers = mConsumers.find(tensor);
        if (consumers == mConsumers.end()) {
            return -1;
        }
        int matched = -1;
        for (int userIdx : consumers->second) {
            auto user = mOps[userIdx].get();
            if (!isLayerNorm(user) || user->inputIndexes.size() != 2 || user->outputIndexes.size() != 2) {
                continue;
            }
            bool hasTensor = user->inputIndexes[0] == tensor || user->inputIndexes[1] == tensor;
            bool hasOther = user->inputIndexes[0] == otherTensor || user->inputIndexes[1] == otherTensor;
            if (!hasTensor || !hasOther) {
                continue;
            }
            if (matched >= 0) {
                return -1;
            }
            matched = userIdx;
        }
        return matched;
    }

    int findSingleConvolutionConsumer(int tensor) const {
        auto consumers = mConsumers.find(tensor);
        if (consumers == mConsumers.end()) {
            return -1;
        }
        int matched = -1;
        for (int userIdx : consumers->second) {
            if (!isConvolution(mOps[userIdx].get())) {
                continue;
            }
            if (matched >= 0) {
                return -1;
            }
            matched = userIdx;
        }
        return matched;
    }

    int findSliceConsumer(int tensor) const {
        auto consumers = mConsumers.find(tensor);
        if (consumers == mConsumers.end() || consumers->second.size() != 1) {
            return -1;
        }
        int userIdx = consumers->second[0];
        auto user = mOps[userIdx].get();
        if (user == nullptr || user->type != OpType_StridedSlice ||
            user->main.type != OpParameter_StridedSliceParam || user->main.AsStridedSliceParam() == nullptr ||
            user->inputIndexes.empty() || user->inputIndexes[0] != tensor || user->outputIndexes.size() != 1) {
            return -1;
        }
        return userIdx;
    }
    struct HiddenBlockPlan {
        int blockIdx;
        int attentionIdx;
        int ropeIdx;
        int hiddenSize;
        int blockHidden;
        int blockInputSource;
        int inputLnIdx;
        int inputLnOut;
        int blockReshapeIdx;
        std::vector<PreConvertMatch> attentionPre;
        PostConvertMatch attentionPost;
        int attentionAddIdx;
        int attentionAddOut;
        bool attentionAddFused;
        int postLnIdx;
        int postLnOut;
        PreConvertMatch gatePre;
        PreConvertMatch upPre;
        int mlpBinaryIdx;
        PostConvertMatch downPost;
        int mlpAddIdx;
        int mlpAddOut;
        LinearAttentionC4Plan linearAttention;
        HiddenBlockPlan()
            : blockIdx(0),
              attentionIdx(-1),
              ropeIdx(-1),
              hiddenSize(0),
              blockHidden(-1),
              blockInputSource(-1),
              inputLnIdx(-1),
              inputLnOut(-1),
              blockReshapeIdx(-1),
              attentionAddIdx(-1),
              attentionAddOut(-1),
              attentionAddFused(false),
              postLnIdx(-1),
              postLnOut(-1),
              mlpBinaryIdx(-1),
              mlpAddIdx(-1),
              mlpAddOut(-1) {}
    };

    bool matchMlpFromPostLayerNorm(HiddenBlockPlan* plan) const {
        auto preMatches = findPreConvertMatchesFromInput(plan->postLnOut, plan->hiddenSize);
        if (preMatches.empty()) {
            return false;
        }
        int mulSiluIdx = -1;
        PreConvertMatch firstPre;
        PreConvertMatch secondPre;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto op = mOps[idx].get();
            if (op == nullptr || op->type != OpType_BinaryOp || op->main.type != OpParameter_BinaryOp ||
                op->main.AsBinaryOp() == nullptr ||
                (op->main.AsBinaryOp()->opType != BinaryOpOperation_MUL_SILU &&
                 op->main.AsBinaryOp()->opType != BinaryOpOperation_MUL) ||
                op->inputIndexes.size() != 2 || op->outputIndexes.size() != 1) {
                continue;
            }
            int firstConv = producerOf(op->inputIndexes[0]);
            int secondConv = producerOf(op->inputIndexes[1]);
            PreConvertMatch candidateFirst;
            PreConvertMatch candidateSecond;
            if (firstConv < 0 || secondConv < 0 || firstConv == secondConv ||
                !findPreConvertForConv(preMatches, firstConv, &candidateFirst) ||
                !findPreConvertForConv(preMatches, secondConv, &candidateSecond)) {
                continue;
            }
            if (mulSiluIdx >= 0) {
                return false;
            }
            mulSiluIdx = idx;
            firstPre = candidateFirst;
            secondPre = candidateSecond;
        }
        if (mulSiluIdx < 0) {
            return false;
        }
        auto mulSilu = mOps[mulSiluIdx].get();
        int downConvIdx = findSingleConvolutionConsumer(mulSilu->outputIndexes[0]);
        if (downConvIdx < 0) {
            return false;
        }
        auto downPost = matchPostConvertFromConv(downConvIdx, plan->hiddenSize);
        if (!downPost.valid) {
            return false;
        }
        int mlpAddIdx = findAddConsumerWithInputs(downPost.reshapeOut, plan->attentionAddOut);
        if (mlpAddIdx < 0) {
            return false;
        }
        plan->gatePre = firstPre;
        plan->upPre = secondPre;
        plan->mlpBinaryIdx = mulSiluIdx;
        plan->downPost = downPost;
        plan->mlpAddIdx = mlpAddIdx;
        plan->mlpAddOut = mOps[mlpAddIdx]->outputIndexes[0];
        return true;
    }

    bool matchHiddenBlockFromAttention(int attentionIdx, HiddenBlockPlan* plan) const {
        if (attentionIdx < 0 || attentionIdx >= (int)mOps.size() || plan == nullptr) {
            return false;
        }
        auto attention = mOps[attentionIdx].get();
        if (attention == nullptr || attention->type != OpType_Attention || attention->inputIndexes.size() < 3 ||
            attention->outputIndexes.size() != 1) {
            return false;
        }
        int ropeIdx = producerOf(attention->inputIndexes[0]);
        if (ropeIdx < 0 || ropeIdx != producerOf(attention->inputIndexes[1])) {
            return false;
        }
        auto rope = mOps[ropeIdx].get();
        if (rope == nullptr || rope->type != OpType_RoPE || rope->inputIndexes.size() < 2) {
            return false;
        }
        int qConvIdx = producerOf(rope->inputIndexes[0]);
        int kConvIdx = producerOf(rope->inputIndexes[1]);
        int vConvIdx = producerOf(attention->inputIndexes[2]);
        if (qConvIdx < 0 || kConvIdx < 0 || vConvIdx < 0) {
            return false;
        }
        auto qPre = matchPreConvertFromConv(qConvIdx);
        auto kPre = matchPreConvertFromConv(kConvIdx);
        auto vPre = matchPreConvertFromConv(vConvIdx);
        if (!qPre.valid || !kPre.valid || !vPre.valid || qPre.inputTensor != kPre.inputTensor ||
            qPre.inputTensor != vPre.inputTensor || qPre.hiddenSize != kPre.hiddenSize ||
            qPre.hiddenSize != vPre.hiddenSize) {
            return false;
        }
        int inputLnIdx = producerOf(qPre.inputTensor);
        if (inputLnIdx < 0) {
            return false;
        }
        auto inputLn = mOps[inputLnIdx].get();
        if (!isLayerNorm(inputLn) || inputLn->inputIndexes.size() != 1 || inputLn->outputIndexes.size() != 1 ||
            inputLn->outputIndexes[0] != qPre.inputTensor) {
            return false;
        }
        int attentionOutput = attention->outputIndexes[0];
        PreConvertMatch gatePre;
        int oConvIdx = findSingleConvolutionConsumer(attentionOutput);
        if (oConvIdx < 0) {
            int mulIdx = singleConsumer(attentionOutput);
            auto mul = mulIdx < 0 ? nullptr : mOps[mulIdx].get();
            if (!isBinaryOp(mul, BinaryOpOperation_MUL) || mul->defaultDimentionFormat != MNN_DATA_FORMAT_NC4HW4 ||
                mul->inputIndexes.size() != 2 || mul->outputIndexes.size() != 1) {
                return false;
            }
            int gateTensor = mul->inputIndexes[0] == attentionOutput ? mul->inputIndexes[1] : mul->inputIndexes[0];
            int gateUnaryIdx = producerOf(gateTensor);
            auto gateUnary = gateUnaryIdx < 0 ? nullptr : mOps[gateUnaryIdx].get();
            if (gateUnary == nullptr || gateUnary->type != OpType_UnaryOp || gateUnary->inputIndexes.size() != 1 ||
                gateUnary->outputIndexes.size() != 1 || gateUnary->defaultDimentionFormat != MNN_DATA_FORMAT_NC4HW4) {
                return false;
            }
            int gateConvIdx = producerOf(gateUnary->inputIndexes[0]);
            gatePre = matchPreConvertFromConv(gateConvIdx);
            if (!gatePre.valid || gatePre.inputTensor != qPre.inputTensor || gatePre.hiddenSize != qPre.hiddenSize) {
                return false;
            }
            oConvIdx = findSingleConvolutionConsumer(mul->outputIndexes[0]);
        }
        if (oConvIdx < 0) {
            return false;
        }
        auto attentionPost = matchPostConvertFromConv(oConvIdx, qPre.hiddenSize);
        if (!attentionPost.valid) {
            return false;
        }
        int blockHidden = inputLn->inputIndexes[0];
        int attentionAddIdx = findAddConsumerWithInputs(attentionPost.reshapeOut, blockHidden);
        int attentionAddOut = -1;
        int postLnIdx = -1;
        bool attentionAddFused = false;
        if (attentionAddIdx >= 0) {
            attentionAddOut = mOps[attentionAddIdx]->outputIndexes[0];
            postLnIdx = findLayerNormConsumer(attentionAddOut);
            if (postLnIdx < 0) {
                return false;
            }
        } else {
            postLnIdx = findLayerNormConsumerWithInputs(attentionPost.reshapeOut, blockHidden);
            if (postLnIdx < 0) {
                return false;
            }
            attentionAddFused = true;
            attentionAddOut = mOps[postLnIdx]->outputIndexes[0];
        }

        plan->hiddenSize = qPre.hiddenSize;
        plan->attentionIdx = attentionIdx;
        plan->ropeIdx = ropeIdx;
        plan->blockHidden = blockHidden;
        plan->inputLnIdx = inputLnIdx;
        plan->inputLnOut = qPre.inputTensor;
        plan->attentionPre.push_back(qPre);
        plan->attentionPre.push_back(kPre);
        plan->attentionPre.push_back(vPre);
        if (gatePre.valid && gatePre.convertIdx != qPre.convertIdx && gatePre.convertIdx != kPre.convertIdx &&
            gatePre.convertIdx != vPre.convertIdx) {
            plan->attentionPre.push_back(gatePre);
        }
        plan->attentionPost = attentionPost;
        plan->attentionAddIdx = attentionAddIdx;
        plan->attentionAddOut = attentionAddOut;
        plan->attentionAddFused = attentionAddFused;
        plan->postLnIdx = postLnIdx;
        plan->postLnOut = attentionAddFused ? mOps[postLnIdx]->outputIndexes[1] : mOps[postLnIdx]->outputIndexes[0];
        int blockHiddenProducer = producerOf(blockHidden);
        if (blockHiddenProducer >= 0 && isReshape(mOps[blockHiddenProducer].get()) &&
            !mOps[blockHiddenProducer]->inputIndexes.empty()) {
            plan->blockReshapeIdx = blockHiddenProducer;
            plan->blockInputSource = mOps[blockHiddenProducer]->inputIndexes[0];
        } else {
            plan->blockInputSource = blockHidden;
        }
        return matchMlpFromPostLayerNorm(plan);
    }

    bool matchHiddenBlockFromLinearAttention(int attentionIdx, HiddenBlockPlan* plan) const {
        if (attentionIdx < 0 || attentionIdx >= (int)mOps.size() || plan == nullptr) {
            return false;
        }
        auto attention = mOps[attentionIdx].get();
        if (attention == nullptr || attention->type != OpType_LinearAttention ||
            attention->main.type != OpParameter_LinearAttentionParam ||
            attention->main.AsLinearAttentionParam() == nullptr || attention->inputIndexes.size() != 4 ||
            attention->outputIndexes.size() != 1) {
            return false;
        }
        auto param = attention->main.AsLinearAttentionParam();
        bool shortConv = param->attn_type == "short_conv";
        bool gatedDelta = param->attn_type == "gated_delta_rule";
        if (!shortConv && !gatedDelta) {
            return false;
        }
        int keyDim = param->num_k_heads * param->head_k_dim;
        int valueDim = param->num_v_heads * param->head_v_dim;
        int expectedQKV = shortConv ? 3 * param->head_v_dim : 2 * keyDim + valueDim;
        if (param->num_v_heads <= 0 || param->head_v_dim <= 0 || expectedQKV <= 0) {
            return false;
        }

        int permuteIdx = producerOf(attention->inputIndexes[0]);
        if (permuteIdx < 0) {
            return false;
        }
        auto permute = mOps[permuteIdx].get();
        if (permute == nullptr || permute->type != OpType_Permute || permute->main.type != OpParameter_Permute ||
            permute->main.AsPermute() == nullptr || permute->inputIndexes.size() != 1 ||
            permute->outputIndexes.size() != 1 ||
            !dimsEqual(permute->main.AsPermute()->dims, std::vector<int>{0, 2, 1}) ||
            !sameConsumers(mConsumers, permute->outputIndexes[0], attentionIdx)) {
            return false;
        }
        auto qkvPost = matchLinearPost(permute->inputIndexes[0]);
        if (!qkvPost.valid || convolutionOutputCount(mOps[qkvPost.convIdx].get()) != expectedQKV ||
            !sameConsumers(mConsumers, qkvPost.reshapeOut, permuteIdx)) {
            return false;
        }
        auto qkvPre = matchPreConvertFromConv(qkvPost.convIdx);
        if (!qkvPre.valid) {
            return false;
        }

        LinearAttentionC4Plan linearPlan;
        linearPlan.shortConv = shortConv;
        linearPlan.attentionIdx = attentionIdx;
        linearPlan.replacements.emplace_back(attention->inputIndexes[0], qkvPost.convOut);
        linearPlan.removeIndexes.insert(permuteIdx);
        linearPlan.removeIndexes.insert(qkvPost.convertIdx);
        linearPlan.removeIndexes.insert(qkvPost.reshapeIdx);

        std::vector<PreConvertMatch> projectionPre = {qkvPre};
        auto appendExpression = [&](const ProjectionExpressionMatch& expression) -> bool {
            if (!expression.valid) {
                return false;
            }
            auto pre = matchPreConvertFromConv(expression.projection.convIdx);
            if (!pre.valid || pre.inputTensor != qkvPre.inputTensor || pre.hiddenSize != qkvPre.hiddenSize) {
                return false;
            }
            projectionPre.push_back(pre);
            linearPlan.replacements.emplace_back(expression.projection.reshapeOut, expression.projection.convOut);
            linearPlan.removeIndexes.insert(expression.projection.convertIdx);
            linearPlan.removeIndexes.insert(expression.projection.reshapeIdx);
            linearPlan.c4Ops.insert(linearPlan.c4Ops.end(), expression.c4Ops.begin(), expression.c4Ops.end());
            linearPlan.c4Constants.insert(expression.c4Constants.begin(), expression.c4Constants.end());
            return true;
        };

        int outputConvIdx = -1;
        PreConvertMatch outputPre;
        if (shortConv) {
            int outputViewIdx = singleConsumer(attention->outputIndexes[0]);
            if (outputViewIdx < 0 || !isReshape(mOps[outputViewIdx].get()) ||
                mOps[outputViewIdx]->outputIndexes.size() != 1 ||
                !findOutputProjection(mOps[outputViewIdx]->outputIndexes[0], valueDim, &outputPre, &outputConvIdx)) {
                return false;
            }
            linearPlan.outputInputC4 = attention->outputIndexes[0];
            linearPlan.removeIndexes.insert(outputViewIdx);
        } else {
            auto gateExpression =
                matchProjectionExpression(attention->inputIndexes[1], attentionIdx, param->num_v_heads);
            auto betaExpression =
                matchProjectionExpression(attention->inputIndexes[2], attentionIdx, param->num_v_heads);
            if (!appendExpression(gateExpression) || !appendExpression(betaExpression)) {
                return false;
            }

            int headReshapeIdx = singleConsumer(attention->outputIndexes[0]);
            if (headReshapeIdx < 0 || !isReshape(mOps[headReshapeIdx].get()) ||
                mOps[headReshapeIdx]->outputIndexes.size() != 1) {
                return false;
            }
            linearPlan.c4Ops.push_back(headReshapeIdx);
            int currentTensor = mOps[headReshapeIdx]->outputIndexes[0];
            int currentUser = singleConsumer(currentTensor);
            while (currentUser >= 0 && mOps[currentUser]->type == OpType_Cast &&
                   mOps[currentUser]->inputIndexes.size() == 1 && mOps[currentUser]->outputIndexes.size() == 1) {
                linearPlan.c4Ops.push_back(currentUser);
                currentTensor = mOps[currentUser]->outputIndexes[0];
                currentUser = singleConsumer(currentTensor);
            }
            if (currentUser < 0 || !isLayerNorm(mOps[currentUser].get()) ||
                mOps[currentUser]->inputIndexes.size() != 1 || mOps[currentUser]->outputIndexes.size() != 1) {
                return false;
            }
            int attentionNormIdx = currentUser;
            linearPlan.c4Ops.push_back(attentionNormIdx);
            int normOutput = mOps[attentionNormIdx]->outputIndexes[0];
            int normBinaryIdx = singleConsumer(normOutput);
            if (normBinaryIdx < 0 || !isBinaryOp(mOps[normBinaryIdx].get(), BinaryOpOperation_MUL) ||
                mOps[normBinaryIdx]->inputIndexes.size() != 2 || mOps[normBinaryIdx]->outputIndexes.size() != 1) {
                return false;
            }
            int zTensor = mOps[normBinaryIdx]->inputIndexes[0] == normOutput ? mOps[normBinaryIdx]->inputIndexes[1]
                                                                             : mOps[normBinaryIdx]->inputIndexes[0];
            auto zExpression = matchProjectionExpression(zTensor, normBinaryIdx, valueDim);
            if (!appendExpression(zExpression)) {
                return false;
            }
            linearPlan.c4Ops.push_back(normBinaryIdx);

            int finalReshapeIdx = singleConsumer(mOps[normBinaryIdx]->outputIndexes[0]);
            if (finalReshapeIdx < 0 || !isReshape(mOps[finalReshapeIdx].get()) ||
                mOps[finalReshapeIdx]->outputIndexes.size() != 1 ||
                !findOutputProjection(mOps[finalReshapeIdx]->outputIndexes[0], valueDim, &outputPre, &outputConvIdx)) {
                return false;
            }
            linearPlan.finalReshapeIdx = finalReshapeIdx;
            linearPlan.finalValueDim = valueDim;
            linearPlan.outputInputC4 = mOps[finalReshapeIdx]->outputIndexes[0];
            linearPlan.c4Ops.push_back(finalReshapeIdx);
        }

        if (outputConvIdx < 0 || !outputPre.valid) {
            return false;
        }
        linearPlan.outputConvIdx = outputConvIdx;
        linearPlan.outputPreConvertOut = outputPre.convertOut;
        linearPlan.removeIndexes.insert(outputPre.reshapeIdx);
        linearPlan.removeIndexes.insert(outputPre.convertIdx);

        auto attentionPost = matchPostConvertFromConv(outputConvIdx, qkvPre.hiddenSize);
        if (!attentionPost.valid) {
            return false;
        }
        int inputLnIdx = producerOf(qkvPre.inputTensor);
        if (inputLnIdx < 0) {
            return false;
        }
        auto inputLn = mOps[inputLnIdx].get();
        if (!isLayerNorm(inputLn) || inputLn->inputIndexes.size() != 1 || inputLn->outputIndexes.size() != 1 ||
            inputLn->outputIndexes[0] != qkvPre.inputTensor) {
            return false;
        }

        int blockHidden = inputLn->inputIndexes[0];
        int attentionAddIdx = findAddConsumerWithInputs(attentionPost.reshapeOut, blockHidden);
        int attentionAddOut = -1;
        int postLnIdx = -1;
        bool attentionAddFused = false;
        if (attentionAddIdx >= 0) {
            attentionAddOut = mOps[attentionAddIdx]->outputIndexes[0];
            postLnIdx = findLayerNormConsumer(attentionAddOut);
            if (postLnIdx < 0) {
                return false;
            }
        } else {
            postLnIdx = findLayerNormConsumerWithInputs(attentionPost.reshapeOut, blockHidden);
            if (postLnIdx < 0) {
                return false;
            }
            attentionAddFused = true;
            attentionAddOut = mOps[postLnIdx]->outputIndexes[0];
        }

        linearPlan.valid = true;
        plan->hiddenSize = qkvPre.hiddenSize;
        plan->blockHidden = blockHidden;
        plan->inputLnIdx = inputLnIdx;
        plan->inputLnOut = qkvPre.inputTensor;
        plan->attentionPre = projectionPre;
        plan->attentionPost = attentionPost;
        plan->attentionAddIdx = attentionAddIdx;
        plan->attentionAddOut = attentionAddOut;
        plan->attentionAddFused = attentionAddFused;
        plan->postLnIdx = postLnIdx;
        plan->postLnOut = attentionAddFused ? mOps[postLnIdx]->outputIndexes[1] : mOps[postLnIdx]->outputIndexes[0];
        plan->linearAttention = linearPlan;
        int blockHiddenProducer = producerOf(blockHidden);
        if (blockHiddenProducer >= 0 && isReshape(mOps[blockHiddenProducer].get()) &&
            !mOps[blockHiddenProducer]->inputIndexes.empty()) {
            plan->blockReshapeIdx = blockHiddenProducer;
            plan->blockInputSource = mOps[blockHiddenProducer]->inputIndexes[0];
        } else {
            plan->blockInputSource = blockHidden;
        }
        return matchMlpFromPostLayerNorm(plan);
    }

    bool canLinkHiddenBlocks(const HiddenBlockPlan& current, const HiddenBlockPlan& next) const {
        if (current.mlpAddOut != next.blockInputSource || current.hiddenSize != next.hiddenSize) {
            return false;
        }
        if (mGraphOutputs.find(current.mlpAddOut) != mGraphOutputs.end()) {
            return false;
        }
        std::set<int> allowedConsumers;
        if (next.blockReshapeIdx >= 0) {
            allowedConsumers.insert(next.blockReshapeIdx);
        } else {
            allowedConsumers.insert(next.inputLnIdx);
            allowedConsumers.insert(next.attentionAddFused ? next.postLnIdx : next.attentionAddIdx);
        }
        auto consumers = mConsumers.find(current.mlpAddOut);
        if (consumers == mConsumers.end() || consumers->second.empty()) {
            return false;
        }
        for (int consumer : consumers->second) {
            if (allowedConsumers.find(consumer) == allowedConsumers.end()) {
                return false;
            }
        }
        return true;
    }

    bool buildHiddenBlockRegions(std::vector<std::vector<HiddenBlockPlan>>* regions) const {
        if (regions == nullptr) {
            return false;
        }
        std::vector<HiddenBlockPlan> candidates;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            HiddenBlockPlan plan;
            if (!matchHiddenBlockFromAttention(idx, &plan) && !matchHiddenBlockFromLinearAttention(idx, &plan)) {
                continue;
            }
            candidates.push_back(plan);
        }
        if (candidates.empty()) {
            return false;
        }
        int count = (int)candidates.size();
        std::vector<int> previous(count, -1);
        std::vector<int> next(count, -1);
        std::vector<bool> valid(count, true);
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                if (i == j || !canLinkHiddenBlocks(candidates[i], candidates[j])) {
                    continue;
                }
                if (next[i] >= 0 || previous[j] >= 0) {
                    valid[i] = false;
                    valid[j] = false;
                    if (next[i] >= 0) {
                        valid[next[i]] = false;
                    }
                    if (previous[j] >= 0) {
                        valid[previous[j]] = false;
                    }
                    continue;
                }
                next[i] = j;
                previous[j] = i;
            }
        }

        std::vector<bool> visited(count, false);
        for (int i = 0; i < count; ++i) {
            if (!valid[i] || visited[i]) {
                continue;
            }
            int prev = previous[i];
            if (prev >= 0 && valid[prev] && candidates[prev].hiddenSize == candidates[i].hiddenSize) {
                continue;
            }
            if (candidates[i].hiddenSize <= 0 || candidates[i].hiddenSize % 4 != 0) {
                continue;
            }
            std::vector<HiddenBlockPlan> region;
            int current = i;
            while (current >= 0 && valid[current] && !visited[current] &&
                   candidates[current].hiddenSize == candidates[i].hiddenSize) {
                visited[current] = true;
                candidates[current].blockIdx = (int)region.size();
                region.push_back(candidates[current]);
                int nextIndex = next[current];
                if (nextIndex >= 0 && candidates[nextIndex].hiddenSize != candidates[i].hiddenSize) {
                    break;
                }
                current = nextIndex;
            }
            if (!region.empty()) {
                regions->emplace_back(std::move(region));
            }
        }
        return !regions->empty();
    }

    struct C4TailPlan {
        bool useC4Tail;
        int finalSliceIdx;
        int finalLayerNormIdx;
        int lmHeadPreReshapeIdx;
        int lmHeadPreConvertIdx;
        int lmHeadConvIdx;
        int constAxis0;
        C4TailPlan()
            : useC4Tail(false),
              finalSliceIdx(-1),
              finalLayerNormIdx(-1),
              lmHeadPreReshapeIdx(-1),
              lmHeadPreConvertIdx(-1),
              lmHeadConvIdx(-1),
              constAxis0(-1) {}
    };

    C4TailPlan matchC4Tail(int tensor) const {
        C4TailPlan plan;
        if (mGraphOutputs.find(tensor) != mGraphOutputs.end()) {
            return plan;
        }
        plan.finalSliceIdx = findSliceConsumer(tensor);
        plan.constAxis0 = findConstIntTensor(0);
        if (plan.finalSliceIdx < 0 || plan.constAxis0 < 0) {
            return plan;
        }
        auto finalSlice = mOps[plan.finalSliceIdx].get();
        if (finalSlice->outputIndexes.size() != 1 || finalSlice->inputIndexes.size() != 5) {
            return plan;
        }
        auto sliceParam = finalSlice->main.AsStridedSliceParam();
        auto axes = constInt32s(finalSlice->inputIndexes[3]);
        auto steps = constInt32s(finalSlice->inputIndexes[4]);
        if (sliceParam == nullptr || sliceParam->fromType != 1 || axes.size() != 1 || axes[0] != 1 ||
            steps.size() != 1 || steps[0] != 1) {
            return plan;
        }
        auto finalLayerNormUsers = mConsumers.find(finalSlice->outputIndexes[0]);
        if (finalLayerNormUsers == mConsumers.end() || finalLayerNormUsers->second.size() != 1) {
            return plan;
        }
        plan.finalLayerNormIdx = finalLayerNormUsers->second[0];
        auto finalLayerNorm = mOps[plan.finalLayerNormIdx].get();
        if (!isLayerNorm(finalLayerNorm) || finalLayerNorm->outputIndexes.size() != 1) {
            return plan;
        }
        auto preReshapeUsers = mConsumers.find(finalLayerNorm->outputIndexes[0]);
        if (preReshapeUsers == mConsumers.end() || preReshapeUsers->second.size() != 1) {
            return plan;
        }
        plan.lmHeadPreReshapeIdx = preReshapeUsers->second[0];
        auto preReshape = mOps[plan.lmHeadPreReshapeIdx].get();
        if (!isReshape(preReshape) || preReshape->outputIndexes.size() != 1) {
            return plan;
        }
        auto preConvertUsers = mConsumers.find(preReshape->outputIndexes[0]);
        if (preConvertUsers == mConsumers.end() || preConvertUsers->second.size() != 1) {
            return plan;
        }
        plan.lmHeadPreConvertIdx = preConvertUsers->second[0];
        auto preConvert = mOps[plan.lmHeadPreConvertIdx].get();
        if (!isConvert(preConvert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) ||
            preConvert->outputIndexes.size() != 1) {
            return plan;
        }
        auto lmHeadUsers = mConsumers.find(preConvert->outputIndexes[0]);
        if (lmHeadUsers == mConsumers.end() || lmHeadUsers->second.size() != 1 ||
            !isConvolution(mOps[lmHeadUsers->second[0]].get())) {
            return plan;
        }
        plan.lmHeadConvIdx = lmHeadUsers->second[0];
        plan.useC4Tail = true;
        return plan;
    }

    bool fuseHiddenStateC4Regions() {
        rebuildMaps();
        std::vector<std::vector<HiddenBlockPlan>> regions;
        if (!buildHiddenBlockRegions(&regions)) {
            return false;
        }

        std::map<int, std::vector<std::unique_ptr<OpT>>> insertBefore;
        std::map<int, std::vector<std::unique_ptr<OpT>>> insertAfter;
        std::set<int> removeIndexes;
        for (auto& plans : regions) {
            int hiddenSize = plans[0].hiddenSize;
            int firstHidden = plans[0].blockHidden;
            int firstLayerNormIdx = plans[0].inputLnIdx;
            auto tailPlan = matchC4Tail(plans.back().mlpAddOut);

            int initialReshape = buildUniqueTensor("FuseTransformerC4_region_pre_reshape");
            int initialC4 = buildUniqueTensor("FuseTransformerC4_region_pre_convert");
            insertBefore[firstLayerNormIdx].push_back(
                makeReshape(mTensors[initialReshape], firstHidden, initialReshape, {-1, hiddenSize, 1, 1}));
            insertBefore[firstLayerNormIdx].push_back(makeConvert(mTensors[initialC4], initialReshape, initialC4,
                                                                  MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4));

            int currentC4 = initialC4;
            for (auto& plan : plans) {
                auto inputLayerNorm = mOps[plan.inputLnIdx].get();
                inputLayerNorm->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                replaceInput(plan.inputLnIdx, plan.blockHidden, currentC4);
                if (plan.blockIdx > 0 && plan.blockReshapeIdx >= 0) {
                    removeIndexes.insert(plan.blockReshapeIdx);
                }
                for (auto& pre : plan.attentionPre) {
                    for (int convIdx : pre.convUsers) {
                        replaceInput(convIdx, pre.convertOut, plan.inputLnOut);
                    }
                    removeIndexes.insert(pre.reshapeIdx);
                    removeIndexes.insert(pre.convertIdx);
                }

                if (plan.linearAttention.valid) {
                    auto& linear = plan.linearAttention;
                    auto linearAttention = mOps[linear.attentionIdx].get();
                    linearAttention->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                    for (auto& replacement : linear.replacements) {
                        for (int opIdx = 0; opIdx < (int)mOps.size(); ++opIdx) {
                            replaceInput(opIdx, replacement.oldTensor, replacement.newTensor);
                        }
                    }
                    for (int opIdx : linear.c4Ops) {
                        if (opIdx >= 0 && opIdx < (int)mOps.size()) {
                            mOps[opIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                        }
                    }
                    for (auto& constant : linear.c4Constants) {
                        auto constOp = mOps[constant.first].get();
                        if (constOp == nullptr || constOp->main.type != OpParameter_Blob ||
                            constOp->main.AsBlob() == nullptr) {
                            continue;
                        }
                        auto blob = constOp->main.AsBlob();
                        blob->dims = {1, constant.second, 1, 1};
                        blob->dataFormat = MNN_DATA_FORMAT_NC4HW4;
                        constOp->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                    }
                    if (linear.finalReshapeIdx >= 0) {
                        auto finalReshape = mOps[linear.finalReshapeIdx].get();
                        finalReshape->inputIndexes.resize(1);
                        finalReshape->main.AsReshape()->dims = {-1, linear.finalValueDim, 1, 1};
                        finalReshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NCHW;
                        finalReshape->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                    }
                    replaceInput(linear.outputConvIdx, linear.outputPreConvertOut, linear.outputInputC4);
                    removeIndexes.insert(linear.removeIndexes.begin(), linear.removeIndexes.end());
                }

                if (plan.attentionAddFused) {
                    auto& layerNormInputs = mOps[plan.postLnIdx]->inputIndexes;
                    layerNormInputs = {currentC4, plan.attentionPost.convOut};
                } else {
                    auto attentionAdd = mOps[plan.attentionAddIdx].get();
                    attentionAdd->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                    replaceInput(plan.attentionAddIdx, plan.blockHidden, currentC4);
                    replaceInput(plan.attentionAddIdx, plan.attentionPost.reshapeOut, plan.attentionPost.convOut);
                    auto& addInputs = attentionAdd->inputIndexes;
                    if (addInputs.size() != 2 || addInputs[0] != currentC4) {
                        addInputs = {currentC4, plan.attentionPost.convOut};
                    }
                }
                mOps[plan.postLnIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                removeIndexes.insert(plan.attentionPost.convertIdx);
                removeIndexes.insert(plan.attentionPost.reshapeIdx);

                for (int convIdx : plan.gatePre.convUsers) {
                    replaceInput(convIdx, plan.gatePre.convertOut, plan.postLnOut);
                }
                removeIndexes.insert(plan.gatePre.reshapeIdx);
                removeIndexes.insert(plan.gatePre.convertIdx);
                for (int convIdx : plan.upPre.convUsers) {
                    replaceInput(convIdx, plan.upPre.convertOut, plan.postLnOut);
                }
                removeIndexes.insert(plan.upPre.reshapeIdx);
                removeIndexes.insert(plan.upPre.convertIdx);

                if (plan.mlpBinaryIdx >= 0) {
                    mOps[plan.mlpBinaryIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                }
                auto mlpAdd = mOps[plan.mlpAddIdx].get();
                mlpAdd->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                replaceInput(plan.mlpAddIdx, plan.downPost.reshapeOut, plan.downPost.convOut);
                auto& addInputs = mlpAdd->inputIndexes;
                if (addInputs.size() != 2 || addInputs[0] != plan.attentionAddOut) {
                    addInputs = {plan.attentionAddOut, plan.downPost.convOut};
                }
                removeIndexes.insert(plan.downPost.convertIdx);
                removeIndexes.insert(plan.downPost.reshapeIdx);
                currentC4 = plan.mlpAddOut;
            }

            if (tailPlan.useC4Tail) {
                auto finalLayerNorm = mOps[tailPlan.finalLayerNormIdx].get();
                auto lmHeadPreConvert = mOps[tailPlan.lmHeadPreConvertIdx].get();
                mOps[tailPlan.finalSliceIdx]->inputIndexes[0] = currentC4;
                mOps[tailPlan.finalSliceIdx]->inputIndexes[3] = tailPlan.constAxis0;
                mOps[tailPlan.finalSliceIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                finalLayerNorm->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                replaceInput(tailPlan.lmHeadConvIdx, lmHeadPreConvert->outputIndexes[0],
                             finalLayerNorm->outputIndexes[0]);
                removeIndexes.insert(tailPlan.lmHeadPreReshapeIdx);
                removeIndexes.insert(tailPlan.lmHeadPreConvertIdx);
                continue;
            }

            auto& lastPlan = plans.back();
            int plainOutput = lastPlan.mlpAddOut;
            int c4Output = buildUniqueTensor("FuseTransformerC4_region_output");
            auto& mlpOutputs = mOps[lastPlan.mlpAddIdx]->outputIndexes;
            for (auto& output : mlpOutputs) {
                if (output == plainOutput) {
                    output = c4Output;
                }
            }
            int finalNchw4 = buildUniqueTensor("FuseTransformerC4_region_post_convert");
            insertAfter[lastPlan.mlpAddIdx].push_back(
                makeConvert(mTensors[finalNchw4], c4Output, finalNchw4, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW));
            auto postReshape =
                makeReshape(mTensors[c4Output] + "_post_reshape", finalNchw4, plainOutput, {1, -1, hiddenSize});
            postReshape->defaultDimentionFormat = MNN_DATA_FORMAT_NHWC;
            insertAfter[lastPlan.mlpAddIdx].push_back(std::move(postReshape));
        }

        std::vector<std::unique_ptr<OpT>> newOps;
        newOps.reserve(mOps.size() + regions.size() * 4);
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto beforeIter = insertBefore.find(idx);
            if (beforeIter != insertBefore.end()) {
                for (auto& insertOp : beforeIter->second) {
                    newOps.emplace_back(std::move(insertOp));
                }
            }
            if (removeIndexes.find(idx) == removeIndexes.end()) {
                newOps.emplace_back(std::move(mOps[idx]));
            }
            auto afterIter = insertAfter.find(idx);
            if (afterIter != insertAfter.end()) {
                for (auto& insertOp : afterIter->second) {
                    newOps.emplace_back(std::move(insertOp));
                }
            }
        }
        mOps.swap(newOps);
        return true;
    }

    bool validateOptimizedGraph() {
        rebuildMaps();
        std::vector<int> producerCount(mTensors.size(), 0);
        for (auto& opPtr : mOps) {
            auto op = opPtr.get();
            if (op == nullptr) {
                return false;
            }
            for (int input : op->inputIndexes) {
                if (input >= (int)mTensors.size()) {
                    return false;
                }
            }
            for (int output : op->outputIndexes) {
                if (output < 0 || output >= (int)mTensors.size() || ++producerCount[output] != 1) {
                    return false;
                }
            }

            auto param = ropeParam(op);
            if (param != nullptr) {
                if (op->inputIndexes.size() < 2 || param->num_head <= 0 || param->kv_num_head <= 0 ||
                    param->head_dim <= 0 || param->rope_cut_head_dim <= 0 ||
                    param->rope_cut_head_dim > param->head_dim || param->rope_cut_head_dim % 2 != 0) {
                    return false;
                }
                if ((param->q_norm != nullptr && (int)param->q_norm->gamma.size() != param->head_dim) ||
                    (param->k_norm != nullptr && (int)param->k_norm->gamma.size() != param->head_dim)) {
                    return false;
                }
                if (!isPackedRoPEInput(op->inputIndexes[0], param->num_head * param->head_dim) ||
                    !isPackedRoPEInput(op->inputIndexes[1], param->kv_num_head * param->head_dim)) {
                    return false;
                }
            }

            if (op->type == OpType_Attention && op->main.type == OpParameter_AttentionParam &&
                op->main.AsAttentionParam() != nullptr) {
                auto param = op->main.AsAttentionParam();
                if (op->outputIndexes.size() != 1) {
                    return false;
                }
                if (!param->output_c4) {
                    continue;
                }
                auto consumers = mConsumers.find(op->outputIndexes[0]);
                if (consumers == mConsumers.end() || consumers->second.empty()) {
                    return false;
                }
                for (int consumer : consumers->second) {
                    auto user = mOps[consumer].get();
                    bool gatedOutput = isBinaryOp(user, BinaryOpOperation_MUL) &&
                                       user->defaultDimentionFormat == MNN_DATA_FORMAT_NC4HW4;
                    if (!isConvolution(user) && !gatedOutput) {
                        return false;
                    }
                }
            }
        }
        for (int output : mGraphOutputs) {
            if (output < 0 || output >= (int)producerCount.size() || producerCount[output] != 1) {
                return false;
            }
        }
        return true;
    }
};

} // namespace

class FuseTransformerC4 : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        if (net == nullptr) {
            return true;
        }
        std::set<int> mainOutputs;
        bool validOutputNames = true;
        for (auto& outputName : net->outputName) {
            int outputIndex = -1;
            for (int i = 0; i < (int)net->tensorName.size(); ++i) {
                if (net->tensorName[i] == outputName) {
                    if (outputIndex >= 0) {
                        validOutputNames = false;
                        break;
                    }
                    outputIndex = i;
                }
            }
            if (outputIndex < 0) {
                validOutputNames = false;
            } else {
                mainOutputs.insert(outputIndex);
            }
        }
        if (validOutputNames) {
            TransformerC4Graph mainGraph(net->oplists, net->tensorName, mainOutputs);
            mainGraph.run();
        }
        for (auto& subgraph : net->subgraphs) {
            if (subgraph == nullptr) {
                continue;
            }
            std::set<int> subgraphOutputs(subgraph->outputs.begin(), subgraph->outputs.end());
            TransformerC4Graph subGraph(subgraph->nodes, subgraph->tensors, subgraphOutputs);
            subGraph.run();
        }
        return true;
    }
};

static PostConverterRegister<FuseTransformerC4> __l("FuseTransformerC4");
