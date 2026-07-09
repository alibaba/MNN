//
//  FuseTransformerC4.cpp
//  MNNConverter
//
//  Created by MNN on 2026/06/23.
//

#include "../PostTreatUtils.hpp"
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

class TransformerC4Graph {
public:
    TransformerC4Graph(std::vector<std::unique_ptr<OpT>>& ops, std::vector<std::string>& tensors)
        : mOps(ops), mTensors(tensors) {}

    bool run() {
        if (!canFuseAsKnownTransformerGraph()) {
            return false;
        }
        return runFusePipeline();
    }

private:
    bool runFusePipeline() {
        bool changed = false;
        changed |= fuseAttentionOutputC4();
        changed |= fuseMulSilu();
        changed |= fuseMlpOutputC4();
        changed |= fuseRoPEInputC4();
        changed |= fuseAttentionValueC4();
        changed |= fuseBinaryLayerNormC4();
        changed |= fuseHiddenStateC4();
        changed |= fuseBinaryLayerNormC4();
        return changed;
    }

    std::vector<std::unique_ptr<OpT>>& mOps;
    std::vector<std::string>& mTensors;
    std::unordered_map<int, int> mProducer;
    std::unordered_map<int, std::vector<int>> mConsumers;

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
            if (reshapeUsers.size() != 1) {
                continue;
            }
            int reshapeIdx = reshapeUsers[0];
            auto reshape = mOps[reshapeIdx].get();
            if (!isReshape(reshape) || reshape->outputIndexes.size() != 1) {
                continue;
            }
            auto dims = reshape->main.AsReshape()->dims;
            if (dims.size() != 4 || dims[2] != 1 || dims[3] != 1) {
                continue;
            }
            auto convertUsers = mConsumers[reshape->outputIndexes[0]];
            if (convertUsers.size() != 1) {
                continue;
            }
            int convertIdx = convertUsers[0];
            auto convert = mOps[convertIdx].get();
            if (!isConvert(convert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) ||
                convert->outputIndexes.size() != 1) {
                continue;
            }
            auto convUsers = mConsumers[convert->outputIndexes[0]];
            if (convUsers.size() != 1 || mOps[convUsers[0]]->type != OpType_Convolution) {
                continue;
            }
            op->main.AsAttentionParam()->output_c4 = true;
            op->outputIndexes = convert->outputIndexes;
            removeIndexes.insert(reshapeIdx);
            removeIndexes.insert(convertIdx);
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
                op->outputIndexes.size() != 1) {
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
        if (outputCount <= 0 || postDims.size() != 3 || postDims[0] != 1 || postDims[2] != outputCount ||
            outputCount % 4 != 0) {
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
            auto op = mOps[idx].get();
            auto param = ropeParam(op);
            if (param == nullptr || op->inputIndexes.size() < 2 || param->num_head > 0 || param->kv_num_head > 0 ||
                param->head_dim > 0) {
                continue;
            }
            auto qMatch = matchRoPEInput(op->inputIndexes[0], idx);
            auto kMatch = matchRoPEInput(op->inputIndexes[1], idx);
            if (!qMatch.valid || !kMatch.valid) {
                continue;
            }
            int headDim = param->rope_cut_head_dim;
            if (headDim <= 0 || qMatch.channel % headDim != 0 || kMatch.channel % headDim != 0 || headDim % 4 != 0) {
                continue;
            }
            op->inputIndexes[0] = qMatch.input;
            op->inputIndexes[1] = kMatch.input;
            param->num_head = qMatch.channel / headDim;
            param->kv_num_head = kMatch.channel / headDim;
            param->head_dim = headDim;
            removeIndexes.insert(qMatch.removeIndexes.begin(), qMatch.removeIndexes.end());
            removeIndexes.insert(kMatch.removeIndexes.begin(), kMatch.removeIndexes.end());
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
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
        if (outputCount <= 0 || postDims.size() != 3 || postDims[0] != 1 || postDims[2] != outputCount ||
            outputCount % 4 != 0) {
            return match;
        }
        match.valid = true;
        match.input = conv->outputIndexes[0];
        match.removeIndexes = removeChain;
        match.removeIndexes.push_back(convertIdx);
        return match;
    }

    bool fuseAttentionValueC4() {
        rebuildMaps();
        std::set<int> removeIndexes;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto op = mOps[idx].get();
            if (op->type != OpType_Attention || op->inputIndexes.size() < 3) {
                continue;
            }
            auto match = matchAttentionValue(op->inputIndexes[2], idx);
            if (!match.valid) {
                continue;
            }
            op->inputIndexes[2] = match.input;
            removeIndexes.insert(match.removeIndexes.begin(), match.removeIndexes.end());
        }
        bool changed = !removeIndexes.empty();
        removeOps(removeIndexes);
        return changed;
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
        if (consumers == mConsumers.end()) {
            return -1;
        }
        int matched = -1;
        for (int userIdx : consumers->second) {
            auto user = mOps[userIdx].get();
            if (user == nullptr || (user->type != OpType_Slice && user->type != OpType_StridedSlice) ||
                user->inputIndexes.empty() || user->inputIndexes[0] != tensor || user->outputIndexes.size() != 1) {
                continue;
            }
            if (matched >= 0) {
                return -1;
            }
            matched = userIdx;
        }
        return matched;
    }
    struct HiddenBlockPlan {
        int blockIdx;
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
        PostConvertMatch downPost;
        int mlpAddIdx;
        int mlpAddOut;
        HiddenBlockPlan()
            : blockIdx(0),
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
            if (!isBinaryOp(op, BinaryOpOperation_MUL_SILU) || op->inputIndexes.size() != 2 ||
                op->outputIndexes.size() != 1) {
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
        int oConvIdx = findSingleConvolutionConsumer(attention->outputIndexes[0]);
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
        plan->blockHidden = blockHidden;
        plan->inputLnIdx = inputLnIdx;
        plan->inputLnOut = qPre.inputTensor;
        plan->attentionPre.push_back(qPre);
        plan->attentionPre.push_back(kPre);
        plan->attentionPre.push_back(vPre);
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

    bool buildHiddenBlockChain(std::vector<HiddenBlockPlan>* plans) const {
        if (plans == nullptr) {
            return false;
        }
        std::vector<HiddenBlockPlan> candidates;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            HiddenBlockPlan plan;
            if (!matchHiddenBlockFromAttention(idx, &plan)) {
                continue;
            }
            candidates.push_back(plan);
        }
        if (candidates.empty()) {
            return false;
        }
        std::set<int> mlpOutputs;
        for (auto& plan : candidates) {
            mlpOutputs.insert(plan.mlpAddOut);
        }
        int start = -1;
        for (int i = 0; i < (int)candidates.size(); ++i) {
            if (mlpOutputs.find(candidates[i].blockInputSource) != mlpOutputs.end()) {
                continue;
            }
            if (start >= 0) {
                return false;
            }
            start = i;
        }
        if (start < 0) {
            return false;
        }
        std::set<int> visited;
        int current = start;
        while (current >= 0) {
            if (visited.find(current) != visited.end()) {
                return false;
            }
            visited.insert(current);
            candidates[current].blockIdx = (int)plans->size();
            plans->push_back(candidates[current]);

            int next = -1;
            int currentOut = candidates[current].mlpAddOut;
            for (int i = 0; i < (int)candidates.size(); ++i) {
                if (visited.find(i) != visited.end() || candidates[i].blockInputSource != currentOut) {
                    continue;
                }
                if (next >= 0) {
                    return false;
                }
                next = i;
            }
            current = next;
        }
        return plans->size() == candidates.size();
    }

    bool validateHiddenPlans(std::vector<HiddenBlockPlan>* plans) {
        rebuildMaps();
        if (!buildHiddenBlockChain(plans) || plans->empty()) {
            return false;
        }
        int hiddenSize = (*plans)[0].hiddenSize;
        if (hiddenSize <= 0 || hiddenSize % 4 != 0) {
            return false;
        }
        for (auto& plan : *plans) {
            if (plan.hiddenSize != hiddenSize) {
                return false;
            }
        }
        return findSliceConsumer(plans->back().mlpAddOut) >= 0;
    }

    bool isCurrentGraphKnownTransformerC4Candidate() {
        std::vector<HiddenBlockPlan> plans;
        if (!validateHiddenPlans(&plans)) {
            return false;
        }
        int attentionCount = 0;
        int sequenceMixerCount = 0;
        for (auto& opPtr : mOps) {
            auto op = opPtr.get();
            if (op == nullptr) {
                continue;
            }
            if (op->type == OpType_Attention) {
                ++attentionCount;
                ++sequenceMixerCount;
            } else if (op->type == OpType_LinearAttention) {
                ++sequenceMixerCount;
            }
        }
        return attentionCount > 0 && attentionCount == (int)plans.size() && sequenceMixerCount == attentionCount;
    }

    bool canFuseAsKnownTransformerGraph() const {
        auto dryRunOps = cloneOps(mOps);
        auto dryRunTensors = mTensors;
        TransformerC4Graph dryRunGraph(dryRunOps, dryRunTensors);
        dryRunGraph.fuseAttentionOutputC4();
        dryRunGraph.fuseMulSilu();
        dryRunGraph.fuseMlpOutputC4();
        dryRunGraph.fuseRoPEInputC4();
        dryRunGraph.fuseAttentionValueC4();
        dryRunGraph.fuseBinaryLayerNormC4();
        return dryRunGraph.isCurrentGraphKnownTransformerC4Candidate();
    }

    bool fuseHiddenStateC4() {
        std::vector<HiddenBlockPlan> plans;
        if (!validateHiddenPlans(&plans)) {
            return false;
        }
        int hiddenSize = plans[0].hiddenSize;
        int firstHidden = plans[0].blockHidden;
        int firstLayerNormIdx = plans[0].inputLnIdx;
        int currentC4 = plans.back().mlpAddOut;
        int finalSliceIdx = findSliceConsumer(currentC4);
        if (finalSliceIdx < 0) {
            return false;
        }
        int finalLayerNormIdx = -1;
        int lmHeadPreReshapeIdx = -1;
        int lmHeadPreConvertIdx = -1;
        int lmHeadConvIdx = -1;
        int constAxis0 = findConstIntTensor(0);
        bool useC4Tail = false;
        auto finalSlice = mOps[finalSliceIdx].get();
        if (constAxis0 >= 0 && finalSlice->outputIndexes.size() == 1 && finalSlice->inputIndexes.size() >= 4) {
            auto finalLayerNormUsers = mConsumers[finalSlice->outputIndexes[0]];
            if (finalLayerNormUsers.size() == 1) {
                finalLayerNormIdx = finalLayerNormUsers[0];
                auto finalLayerNorm = mOps[finalLayerNormIdx].get();
                if (isLayerNorm(finalLayerNorm) && finalLayerNorm->outputIndexes.size() == 1) {
                    auto preReshapeUsers = mConsumers[finalLayerNorm->outputIndexes[0]];
                    if (preReshapeUsers.size() == 1) {
                        lmHeadPreReshapeIdx = preReshapeUsers[0];
                        auto preReshape = mOps[lmHeadPreReshapeIdx].get();
                        if (isReshape(preReshape) && preReshape->outputIndexes.size() == 1) {
                            auto preConvertUsers = mConsumers[preReshape->outputIndexes[0]];
                            if (preConvertUsers.size() == 1) {
                                lmHeadPreConvertIdx = preConvertUsers[0];
                                auto preConvert = mOps[lmHeadPreConvertIdx].get();
                                if (isConvert(preConvert, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4) &&
                                    preConvert->outputIndexes.size() == 1) {
                                    auto lmHeadUsers = mConsumers[preConvert->outputIndexes[0]];
                                    if (lmHeadUsers.size() == 1 && isConvolution(mOps[lmHeadUsers[0]].get())) {
                                        lmHeadConvIdx = lmHeadUsers[0];
                                        useC4Tail = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        int initialReshape = buildUniqueTensor("FuseTransformerC4_pre_reshape");
        int initialC4 = buildUniqueTensor("FuseTransformerC4_pre_convert");
        int finalNchw4 = -1;
        int finalNchw = -1;

        std::map<int, std::vector<std::unique_ptr<OpT>>> insertBefore;
        insertBefore[firstLayerNormIdx].push_back(
            makeReshape(mTensors[initialReshape], firstHidden, initialReshape, {-1, hiddenSize, 1, 1}));
        insertBefore[firstLayerNormIdx].push_back(
            makeConvert(mTensors[initialC4], initialReshape, initialC4, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4));
        if (!useC4Tail) {
            finalNchw4 = buildUniqueTensor("FuseTransformerC4_post_convert");
            finalNchw = buildUniqueTensor("FuseTransformerC4_post_reshape");
            insertBefore[finalSliceIdx].push_back(
                makeConvert(mTensors[finalNchw4], currentC4, finalNchw4, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW));
            insertBefore[finalSliceIdx].push_back(
                makeReshape(mTensors[finalNchw], finalNchw4, finalNchw, {1, -1, hiddenSize}));
        }

        std::set<int> removeIndexes;
        currentC4 = initialC4;
        for (auto& plan : plans) {
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

            if (plan.attentionAddFused) {
                auto& layerNormInputs = mOps[plan.postLnIdx]->inputIndexes;
                layerNormInputs = {currentC4, plan.attentionPost.convOut};
                mOps[plan.postLnIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            } else {
                replaceInput(plan.attentionAddIdx, plan.blockHidden, currentC4);
                replaceInput(plan.attentionAddIdx, plan.attentionPost.reshapeOut, plan.attentionPost.convOut);
                auto& addInputs = mOps[plan.attentionAddIdx]->inputIndexes;
                if (addInputs.size() != 2 || addInputs[0] != currentC4) {
                    addInputs = {currentC4, plan.attentionPost.convOut};
                }
            }
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

            replaceInput(plan.mlpAddIdx, plan.downPost.reshapeOut, plan.downPost.convOut);
            auto& add1Inputs = mOps[plan.mlpAddIdx]->inputIndexes;
            if (add1Inputs.size() != 2 || add1Inputs[0] != plan.attentionAddOut) {
                add1Inputs = {plan.attentionAddOut, plan.downPost.convOut};
            }
            removeIndexes.insert(plan.downPost.convertIdx);
            removeIndexes.insert(plan.downPost.reshapeIdx);
            currentC4 = plan.mlpAddOut;
        }
        if (useC4Tail) {
            auto finalLayerNorm = mOps[finalLayerNormIdx].get();
            auto lmHeadPreConvert = mOps[lmHeadPreConvertIdx].get();
            mOps[finalSliceIdx]->inputIndexes[0] = currentC4;
            mOps[finalSliceIdx]->inputIndexes[3] = constAxis0;
            mOps[finalSliceIdx]->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            finalLayerNorm->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            replaceInput(lmHeadConvIdx, lmHeadPreConvert->outputIndexes[0], finalLayerNorm->outputIndexes[0]);
            removeIndexes.insert(lmHeadPreReshapeIdx);
            removeIndexes.insert(lmHeadPreConvertIdx);
        } else {
            mOps[finalSliceIdx]->inputIndexes[0] = finalNchw;
        }

        std::vector<std::unique_ptr<OpT>> newOps;
        for (int idx = 0; idx < (int)mOps.size(); ++idx) {
            auto insertIter = insertBefore.find(idx);
            if (insertIter != insertBefore.end()) {
                for (auto& insertOp : insertIter->second) {
                    newOps.emplace_back(std::move(insertOp));
                }
            }
            if (removeIndexes.find(idx) == removeIndexes.end()) {
                newOps.emplace_back(std::move(mOps[idx]));
            }
        }
        mOps.swap(newOps);
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
        TransformerC4Graph mainGraph(net->oplists, net->tensorName);
        mainGraph.run();
        for (auto& subgraph : net->subgraphs) {
            if (subgraph == nullptr) {
                continue;
            }
            TransformerC4Graph subGraph(subgraph->nodes, subgraph->tensors);
            subGraph.run();
        }
        return true;
    }
};

static PostConverterRegister<FuseTransformerC4> __l("FuseTransformerC4");
