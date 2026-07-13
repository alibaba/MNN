#include <cstring>
#include <vector>

#include <MNN/MNNDefine.h>
#include "MNN_generated.h"

#include "../optimizer/Global.hpp"
#include "SafetensorConverter.hpp"
#include "SafetensorModelRegistry.hpp"
#include "SafetensorUtils.hpp"
#include "WorkflowJson.hpp"
#include "HuggingFaceQwen3.hpp"

using namespace MNN::Express;
using namespace MNN::Express::SafeTensorUtils;

namespace MNN {
namespace SafeTensors {

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

    if (nullptr == weightOI->readMap<float>()) {
        weightOI = _Cast<float>(weightOI);
        weightOI.fix(VARP::CONSTANT);
    }

    std::vector<float> weightData(weightOI->getInfo()->size);
    ::memcpy(weightData.data(), weightOI->readMap<float>(), weightData.size() * sizeof(float));

    std::vector<float> biasData(outDim, 0.0f);
    if (nullptr != bias) {
        if (nullptr == bias->readMap<float>()) {
            bias = _Cast<float>(bias);
            bias.fix(VARP::CONSTANT);
        }
        ::memcpy(biasData.data(), bias->readMap<float>(), outDim * sizeof(float));
    }

    return _Conv(std::move(weightData), std::move(biasData), x4d, {inDim, outDim}, {1, 1});
}

class HuggingFaceQwen3 {
public:
    HuggingFaceQwen3(const Converter* converter) : mConverter(converter) {}

    struct BlockInfo {
        int hiddenSize = 0;
        int headDim = 0;
        int numberHead = 0;
        int ropeCutHeadDim = 0;

        VARP cosEven;
        VARP cosOdd;
        VARP sinEven;
        VARP sinOdd;

        VARP shapeQKV;
    };

    std::pair<VARP, VARP> makeBlock(VARP hiddenState, VARP add, const BlockInfo& info, VARP mask, int blockIndex) {
        auto blockPrefix = std::string("model.layers.") + std::to_string(blockIndex) + ".";
        auto attnPrefix  = blockPrefix + "self_attn.";
        auto mlpPrefix   = blockPrefix + "mlp.";

        auto setName = [](const VARP& v, const std::string& name) {
            if (nullptr != v.get()) {
                v->setName(name);
            }
        };

        auto load = [this](const std::string& name) {
            if (mConverter->hasTensor(name)) {
                return mConverter->loadTensor(name, false);
            }
            return MNN::Express::VARP(nullptr);
        };

        const int hiddenSize = info.hiddenSize;
        const float ln_eps = 1.0e-6f;
        const bool useC4Opt = true;

        // RMSNorm + QKV
        auto ln1Weight = load(blockPrefix + "input_layernorm.weight");
        
        VARP hiddenStateNorm;
        if (nullptr != add.get()) {
            auto res = _BinaryLayerNorm(hiddenState, add, {ln1Weight, nullptr, ln_eps, true, hiddenSize, true});
            hiddenStateNorm = res.second;
            hiddenState = res.first;
        } else {
            hiddenStateNorm = _TransformerLayerNorm(hiddenState, {ln1Weight, nullptr, ln_eps, true, hiddenSize, true});
        }
        setName(hiddenStateNorm, blockPrefix + "input_layernorm.out");
        auto hiddenStateNorm4d = hiddenStateNorm;

        auto qWeight = load(attnPrefix + "q_proj.weight");
        auto kWeight = load(attnPrefix + "k_proj.weight");
        auto vWeight = load(attnPrefix + "v_proj.weight");
        auto oWeight = load(attnPrefix + "o_proj.weight");

        VARP qBias;
        VARP kBias;
        VARP vBias;
        if (mConverter->hasTensor(attnPrefix + "q_proj.bias")) {
            qBias = load(attnPrefix + "q_proj.bias");
        }
        if (mConverter->hasTensor(attnPrefix + "k_proj.bias")) {
            kBias = load(attnPrefix + "k_proj.bias");
        }
        if (mConverter->hasTensor(attnPrefix + "v_proj.bias")) {
            vBias = load(attnPrefix + "v_proj.bias");
        }

        auto qWeightInfo = qWeight->getInfo();
        auto kWeightInfo = kWeight->getInfo();
        auto vWeightInfo = vWeight->getInfo();
        if (nullptr == qWeightInfo || nullptr == kWeightInfo || nullptr == vWeightInfo) {
            return {nullptr, nullptr};
        }

        const int queryHiddenSize = qWeightInfo->dim[0];
        int headDim = info.headDim;
        int numHeads = info.numberHead > 0 ? info.numberHead : (queryHiddenSize / headDim);
        if (headDim <= 0 || numHeads <= 0 || numHeads * headDim != queryHiddenSize) {
            return {nullptr, nullptr};
        }
        const int attnOutSize = headDim * numHeads;

        auto q = _linear2d(hiddenStateNorm4d, qWeight, qBias);
        setName(q, attnPrefix + "q_proj.out");
        auto k = _linear2d(hiddenStateNorm4d, kWeight, kBias);
        setName(k, attnPrefix + "k_proj.out");
        auto v = _linear2d(hiddenStateNorm4d, vWeight, vBias);
        setName(v, attnPrefix + "v_proj.out");

        auto shapeqkv = info.shapeQKV;

        q = _Reshape(q, shapeqkv);
        setName(q, attnPrefix + "q_proj.out_reshape");
        RopeInfo ropeParam;
        ropeParam.cutHeadDim = info.ropeCutHeadDim;
        
        if (mConverter->hasTensor(attnPrefix + "q_norm.weight")) {
            auto qNorm = load(attnPrefix + "q_norm.weight");
            ropeParam.qNorm = {qNorm, nullptr, ln_eps, true};
        }

        k = _Reshape(k, shapeqkv);
        setName(k, attnPrefix + "k_proj.out_reshape");
        if (mConverter->hasTensor(attnPrefix + "k_norm.weight")) {
            auto kNorm = load(attnPrefix + "k_norm.weight");
            ropeParam.kNorm = {kNorm, nullptr, ln_eps, true};
        }

        v = _Reshape(v, shapeqkv);
        setName(v, attnPrefix + "v_proj.out_reshape");

        // RoPE
        {
            auto ropeOutputs = _TransformerRoPE(q, k, info.cosEven, info.cosOdd, info.sinEven, info.sinOdd, ropeParam);
            q = ropeOutputs[0];
            k = ropeOutputs[1];
            setName(q, attnPrefix + "q_after_rope");
            setName(k, attnPrefix + "k_after_rope");
        }

        auto attn = _GPT2Attention(numHeads, headDim, q, k, v, nullptr, nullptr, nullptr, nullptr, mask, useC4Opt);
        setName(attn, attnPrefix + "attention.out");

        if (attnOutSize != numHeads * headDim) {
            return {nullptr, nullptr};
        }
        auto o = _linear2d(attn, oWeight);
        setName(o, attnPrefix + "o_proj.out");

        // RMSNorm + MLP
        auto ln2Weight = load(blockPrefix + "post_attention_layernorm.weight");
        auto fuseLayerNorm = _BinaryLayerNorm(hiddenState, o, {ln2Weight, nullptr, ln_eps, true, hiddenSize, true});
        hiddenStateNorm = fuseLayerNorm.second;
        hiddenState = fuseLayerNorm.first;
        setName(hiddenState, blockPrefix + "resid1");
        setName(hiddenStateNorm, blockPrefix + "post_attention_layernorm.out");
        hiddenStateNorm4d = hiddenStateNorm;

        auto gateWeight = load(mlpPrefix + "gate_proj.weight");
        auto upWeight = load(mlpPrefix + "up_proj.weight");
        auto downWeight = load(mlpPrefix + "down_proj.weight");

        auto gate = _linear2d(hiddenStateNorm4d, gateWeight);
        setName(gate, mlpPrefix + "gate_proj.out");
        auto up = _linear2d(hiddenStateNorm4d, upWeight);
        setName(up, mlpPrefix + "up_proj.out");

        auto ffn = _MulSilu(up, gate);
        setName(ffn, mlpPrefix + "mul_silu.out");

        ffn = _linear2d(ffn, downWeight);
        setName(ffn, mlpPrefix + "down_proj.out");

        return {hiddenState, ffn};
    }

private:
    const Converter* mConverter = nullptr;
};

void HuggingFaceQwen3Convert(const Converter* converter, MNN::NetT* dst, const HuggingFaceQwen3Config& config) {
    if (nullptr == converter || nullptr == dst) {
        return;
    }

    HuggingFaceQwen3 qwen3(converter);

    int blockSize = config.blockNumber;
    if (blockSize <= 0) {
        const int maxBlockSize = 256;
        for (int blockIndex = 0; blockIndex < maxBlockSize; ++blockIndex) {
            auto prefix = std::string("model.layers.") + std::to_string(blockIndex) + ".self_attn.q_proj.weight";
            if (!converter->hasTensor(prefix)) {
                blockSize = blockIndex;
                break;
            }
        }
    }

    int hiddenSize = config.hiddenSize;
    int headDim = config.headDim;
    int numHead = config.numHead;

    if (hiddenSize <= 0 || headDim <= 0 || numHead <= 0) {
        auto qWeight0 = converter->loadTensor("model.layers.0.self_attn.q_proj.weight");
        auto kWeight0 = converter->loadTensor("model.layers.0.self_attn.k_proj.weight");
        if (nullptr != qWeight0.get() && nullptr != qWeight0->getInfo() && qWeight0->getInfo()->dim.size() >= 2) {
            const int queryHiddenSize = qWeight0->getInfo()->dim[0];
            const int inputHiddenSize = qWeight0->getInfo()->dim[1];
            if (hiddenSize <= 0) {
                hiddenSize = inputHiddenSize;
            }

            if (numHead > 0 && headDim <= 0 && queryHiddenSize % numHead == 0) {
                headDim = queryHiddenSize / numHead;
            } else if (headDim > 0 && numHead <= 0 && queryHiddenSize % headDim == 0) {
                numHead = queryHiddenSize / headDim;
            } else if (headDim <= 0 && numHead <= 0) {
                int kvHiddenSize = 0;
                if (nullptr != kWeight0.get() && nullptr != kWeight0->getInfo() && kWeight0->getInfo()->dim.size() >= 2) {
                    kvHiddenSize = kWeight0->getInfo()->dim[0];
                }
                static const int candidates[] = {128, 96, 80, 72, 64, 48, 40, 32};
                for (int cand : candidates) {
                    if (cand <= 0) {
                        continue;
                    }
                    if (queryHiddenSize % cand != 0) {
                        continue;
                    }
                    if (kvHiddenSize > 0 && kvHiddenSize % cand != 0) {
                        continue;
                    }
                    int candHead = queryHiddenSize / cand;
                    if (candHead > 0 && candHead <= 64) {
                        headDim = cand;
                        numHead = candHead;
                        break;
                    }
                }
            }
        }
    }

    HuggingFaceQwen3::BlockInfo blockInfo;
    blockInfo.hiddenSize = hiddenSize > 0 ? hiddenSize : 1024;
    blockInfo.headDim = headDim > 0 ? headDim : 128;
    blockInfo.numberHead = numHead > 0 ? numHead : 16;
    blockInfo.ropeCutHeadDim = config.ropeCutHeadDim;

    auto embed = _Input({1, -1, blockInfo.hiddenSize}, NCHW, halide_type_of<float>());
    embed->setName("input_embedding");

    auto position = _Input({1, -1}, NCHW, halide_type_of<int>());
    position->setName("position_ids");

    auto mask = _Input({}, NCHW, halide_type_of<float>());
    mask->setName("mask");

    auto one = _Unsqueeze(_Scalar<int32_t>(1), {0});
    auto negone = _Unsqueeze(_Scalar<int>(-1), {0});
    auto shapeHiddenState = _Shape(embed, true);
    auto seqLenVar = _Slice(shapeHiddenState, _Unsqueeze(_Scalar<int32_t>(1), {0}), one);
    auto batchLenVar = _Slice(shapeHiddenState, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);

    auto headDimVar = _Unsqueeze(_Scalar<int32_t>(blockInfo.headDim), {0});

    const int posEmbEnd = config.maxPositionEmbeddings > 0 ? config.maxPositionEmbeddings : 32768;
    const float ropeTheta = config.ropeTheta > 0.0f ? config.ropeTheta : 100000.0f;
    auto posEmb = _PrecomputePosEmbedding(blockInfo.headDim, posEmbEnd, ropeTheta);
    posEmb.fix(VARP::CONSTANT);
    posEmb->setName("precompute_posemb");

    posEmb = _GatherV2(posEmb, position, _Scalar<int>(1));
    auto cosAndsin = _Split(posEmb, {2}, 0);

    blockInfo.shapeQKV = _Concat({batchLenVar, seqLenVar, negone, headDimVar}, 0);
    blockInfo.shapeQKV->setName("shape_qkv");

    blockInfo.cosEven = _Squeeze(cosAndsin[0], {0});
    blockInfo.cosOdd = _Squeeze(cosAndsin[0], {0});
    blockInfo.sinEven = _Squeeze(cosAndsin[1], {0});
    blockInfo.sinOdd = _Squeeze(cosAndsin[1], {0});

    auto hiddenState = _Reshape(embed, {-1, hiddenSize, 1, 1});
    hiddenState = _Convert(hiddenState, NC4HW4);
    VARP add = nullptr;
    for (int blockIndex = 0; blockIndex < blockSize; ++blockIndex) {
        auto res = qwen3.makeBlock(hiddenState, add, blockInfo, mask, blockIndex);
        hiddenState = res.first;
        add = res.second;
        hiddenState->setName("block" + std::to_string(blockIndex));
    }

    // Final RMSNorm
    if (add.get() != nullptr) {
        hiddenState = _Add(hiddenState, add);
    }
    auto normWeight = converter->loadTensor("model.norm.weight");
    hiddenState = _TransformerLayerNorm(hiddenState, {normWeight, nullptr, 1.0e-6f, true, blockInfo.hiddenSize, true});
    hiddenState = _Reshape(hiddenState, shapeHiddenState);
    hiddenState->setName("hidden_state");

    std::vector<VARP> outputs = {hiddenState};
    std::vector<std::string> outputNames = {"hidden_state"};
    if (config.outputLastHiddenState) {
        auto lastHiddenState = _MakeLastHiddenStateOutput(hiddenState, blockInfo.hiddenSize);
        outputs.emplace_back(lastHiddenState);
        outputNames.emplace_back("last_hidden_state");
    }

    Variable::save(outputs, dst);
    dst->sourceType = NetSource_ONNX;
    dst->outputName = std::move(outputNames);
}

namespace {
static bool _convertHuggingFaceDecoderModel(const Converter* converter, const rapidjson::Value* model, modelConfig& modelPath) {
    if (nullptr == converter) {
        return false;
    }

    auto netT = std::unique_ptr<MNN::NetT>(new MNN::NetT);
    HuggingFaceQwen3Config config;

    if (nullptr != model && model->IsObject()) {
        auto blocks = WorkflowJson::getArray(*model, "blocks");
        if (nullptr != blocks) {
            for (auto& block : blocks->GetArray()) {
                if (!block.IsObject()) {
                    continue;
                }
                auto type = WorkflowJson::getString(block, "type");
                if (type == "QwenTransformer") {
                    config.hiddenSize = WorkflowJson::getInt(block, "hiddenSize", config.hiddenSize);
                    config.headDim = WorkflowJson::getInt(block, "headDim", config.headDim);
                    config.numHead = WorkflowJson::getInt(block, "numHead", config.numHead);
                    config.kvNumHead = WorkflowJson::getInt(block, "kvNumHead", config.kvNumHead);
                    config.blockNumber = WorkflowJson::getInt(block, "number", config.blockNumber);
                    config.maxPositionEmbeddings = WorkflowJson::getInt(block, "maxPositionEmbeddings", config.maxPositionEmbeddings);
                    config.maxPositionEmbeddings = WorkflowJson::getInt(block, "max_position_embeddings", config.maxPositionEmbeddings);
                    // backward compatible field name (legacy)
                    config.maxPositionEmbeddings = WorkflowJson::getInt(block, "bit", config.maxPositionEmbeddings);
                    config.ropeTheta = WorkflowJson::getFloat(block, "ropeTheta", config.ropeTheta);
                    config.ropeTheta = WorkflowJson::getFloat(block, "rope_theta", config.ropeTheta);
                    config.ropeCutHeadDim = WorkflowJson::getInt(block, "ropeCutHeadDim", config.ropeCutHeadDim);
                    config.ropeCutHeadDim = WorkflowJson::getInt(block, "rope_cut_head_dim", config.ropeCutHeadDim);
                    break;
                }
            }
        }
    }

    auto path = modelPath.MNNModel;
    modelPath.MNNModel = path + "decoder.mnn";
    HuggingFaceQwen3Convert(converter, netT.get(), config);
    optimizeAndWrite(modelPath, netT);
    return true;
}

REGISTER_SAFETENSOR_MODEL_BUILDER("hf_decoder", _convertHuggingFaceDecoderModel);
} // namespace

} // namespace SafeTensors
} // namespace MNN
