// Copyright © Alibaba Group Holding Limited. All rights reserved.
#include "CoreMLAttention.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/AttentionMaskConvention.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cmath>
namespace MNN {
CoreMLAttention::CoreMLAttention(Backend* bn, const Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs)
    : CoreMLCommonExecution(bn, op) {
    mKVCache = op->main_as_AttentionParam()->kv_cache();
    // Stateful decoding has a fixed capacity and is not part of the CV backend contract.
    const char* experimental = std::getenv("MNN_COREML_EXPERIMENTAL_KV");
    mValid = !mKVCache || (experimental && std::strcmp(experimental, "1") == 0);
}
ErrorCode CoreMLAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || inputs.size() > 4) {
        return NOT_SUPPORT;
    }
    auto q = inputs[0]->shape(), k = inputs[1]->shape(), v = inputs[2]->shape();
    bool packedValue = TensorUtils::getDescribe(inputs[2])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    if (packedValue && k.size() == 4 && v.size() == 4 && v[0] == k[1] && v[2] == 1 && v[3] == 1 && k[2] > 0 &&
        v[1] % k[2] == 0) {
        v = {1, k[1], k[2], v[1] / k[2]};
    }
    if (q.size() != 4 || k.size() != 4 || v.size() != 4 || q[0] != 1 || k[0] != 1 || v[0] != 1 || k[2] <= 0 ||
        q[2] % k[2] || k[2] != v[2] || q[3] != k[3] || k[1] != v[1]) {
        MNN_ERROR("[CoreML] Unsupported Q/K/V shapes:");
        for (const auto& dims : {q, k, v}) {
            MNN_ERROR(" [");
            for (auto dim : dims) {
                MNN_ERROR(" %d", dim);
            }
            MNN_ERROR(" ]");
        }
        MNN_ERROR("\n");
        return NOT_SUPPORT;
    }
    mSeqLen = q[1];
    mKeyLen = k[1];
    if (mKVCache && mKeyLen != mSeqLen) {
        return NOT_SUPPORT;
    }
    int capacity = mKVCache ? mMaxSeqLen : mKeyLen;
    if (mKeyLen > capacity) {
        return INPUT_DATA_ERROR;
    }
    mMask.reset(Tensor::create({1, 1, mSeqLen, capacity}, halide_type_t(halide_type_float, 16)));
    auto maskName = mCoreMLBackend->addInput(mMask.get(), "attention_mask");
    std::vector<std::string> data(3);
    std::vector<std::vector<int>> shape = {q, k, v};
    for (int i = 0; i < 3; ++i) {
        auto trans = operation("transpose");
        auto name = mCoreMLBackend->getTensorName(inputs[i]);
        if (i == 2 && packedValue) {
            name = operation("reshape").input("x", name).shape("shape", v, "value_shape").output(v, "value");
        }
        trans.input("x", name).array<int>("perm", {0, 2, 1, 3}, "qkv_perm");
        data[i] = trans.output({1, shape[i][2], shape[i][1], shape[i][3]}, "qkv_" + std::to_string(i));
    }
    if (mKVCache) {
        // Explicit floating-point placement avoids dynamic slice indices and
        // keeps every token position on the same static-shaped program.
        mUpdate.reset(Tensor::create({1, 1, capacity, mKeyLen}, halide_type_t(halide_type_float, 16)));
        mKeep.reset(Tensor::create({1, 1, capacity, 1}, halide_type_t(halide_type_float, 16)));
        auto updateMask = mCoreMLBackend->addInput(mUpdate.get(), "kv_update");
        auto keepMask = mCoreMLBackend->addInput(mKeep.get(), "kv_keep");
        for (int i = 1; i < 3; ++i) {
            std::string suffix = std::to_string(i);
            std::vector<int> stateShape = {1, shape[i][2], capacity, shape[i][3]};
            auto state = mCoreMLBackend->addState("kv_state_" + suffix, stateShape);
            auto cache = operation("read_state").input("input", state).output(stateShape, "cache_" + suffix);
            auto placed = operation("matmul")
                              .input("x", updateMask)
                              .input("y", data[i])
                              .scalar("transpose_x", false)
                              .scalar("transpose_y", false)
                              .output(stateShape, "placed_" + suffix);
            auto kept = operation("mul").input("x", cache).input("y", keepMask).output(stateShape, "kept_" + suffix);
            auto updated = operation("add").input("x", kept).input("y", placed).output(stateShape, "updated_" + suffix);
            operation("write_state").input("data", updated).input("input", state).finish("write_" + suffix);
            data[i] = operation("read_state").input("input", state).output(stateShape, "committed_" + suffix);
        }
    }
    // GQA: repeat each KV head for its corresponding group of query heads.
    int group = q[2] / k[2];
    if (group > 1) {
        for (int i = 1; i < 3; ++i) {
            auto suffix = std::to_string(i);
            auto expanded = operation("expand_dims")
                                .input("x", data[i])
                                .array<int>("axes", {2}, "gqa_axis")
                                .output({1, k[2], 1, capacity, shape[i][3]}, "expand_" + suffix);
            auto tiled = operation("tile")
                             .input("x", expanded)
                             .array<int>("reps", {1, 1, group, 1, 1}, "gqa_reps")
                             .output({1, k[2], group, capacity, shape[i][3]}, "tile_" + suffix);
            data[i] = operation("reshape")
                          .input("x", tiled)
                          .shape("shape", {1, q[2], capacity, shape[i][3]}, "gqa_shape_" + suffix)
                          .output({1, q[2], capacity, shape[i][3]}, "gqa_" + suffix);
        }
    }
    // Keep attention scores in fp32: fp16 rounding of long-context logits is
    // amplified by softmax even when the projections and KV storage are fp16.
    const auto fp32 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32;
    auto castFloat = [&](const std::string& name, const std::vector<int>& dims, const std::string& suffix) {
        return operation("cast").input("x", name).scalar("dtype", "fp32", "score_dtype").output(dims, suffix, fp32);
    };
    auto query = castFloat(data[0], {1, q[2], mSeqLen, q[3]}, "query_float");
    auto key = castFloat(data[1], {1, q[2], capacity, k[3]}, "key_float");
    auto value = castFloat(data[2], {1, q[2], capacity, v[3]}, "value_float");
    auto mask = castFloat(maskName, {1, 1, mSeqLen, capacity}, "mask_float");
    auto meta = static_cast<KVMeta*>(backend()->getMetaPtr());
    float scale = meta && meta->attn_scale > 0 ? meta->attn_scale : 1.0f / std::sqrt(static_cast<float>(q[3]));
    query = operation("mul")
                .input("x", query)
                .scalarFloat32("y", scale, "scale")
                .output({1, q[2], mSeqLen, q[3]}, "scaled_query", fp32);
    auto scores = operation("matmul")
                      .input("x", query)
                      .input("y", key)
                      .scalar("transpose_x", false, "dot_transpose_x")
                      .scalar("transpose_y", true, "dot_transpose_y")
                      .output({1, q[2], mSeqLen, capacity}, "scores", fp32);
    scores = operation("add").input("x", scores).input("y", mask).output({1, q[2], mSeqLen, capacity}, "masked", fp32);
    auto probabilities = operation("softmax")
                             .input("x", scores)
                             .scalar("axis", -1, "softmax_axis")
                             .output({1, q[2], mSeqLen, capacity}, "probabilities", fp32);
    auto attention = operation("matmul")
                         .input("x", probabilities)
                         .input("y", value)
                         .scalar("transpose_x", false, "product_transpose_x")
                         .scalar("transpose_y", false, "product_transpose_y")
                         .output({1, q[2], mSeqLen, v[3]}, "attention_float", fp32);
    auto result = operation("cast")
                      .input("x", attention)
                      .scalar("dtype", "fp16", "output_dtype")
                      .output({1, q[2], mSeqLen, v[3]}, "attention");
    auto transposed = operation("transpose")
                          .input("x", result)
                          .array<int>("perm", {0, 2, 1, 3}, "post_perm")
                          .output({1, mSeqLen, q[2], v[3]}, "post");
    operation("reshape").input("x", transposed).shape("shape", outputs[0]->shape(), "output_shape").output(outputs);
    return NO_ERROR;
}
ErrorCode CoreMLAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    int past = 0;
    auto meta = static_cast<KVMeta*>(backend()->getMetaPtr());
    if (mKVCache && meta) {
        if (meta->remove > meta->previous || meta->n_reserve || meta->file_flag != KVMeta::NoChange ||
            (meta->add != 0 && meta->add != static_cast<size_t>(mKeyLen))) {
            return NOT_SUPPORT;
        }
        past = static_cast<int>(meta->previous - meta->remove);
    }
    int length = past + mKeyLen;
    int capacity = mKVCache ? mMaxSeqLen : mKeyLen;
    if (length > capacity) {
        MNN_ERROR("[CoreML] KV cache capacity exceeded: %d > %d\n", length, capacity);
        return INPUT_DATA_ERROR;
    }
    if (mKVCache) {
        std::fill_n(mKeep->host<int16_t>(), capacity, static_cast<int16_t>(0x3c00));
        std::fill_n(mUpdate->host<int16_t>(), mUpdate->elementSize(), static_cast<int16_t>(0));
        for (int i = 0; i < mKeyLen; ++i) {
            mKeep->host<int16_t>()[past + i] = 0;
            mUpdate->host<int16_t>()[static_cast<size_t>(past + i) * mKeyLen + i] = 0x3c00;
        }
    }
    std::vector<float> mask(static_cast<size_t>(mSeqLen) * capacity, -std::numeric_limits<float>::infinity());
    const Tensor* supplied = inputs.size() > 3 ? inputs[3] : nullptr;
    int suppliedCount = supplied ? static_cast<int>(TensorUtils::getRawSize(supplied)) : 0;
    int maskRank = supplied ? supplied->dimensions() : 0;
    bool causal = IsAttentionMaskCausal(supplied != nullptr, maskRank, mKVCache);
    bool dense = !IsAttentionMaskCausalSentinel(supplied != nullptr, maskRank) &&
                 !IsAttentionMaskBidirectionalSentinel(supplied != nullptr, maskRank, suppliedCount);
    std::vector<float> suppliedFloat;
    int cols = dense ? supplied->length(maskRank - 1) : 0;
    if (dense && (cols <= 0 || !supplied->host<void>())) {
        return INPUT_DATA_ERROR;
    }
    int rows = dense ? suppliedCount / cols : 0;
    if (dense && rows != 1 && rows != mSeqLen) {
        return NOT_SUPPORT;
    }
    if (dense && supplied->getType().code == halide_type_float) {
        suppliedFloat.resize(suppliedCount);
        MNNDequantizeFP16(supplied->host<int16_t>(), suppliedFloat.data(), suppliedCount);
    }
    for (int row = 0; row < mSeqLen; ++row) {
        for (int col = 0; col < length; ++col) {
            float value = 0;
            if (causal) {
                if (col > past + row) {
                    continue;
                }
            } else if (dense) {
                int column = col - std::max(0, length - cols);
                if (column >= 0 && column < cols) {
                    int index = (rows == 1 ? 0 : row) * cols + column;
                    if (index >= suppliedCount) {
                        return INPUT_DATA_ERROR;
                    }
                    if (supplied->getType() == halide_type_of<int>()) {
                        if (supplied->host<int>()[index] == 0) {
                            continue;
                        }
                    } else {
                        value = suppliedFloat[index];
                    }
                }
            }
            mask[static_cast<size_t>(row) * capacity + col] = value;
        }
    }
    MNNQuantizeFP16(mask.data(), mMask->host<int16_t>(), mask.size());
    return NO_ERROR;
}
REGISTER_COREML_OP_CREATOR(CoreMLAttention, OpType_Attention);
} // namespace MNN
