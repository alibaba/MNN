//
//  ShapeAttention.cpp
//  MNN
//
//  Created by MNN on 2023/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
class RoPESizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 4);
        MNN_ASSERT(outputs.size() == 2);
        auto param = op->main_as_RoPEParam();
        if (param == nullptr || param->num_head() <= 0 || param->kv_num_head() <= 0 || param->head_dim() <= 0) {
            MNN_ERROR("RoPE: invalid C4 head config.\n");
            return false;
        }
        auto q = inputs[0], k = inputs[1];
        if (TensorUtils::getDescribe(q)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 || q->dimensions() != 4 ||
            k->dimensions() != 4 || q->length(0) != k->length(0) || q->length(2) != 1 || q->length(3) != 1 ||
            k->length(2) != 1 || k->length(3) != 1 || q->length(1) != param->num_head() * param->head_dim() ||
            k->length(1) != param->kv_num_head() * param->head_dim()) {
            MNN_ERROR("RoPE: input must be C4 packed q/k tensors.\n");
            return false;
        }
        auto qo = outputs[0], ko = outputs[1];
        qo->buffer().dimensions = 4;
        qo->buffer().dim[0].extent = q->length(0);
        qo->buffer().dim[1].extent = q->length(1);
        qo->buffer().dim[2].extent = 1;
        qo->buffer().dim[3].extent = 1;
        qo->buffer().type = q->buffer().type;
        TensorUtils::getDescribe(qo)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        ko->buffer().dimensions = 4;
        ko->buffer().dim[0].extent = k->length(0);
        ko->buffer().dim[1].extent = k->length(1);
        ko->buffer().dim[2].extent = 1;
        ko->buffer().dim[3].extent = 1;
        ko->buffer().type = k->buffer().type;
        TensorUtils::getDescribe(ko)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        return true;
    }
};

class FmhaV2SizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input0 = inputs[0], output0 = outputs[0];
        MNN_ASSERT(inputs.size() == 1);
        MNN_ASSERT(input0->buffer().dimensions == 3);

        output0->buffer().dim[0].extent = input0->buffer().dim[0].extent;
        output0->buffer().dim[1].extent = input0->buffer().dim[1].extent;
        output0->buffer().dim[2].extent = input0->buffer().dim[2].extent / 3;
        output0->buffer().dimensions = 3;
        // MNN_PRINT("fmhaV2 shape:%d %d, %d %d %d %d %d\n", input0->buffer().dimensions, output0->buffer().dimensions,
        // input0->buffer().dim[0].extent, input0->buffer().dim[1].extent, input0->buffer().dim[2].extent,
        // input0->buffer().dim[3].extent, input0->buffer().dim[4].extent); MNN_ASSERT(input0->buffer().dim[3].extent ==
        // 3);
        output0->buffer().type = input0->buffer().type;
        TensorUtils::getDescribe(output0)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        // printf("fmhaV2 shape:%d %d, %d %d %d\n", input0->buffer().dimensions, output0->buffer().dimensions,
        // input0->buffer().dim[0].extent, input0->buffer().dim[1].extent, input0->buffer().dim[2].extent);
        return true;
    }
};

class FmhcaSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(outputs.size() == 1);
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto output0 = outputs[0];
        MNN_ASSERT(input0->buffer().dimensions == 3);
        MNN_ASSERT(input1->buffer().dimensions == 3);

        output0->buffer().dim[0].extent = input0->buffer().dim[0].extent;
        output0->buffer().dim[1].extent = input0->buffer().dim[1].extent;
        output0->buffer().dim[2].extent = input0->buffer().dim[2].extent;
        output0->buffer().dimensions = 3;
        // MNN_ASSERT(input1->buffer().dim[0].extent == input0->buffer().dim[0].extent);
        // MNN_ASSERT(input1->buffer().dim[2].extent == input0->buffer().dim[2].extent);
        // MNN_ASSERT(input1->buffer().dim[4].extent == input0->buffer().dim[3].extent);
        output0->buffer().type = input0->buffer().type;
        TensorUtils::getDescribe(output0)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        // printf("fmhca shape:%d %d %d, %d %d %d\n", input0->buffer().dimensions, input1->buffer().dimensions,
        // output0->buffer().dimensions, input0->buffer().dim[0].extent, input0->buffer().dim[1].extent,
        // input0->buffer().dim[2].extent);
        return true;
    }
};

class LayerNormC4SizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        const bool single = inputs.size() == 1 && outputs.size() == 1;
        const bool binary = inputs.size() == 2 && outputs.size() == 2;
        if ((!single && !binary) || inputs[0] == nullptr) {
            return false;
        }
        if (binary && (inputs[0]->shape() != inputs[1]->shape() || inputs[0]->getType() != inputs[1]->getType())) {
            MNN_ERROR("LayerNorm: residual inputs must have the same shape and type.\n");
            return false;
        }
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (op->defaultDimentionFormat() == MNN_DATA_FORMAT_NC4HW4) {
            format = MNN_DATA_FORMAT_NC4HW4;
        }
        const auto& inputBuffer = inputs[0]->buffer();
        for (auto output : outputs) {
            auto& outputBuffer = output->buffer();
            outputBuffer.dimensions = inputBuffer.dimensions;
            ::memcpy(outputBuffer.dim, inputBuffer.dim, sizeof(halide_dimension_t) * inputBuffer.dimensions);
            outputBuffer.type = inputBuffer.type;
            TensorUtils::getDescribe(output)->dimensionFormat = format;
        }
        return true;
    }
};

class AttentionSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input = inputs[0], output = outputs[0];
        auto param = op->main_as_AttentionParam();
        if (param == nullptr || input == nullptr || output == nullptr || input->buffer().dimensions != 4) {
            return false;
        }
        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            auto headDim = param->head_dim();
            if (inputs.size() < 3 || headDim <= 0 || input->dimensions() != 4 || inputs[1]->dimensions() != 4 ||
                inputs[2]->dimensions() != 4 || inputs[1]->length(0) != inputs[2]->length(0) || input->length(2) != 1 ||
                input->length(3) != 1 || inputs[1]->length(2) != 1 || inputs[1]->length(3) != 1 ||
                inputs[2]->length(2) != 1 || inputs[2]->length(3) != 1 || input->length(1) <= 0 ||
                inputs[1]->length(1) <= 0 || input->length(1) % headDim != 0 || inputs[1]->length(1) % headDim != 0 ||
                inputs[1]->length(1) != inputs[2]->length(1) ||
                (input->length(1) / headDim) % (inputs[1]->length(1) / headDim) != 0 ||
                TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
                TensorUtils::getDescribe(inputs[2])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                MNN_ERROR("Attention: invalid C4 q/k/v head configuration.\n");
                return false;
            }
            output->buffer().dimensions = 4;
            for (int i = 0; i < 4; ++i) {
                output->buffer().dim[i].extent = input->buffer().dim[i].extent;
            }
            output->buffer().type = input->buffer().type;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        } else if (param->output_c4()) {
            output->buffer().dim[0].extent = input->buffer().dim[0].extent * input->buffer().dim[1].extent;
            output->buffer().dim[1].extent = input->buffer().dim[2].extent * input->buffer().dim[3].extent;
            output->buffer().dim[2].extent = 1;
            output->buffer().dim[3].extent = 1;
            output->buffer().dimensions = 4;
            output->buffer().type = input->buffer().type;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        } else {
            output->buffer().dim[0].extent = input->buffer().dim[0].extent;
            output->buffer().dim[1].extent = input->buffer().dim[1].extent;
            output->buffer().dim[2].extent = input->buffer().dim[2].extent * input->buffer().dim[3].extent;
            output->buffer().dimensions = 3;
            output->buffer().type = input->buffer().type;
            TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        }
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr) {
            return 0.f;
        }
        auto param = op->main_as_AttentionParam();
        bool inputC4 = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        auto seqLen = static_cast<float>(inputC4 ? inputs[0]->length(0) : inputs[0]->length(1));
        auto kvSeqLen = static_cast<float>(inputC4 ? inputs[1]->length(0) : inputs[1]->length(1));
        auto headDim = static_cast<float>(inputC4 && param != nullptr ? param->head_dim() : inputs[0]->length(3));
        auto numHead =
            static_cast<float>(inputC4 && headDim > 0.0f ? inputs[0]->length(1) / headDim : inputs[0]->length(2));
        float flops = 0.f;
        // qk + qkv
        flops += (2 * numHead * seqLen * headDim * kvSeqLen);
        // softmax
        flops += (numHead * seqLen * kvSeqLen);
        return flops / FLOPS_M;
    }
};

class LinearAttentionSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input = inputs[0];
        auto output = outputs[0];
        auto param = op->main_as_LinearAttentionParam();

        if (param == nullptr || input == nullptr || output == nullptr) {
            return false;
        }

        auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        if (inputFormat == MNN_DATA_FORMAT_NC4HW4) {
            // LLM Linear projections flatten batch and sequence into N. The
            // packed path currently supports the batch=1 LLM execution model.
            // Keep head_v_dim as C so per-head RMSNorm can remain packed.
            if (input->dimensions() != 4 || input->length(0) <= 0 || input->length(1) <= 0 || input->length(2) != 1 ||
                input->length(3) != 1 || param->num_v_heads() <= 0 || param->head_v_dim() <= 0) {
                MNN_ERROR("LinearAttention: invalid C4 input or head configuration.\n");
                return false;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = input->length(0) * param->num_v_heads();
            output->buffer().dim[1].extent = param->head_v_dim();
            output->buffer().dim[2].extent = 1;
            output->buffer().dim[3].extent = 1;
            output->buffer().type = input->buffer().type;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            return true;
        }

        int batch = input->length(0);
        int seq_len = input->length(2);
        int num_v_heads = param->num_v_heads();
        int head_v_dim = param->head_v_dim();

        // Output: [Batch, SeqLen, NumVHeads, HeadVDim]
        output->buffer().dimensions = 4;
        output->buffer().dim[0].extent = batch;
        output->buffer().dim[1].extent = seq_len;
        output->buffer().dim[2].extent = num_v_heads;
        output->buffer().dim[3].extent = head_v_dim;

        output->buffer().type = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto param = op->main_as_LinearAttentionParam();
        auto input = inputs[0];
        float L = static_cast<float>(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4
                                         ? input->length(0)
                                         : input->length(2));
        float D = static_cast<float>(input->length(1));
        int H = param->num_v_heads();
        int dk = param->head_k_dim();
        int dv = param->head_v_dim();
        int K = inputs[3]->length(2);
        float flops = 0.f;
        // Conv1D + SiLU: D * L * (2*K + 4)
        flops += D * L * (2.f * K + 4.f);
        // Per timestep per head: DualMatVec (4*dk*dv) + DecayRankOneUpdate (3*dk*dv) + delta (3*dv)
        flops += L * H * (7.f * dk * dv + 3.f * dv);
        return flops / FLOPS_M;
    }
};

REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(FmhaV2SizeComputer, OpType_FmhaV2);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(FmhcaSizeComputer, OpType_Fmhca);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(RoPESizeComputer, OpType_RoPE);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(LayerNormC4SizeComputer, OpType_LayerNorm);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(AttentionSizeComputer, OpType_Attention);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(LinearAttentionSizeComputer, OpType_LinearAttention);
#endif

} // namespace MNN
