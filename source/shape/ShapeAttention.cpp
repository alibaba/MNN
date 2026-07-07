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
            TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 || q->dimensions() < 2 ||
            k->dimensions() < 2 || q->length(1) != param->num_head() * param->head_dim() ||
            k->length(1) != param->kv_num_head() * param->head_dim()) {
            MNN_ERROR("RoPE: input must be C4 packed q/k tensors.\n");
            return false;
        }
        auto qo = outputs[0], ko = outputs[1];
        qo->buffer().dimensions = 4;
        qo->buffer().dim[0].extent = 1;
        qo->buffer().dim[1].extent = q->length(0);
        qo->buffer().dim[2].extent = param->num_head();
        qo->buffer().dim[3].extent = param->head_dim();
        qo->buffer().type = q->buffer().type;
        TensorUtils::getDescribe(qo)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        ko->buffer().dimensions = 4;
        ko->buffer().dim[0].extent = 1;
        ko->buffer().dim[1].extent = k->length(0);
        ko->buffer().dim[2].extent = param->kv_num_head();
        ko->buffer().dim[3].extent = param->head_dim();
        ko->buffer().type = k->buffer().type;
        TensorUtils::getDescribe(ko)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
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

class AttentionSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input = inputs[0], output = outputs[0];
        MNN_ASSERT(input->buffer().dimensions == 4);
        if (op->main_as_AttentionParam()->output_c4()) {
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
        auto seqLen = static_cast<float>(outputs[0]->length(1));
        auto headDim = static_cast<float>(outputs[0]->length(2));
        float flops = 0.f;
        // qk + qkv
        flops += (2 * seqLen * headDim * seqLen);
        // softmax
        flops += (seqLen * seqLen);
        return flops / FLOPS_M;
    }
};

class LinearAttentionSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input = inputs[0];
        auto output = outputs[0];
        auto param = op->main_as_LinearAttentionParam();

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
        float L = static_cast<float>(input->length(2));
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
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(AttentionSizeComputer, OpType_Attention);
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(LinearAttentionSizeComputer, OpType_LinearAttention);
#endif

} // namespace MNN
