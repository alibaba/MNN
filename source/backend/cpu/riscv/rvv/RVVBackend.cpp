//
//  RVVBackend.cpp
//  MNN
//
//  Created by MNN on 2026/11/19.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "RVVBackend.hpp"
#include "MNNRvvPrefill.hpp"
#include "backend/cpu/CPURaster.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#if defined(MNN_USE_SPACEMIT_IME2) && defined(MNN_LOW_MEMORY) && defined(MNN_SUPPORT_TRANSFORMER_FUSE)
#include "backend/cpu/riscv/spacemit_ime2/MNNSpacemitIme2ConvInt8Executor.hpp"
#endif

namespace MNN {
namespace {
class RvvMulSilu final : public Execution {
public:
    using Execution::Execution;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const auto cpu = static_cast<CPUBackend*>(backend());
        const int size = cpu->getTensorSize(outputs[0]);
        const auto schedule = cpu->multiThreadDivide(size);
        const bool low = cpu->precisionMode() == BackendConfig::Precision_Low;
        MNN_CONCURRENCY_BEGIN(id, schedule.second) {
            const int start = id * schedule.first;
            const int count = id == schedule.second - 1 ? size - start : schedule.first;
            if (count > 0) {
                MNNMulSiLu_RVV(outputs[0]->host<float>() + start, inputs[1]->host<float>() + start,
                               inputs[0]->host<float>() + start, count, low);
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
};

class RvvRaster final : public CPURaster {
public:
    using CPURaster::CPURaster;
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto out = outputs[0];
        mInput = nullptr;
        if (inputs.size() == 1 && CPUBackend::getBytes(backend(), out) == 4) {
            OpCommonUtils::rasterInputReset(inputs, out);
            if (canCopyRows(inputs[0], out)) {
                mInput = inputs[0];
                return NO_ERROR;
            }
        }
        return CPURaster::onResize(inputs, outputs);
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mInput) {
            return CPURaster::onExecute(inputs, outputs);
        }
        auto out = outputs[0];
        const int rows = out->batch(), width = mInput->channel(), heads = out->channel() / width;
        const int workers = std::min(static_cast<CPUBackend*>(backend())->threadNumber(), width / 4);
        auto src = mInput->host<float>();
        auto dst = out->host<float>();
        MNN_CONCURRENCY_BEGIN(id, workers) {
            for (int c = id * 4; c < width; c += workers * 4) {
                for (int r = 0; r < rows; r += 16) {
                    const int count = std::min(16, rows - r);
                    for (int h = 0; h < heads; ++h) {
                        auto from = src + size_t(c) * rows * heads + (size_t(r) * heads + h) * 4;
                        auto to = dst + (size_t(h) * width + c) * rows + r * 4;
                        for (int j = 0; j < count; ++j) {
                            ::memcpy(to + j * 4, from + j * heads * 4, 4 * sizeof(float));
                        }
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
private:
    // The logical element ((r * heads + h) * width + c) is preserved by a
    // contiguous reshape. Packing its channel dimension in groups of four
    // gives different physical addresses on the two sides when width > 4.
    static bool canCopyRows(Tensor* in, Tensor* out) {
        auto matrix = [](Tensor* t) {
            return t && TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                   t->dimensions() == 4 && t->width() == 1 && t->height() == 1;
        };
        const auto& regions = TensorUtils::getDescribe(out)->regions;
        if (!matrix(in) || !matrix(out) || regions.size() != 1 || out->batch() <= 0 ||
            in->channel() <= 0 || in->channel() % 4 || out->channel() <= in->channel() ||
            out->channel() % in->channel() || in->batch() != size_t(out->batch()) * (out->channel() / in->channel())) {
            return false;
        }
        const auto& region = regions[0];
        if (region.origin != in || region.src.offset || region.dst.offset || !TensorUtils::isCopyRegion(region)) {
            return false;
        }
        size_t span = 1;
        for (int d = 2; d >= 0; --d) {
            if (region.size[d] <= 0 || (region.size[d] > 1 &&
                (region.src.stride[d] != span || region.dst.stride[d] != span))) {
                return false;
            }
            span *= region.size[d];
        }
        return span == size_t(in->batch()) * in->channel();
    }
    Tensor* mInput = nullptr;
};

// The RVV C4 path unpacks directly into each token's output. The CPU rotation
// kernel supports in-place input, including after Q/K normalization.
class RvvRoPE final : public Execution {
public:
    static Execution* create(CPUBackend* cpu, const Op* op, const std::vector<Tensor*>& inputs) {
        const auto p = op->main_as_RoPEParam();
        if (!p || p->num_head() <= 0 || p->kv_num_head() <= 0 || p->head_dim() <= 0 ||
            (int64_t(p->head_dim()) * p->num_head()) % 4 || (int64_t(p->head_dim()) * p->kv_num_head()) % 4) {
            return nullptr;
        }
        std::shared_ptr<RoPEParamT> param(p->UnPack());
        for (auto norm : {param->q_norm.get(), param->k_norm.get()}) {
            if (norm && !norm->gamma.empty()) {
                if (norm->gamma.size() < size_t(p->head_dim())) {
                    return nullptr;
                }
                if (norm->beta.size() != norm->gamma.size()) {
                    norm->beta.assign(norm->gamma.size(), 0.0f);
                }
            }
        }
        std::unique_ptr<RvvRoPE> exe(new RvvRoPE(cpu, std::move(param)));
        return exe->supports(inputs) ? exe.release() : nullptr;
    }

    RvvRoPE(Backend* backend, std::shared_ptr<const RoPEParamT> param) : Execution(backend), mParam(std::move(param)) {}

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        return supports(inputs) ? NO_ERROR : NOT_SUPPORT;
    }

    bool supports(const std::vector<Tensor*>& inputs) const {
        if (inputs.size() < 4) return false;
        for (int j = 0; j < 2; ++j) {
            const auto t = inputs[j];
            const int heads = j == 0 ? mParam->num_head : mParam->kv_num_head;
            if (TensorUtils::getDescribe(t)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 || t->dimensions() != 4 ||
                t->length(0) != inputs[0]->length(0) || t->length(1) != heads * mParam->head_dim ||
                t->length(2) != 1 || t->length(3) != 1) {
                return false;
            }
        }
        return true;
    }

    bool onClone(Backend* bn, const Op*, Execution** dst) override {
        if (!dst) {
            return true;
        }
        if (!bn || bn->type() != MNN_FORWARD_CPU) {
            return false;
        }
        auto core = static_cast<CPUBackend*>(bn)->functions();
        if (!core->supportRVV || core->bytes != 4 || core->pack != 4 || !core->MNNRoPECompute) {
            return false;
        }
        *dst = new RvvRoPE(bn, mParam);
        return true;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto cpu = static_cast<CPUBackend*>(backend());
        const int tokens = inputs[0]->length(0);
        const int workers = std::min(cpu->computeThreadNumber(tokens), tokens);
        const int dim = mParam->head_dim;
        int cut = mParam->rope_cut_head_dim;
        cut = (cut > 0 && cut <= dim ? cut : dim) / 2 * 2;
        auto rotate = cpu->functions()->MNNRoPECompute;
        MNN_CONCURRENCY_BEGIN(worker, workers) {
            for (int t = tokens * worker / workers; t < tokens * (worker + 1) / workers; ++t) {
                const float* cos = inputs[2]->host<float>() + size_t(t) * cut;
                const float* sin = inputs[3]->host<float>() + size_t(t) * cut;
                for (int j = 0; j < 2; ++j) {
                    const int heads = j == 0 ? mParam->num_head : mParam->kv_num_head;
                    const auto& norm = j == 0 ? mParam->q_norm : mParam->k_norm;
                    auto out = outputs[j]->host<float>() + size_t(t) * heads * dim;
                    auto src = inputs[j]->host<float>() + t * 4;
                    for (int c = 0; c < heads * dim; c += 4) {
                        ::memcpy(out + c, src + size_t(c) * tokens, 4 * sizeof(float));
                    }
                    if (norm && !norm->gamma.empty()) {
                        for (int h = 0; h < heads; ++h) {
                            float* row = out + h * dim;
                            MNNNorm(row, row, norm->gamma.data(), norm->beta.data(), norm->epsilon, dim, norm->useRMSNorm);
                        }
                    }
                    rotate(out, out, cos, cos + cut / 2, sin, sin + cut / 2, heads, dim, mParam->rope_cut_head_dim);
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
private:
    std::shared_ptr<const RoPEParamT> mParam;
};

} // namespace

Execution* RVVBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op) {
    if (op->type() == OpType_Raster && functions()->pack == 4) {
        return new RvvRaster(this);
    }
    if (op->type() == OpType_BinaryOp && op->main_as_BinaryOp()->opType() == BinaryOpOperation_MUL_SILU &&
        functions()->bytes == 4) {
        return new RvvMulSilu(this);
    }
    if (op->type() == OpType_RoPE && functions()->pack == 4 && functions()->bytes == 4 && functions()->MNNRoPECompute) {
        if (auto execution = RvvRoPE::create(this, op, inputs)) {
            return execution;
        }
    }
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    if (op->type() == OpType_LinearAttention) {
        if (auto execution = MNNRvvCreateLinearAttention(this, op, inputs)) {
            return execution;
        }
    }
#endif
    auto exe = CPUBackend::onCreate(inputs, outputs, op);
#if defined(MNN_USE_SPACEMIT_IME2) && defined(MNN_LOW_MEMORY) && defined(MNN_SUPPORT_TRANSFORMER_FUSE)
    if (op->type() == OpType_FusedLinear && exe) {
        return MNNSpacemitIme2WrapFusedProjection(exe);
    }
#endif
    return exe;
}
} // namespace MNN
