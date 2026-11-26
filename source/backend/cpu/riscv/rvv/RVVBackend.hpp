//
//  RVVBackend.hpp
//  MNN
//
//  Created by MNN on 2026/11/19.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#pragma once

#include "backend/cpu/CPUBackend.hpp"
#include <unordered_map>
#include <unordered_set>

namespace MNN {

struct RvvLinearAttnHandoff {
    const float* ptr = nullptr;
    int batch = 0;
    int length = 0;
    int heads = 0;
    int dv = 0;
};

struct RvvGateUpHandoff {
    const float* ptr = nullptr;
    const void* packedA = nullptr;
    size_t rows = 0;
    size_t depth = 0;
};

struct RvvHandoffContext {
    const Backend* backend = nullptr;
    std::unordered_set<const Tensor*> linearAttnToGatedRms;
    std::unordered_map<const Tensor*, RvvLinearAttnHandoff> activeLinearAttn;
    std::unordered_set<const Tensor*> gateUpToDownProj;
    std::unordered_map<const Tensor*, RvvGateUpHandoff> activeGateUp;
};

RvvHandoffContext* MNNRvvGetHandoffContext();
void MNNRvvRegisterHalfConv(const Backend* backend, const Execution* exe);

class RVVBackend final : public CPUBackend {
public:
    using CPUBackend::CPUBackend;
    using CPUBackend::onResizeEnd;
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op) override;
    bool onClearBuffer() override;
    void onResizeBegin() override;
    ErrorCode onResizeEnd() override;
    ErrorCode onResizeEnd(const std::vector<Command*>& commands) override;
    void onExecuteBegin() const override;
    void onExecuteEnd() const override;

private:
    std::unordered_set<const Execution*> mResizeHalfConvs;
    mutable RvvHandoffContext mHandoff;
    mutable RvvHandoffContext* mPrevHandoff = nullptr;
};
} // namespace MNN
