//
//  RVVBackend.hpp
//  MNN
//
//  Created by MNN on 2026/11/19.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#pragma once

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class RVVBackend final : public CPUBackend {
public:
    using CPUBackend::CPUBackend;
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op) override;
};
} // namespace MNN
