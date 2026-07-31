//
//  CPUExtension.hpp
//  MNN
//
//  Created by MNN on 2026/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUExtension_hpp
#define CPUExtension_hpp

#include <memory>

#include "core/ConvolutionCommon.hpp"

namespace MNN {

struct CPUExtension {
    using CreateInt8GemmExecution = Execution* (*)(Backend*, const Op*, std::shared_ptr<ConvolutionCommon::Int8Common>,
                                                   bool);
    using CreateAttentionExecution = Execution* (*)(Backend*, bool);

    constexpr CPUExtension(CreateInt8GemmExecution int8Gemm = nullptr, CreateAttentionExecution attention = nullptr)
        : createInt8GemmExecution(int8Gemm), createAttentionExecution(attention) {}

    CreateInt8GemmExecution createInt8GemmExecution;
    CreateAttentionExecution createAttentionExecution;
};

} // namespace MNN

#endif /* CPUExtension_hpp */
