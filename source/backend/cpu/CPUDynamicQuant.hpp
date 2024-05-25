//
//  CPUDynamicQuant.hpp
//  MNN
//
//  Created by MNN on 2023/07/11
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDynamicQuant_hpp
#define CPUDynamicQuant_hpp

#include "core/Execution.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
class CPUDynamicQuant : public Execution {
public:
    CPUDynamicQuant(const MNN::Op* op, Backend* backend);
    virtual ~CPUDynamicQuant();

    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
};
} // namespace MNN
#endif /* CPUDynamicQuant_hpp */
