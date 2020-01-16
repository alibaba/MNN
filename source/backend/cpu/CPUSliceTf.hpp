//
//  CPUSliceTf.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSliceTf_hpp
#define CPUSliceTf_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUSliceTf : public Execution {
public:
    CPUSliceTf(Backend *b);
    virtual ~CPUSliceTf() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN
#endif /* CPUSliceTf_hpp */
