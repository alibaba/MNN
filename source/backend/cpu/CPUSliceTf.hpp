//
//  CPUSliceTf.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSliceTf_hpp
#define CPUSliceTf_hpp

#include "Execution.hpp"

namespace MNN {
template <typename T>
class CPUSliceTf : public Execution {
public:
    CPUSliceTf(Backend *b, const MNN::Op *op);
    virtual ~CPUSliceTf() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN
#endif /* CPUSliceTf_hpp */
