//
//  CPUNonMaxSuppressionV2.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUNonMaxSuppressionV2_hpp
#define CPUNonMaxSuppressionV2_hpp

#include "Execution.hpp"

namespace MNN {
template <typename T>
class CPUNonMaxSuppressionV2 : public Execution {
public:
    CPUNonMaxSuppressionV2(Backend *backend, const Op *op);
    virtual ~CPUNonMaxSuppressionV2() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUNonMaxSuppressionV2_hpp */
