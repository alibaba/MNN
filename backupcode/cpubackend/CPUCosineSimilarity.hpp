//
//  CPUCosineSimilarity.hpp
//  MNN
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCosineSimilarity_hpp
#define CPUCosineSimilarity_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUCosineSimilarity : public Execution {
public:
    CPUCosineSimilarity(Backend *bn, const MNN::Op *op) : MNN::Execution(bn) {
        // nothing to do
    }
    virtual ~CPUCosineSimilarity() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUCosineSimilarity */
