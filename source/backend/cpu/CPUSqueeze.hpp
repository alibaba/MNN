//
//  CPUSqueeze.hpp
//  MNN
//
//  Created by MNN on 2018/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSqueeze_hpp
#define CPUSqueeze_hpp

#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUSqueeze : public Execution {
public:
    CPUSqueeze(Backend *b, const SqueezeParam *parameter);
    virtual ~CPUSqueeze() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const SqueezeParam *mParameter;
};
} // namespace MNN

#endif /* CPUSqueeze_hpp */
