//
//  CPUPriorbox.hpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPriorbox_hpp
#define CPUPriorbox_hpp

#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUPriorBox : public Execution {
public:
    CPUPriorBox(Backend *b, const MNN::Op *op);
    virtual ~CPUPriorBox() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const MNN::PriorBox *mParameter;
};

} // namespace MNN
#endif /* CPUPriorbox_hpp */
