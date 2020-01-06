//
//  CPUThreshold.hpp
//  MNN
//
//  Created by MNN on 2019/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUThreshold_hpp
#define CPUThreshold_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUThreshold : public Execution {
public:
    CPUThreshold(Backend *b, float threshold) : Execution(b), mThreshold(threshold) {
        // nothing to do
    }
    virtual ~CPUThreshold() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    float mThreshold;
};

} // namespace MNN

#endif /* CPUThreshold_hpp */
