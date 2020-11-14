//
//  CPUElu.hpp
//  MNN
//
//  Created by MNN on 2019/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUElu_hpp
#define CPUElu_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUElu : public Execution {
public:
    CPUElu(Backend *b, float alpha) : Execution(b), mAlpha(alpha) {
        // nothing to do
    }
    virtual ~CPUElu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    float mAlpha;
};

} // namespace MNN

#endif /* CPUElu_hpp */
