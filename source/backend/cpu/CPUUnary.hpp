//
//  CPUUnary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnary_hpp
#define CPUUnary_hpp

#include "core/Execution.hpp"
#include "compute/CommonOptFunction.h"

namespace MNN {
class CPUUnary : public Execution {
public:
    CPUUnary(Backend *b, MNNUnaryExecute proc);
    virtual ~CPUUnary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static MNNUnaryExecute selectForFloat(int type, int precision);
protected:
    MNNUnaryExecute mProc;
};
} // namespace MNN
#endif /* CPUUnary_hpp */
