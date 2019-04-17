//
//  CPUUnary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnary_hpp
#define CPUUnary_hpp

#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUUnary : public Execution {
public:
    CPUUnary(Backend *b, UnaryOpOperation type);
    virtual ~CPUUnary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    UnaryOpOperation mType;
};
} // namespace MNN
#endif /* CPUUnary_hpp */
