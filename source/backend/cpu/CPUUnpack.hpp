//
//  CPUUnpack.hpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnpack_hpp
#define CPUUnpack_hpp

#include "Execution.hpp"

namespace MNN {
class CPUUnpack : public Execution {
public:
    CPUUnpack(Backend *backend, const Op *op, int axis);
    virtual ~CPUUnpack() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
};
} // namespace MNN
#endif /* CPUUnpack_hpp */
