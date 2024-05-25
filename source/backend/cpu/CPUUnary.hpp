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
    CPUUnary(Backend *b, MNNUnaryExecute proc, MNNUnaryExecuteInt8 procInt8, const Op* op);
    virtual ~CPUUnary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static MNNUnaryExecute selectForFloat(int type, int precision);
    static MNNUnaryExecuteInt8 selectForInt8(int type);
protected:
    MNNUnaryExecute mProc;
    MNNUnaryExecuteInt8 mProcInt8;
    std::vector<float> mInpScale;
    std::vector<float> mOupScale;
    std::vector<ssize_t> mInpZeroPoint;
    std::vector<ssize_t> mOupZeroPoint;
    std::vector<ssize_t> mMaxMinValue;
    std::vector<int8_t> mTableBuffer;
};
} // namespace MNN
#endif /* CPUUnary_hpp */
