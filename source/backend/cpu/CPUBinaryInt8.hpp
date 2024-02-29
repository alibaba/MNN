//
//  CPUBinaryInt8.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBinaryInt8_hpp
#define CPUBinaryInt8_hpp

#include "core/Execution.hpp"
#include "backend/cpu/CPURelu.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/Int8FunctionsOpt.h"
namespace MNN {
class CPUBinaryInt8 : public Execution {
public:
    CPUBinaryInt8(Backend *b, MNNBinaryExecInt8 proc, int activationType) : Execution(b) {
        mProc = proc;
        mActivationType = activationType;
    }
    virtual ~CPUBinaryInt8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static MNNBinaryExecInt8 selectForInt8(int opType);
private:
    MNNBinaryExecInt8 mProc;
    int mNeedBroadcastIndex = -1;
    int mTotalSize;
    int mActivationType = 0;
    int mMinValue = -127;
    std::vector<ssize_t> mQuantScalesInt32; // input0 and input1
    std::vector<float> mQuantScalesFp32;  // input0, input1 and output
    std::vector<ssize_t> mInputZeros;
    std::vector<ssize_t> mOutputZeros;
    std::vector<float> mInputScales;
    std::vector<float> mOutputScales;
};
} // namespace MNN
#endif /* CPUBinary_hpp */
