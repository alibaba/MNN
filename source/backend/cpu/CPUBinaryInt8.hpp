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
    std::shared_ptr<Execution> mActivationExe;
    std::vector<float> mInputQuant0;
    std::vector<float> mInputQuant1;
    std::vector<float> mOutputQuant;
};
} // namespace MNN
#endif /* CPUBinary_hpp */
