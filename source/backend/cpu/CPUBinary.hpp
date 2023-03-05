//
//  CPUBinary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBinary_hpp
#define CPUBinary_hpp

#include "core/Execution.hpp"
#include "backend/cpu/CPURelu.hpp"
#include "compute/CommonOptFunction.h"
namespace MNN {
class CPUBinary : public Execution {
public:
    CPUBinary(Backend *b, MNNBinaryExecute proc, int activationType) : Execution(b) {
        mProc = proc;
        mActivationType = activationType;
    }
    virtual ~CPUBinary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static MNNBinaryExecute selectForFloat(int opType);
    static MNNBinaryExecute selectForInt(int opType);
private:
    MNNBinaryExecute mProc;
    int mNeedBroadcastIndex = -1;
    int mTotalSize;
    int mActivationType = 0;
    std::shared_ptr<Execution> mActivationExe;
};
} // namespace MNN
#endif /* CPUBinary_hpp */
