//
//  CPUInt8ToFloat.hpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInt8ToFloat_hpp
#define CPUInt8ToFloat_hpp

#include "core/Execution.hpp"
#include <MNN/Tensor.hpp>

namespace MNN {

class CPUInt8ToFloat : public Execution {
public:
    CPUInt8ToFloat(Backend *backend, const MNN::Op *param);
    virtual ~CPUInt8ToFloat();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScales;

    bool mSingle = false;
    int8_t mZeroPoint;
};

} // namespace MNN

#endif /* CPUInt8ToFloat_hpp */
