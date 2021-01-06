//
//  CPUFloatToInt8.hpp
//  MNN
//
//  Created by MNN on 2019/6/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUFloatToInt8_hpp
#define CPUFloatToInt8_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUFloatToInt8 : public Execution {
public:
    CPUFloatToInt8(Backend *backend, const MNN::Op *param);
    virtual ~CPUFloatToInt8();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScales;
    int8_t mZeroPoint;
    int8_t mClampMin;
    int8_t mClampMax;
    int mClipBits;
    bool mSingle = false;
};

} // namespace MNN

#endif /* CPUFloatToInt8_hpp */
