//
//  CPUPoolInt8.hpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPoolInt8_hpp
#define CPUPoolInt8_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

class CPUPoolInt8 : public Execution {
public:
    CPUPoolInt8(Backend *backend, const Pool *parameter);
    virtual ~CPUPoolInt8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Pool *mParameter;
    std::function<void(const Tensor *src, Tensor *dst)> mThreadFunction;
    // C16NHW16 buffer
    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
};

} // namespace MNN

#endif /* CPUPoolInt8_hpp */
