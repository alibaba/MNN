//
//  CPUQuantizedConcat.hpp
//  MNN
//
//  Created by MNN on 2018/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedConcat_hpp
#define CPUQuantizedConcat_hpp

#include "Execution.hpp"

namespace MNN {

class CPUQuantizedConcat : public Execution {
public:
    CPUQuantizedConcat(Backend *backend, const Op *op);
    virtual ~CPUQuantizedConcat() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
    std::vector<int> mInputZeroPoint;
    std::vector<float> mInputScale;
    int mOutputZeroPoint;
    float mOutputScale;
};

} // namespace MNN
#endif /* CPUQuantizedConcat_hpp */
