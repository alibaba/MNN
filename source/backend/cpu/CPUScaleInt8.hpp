//
//  CPUScaleInt8.hpp
//  MNN
//
//  Created by MNN on 2023/05/04.
//

#ifndef CPUScaleInt8_hpp
#define CPUScaleInt8_hpp

#include <MNN/Tensor.hpp>
#include "core/Execution.hpp"

namespace MNN {
class CPUScaleInt8 : public Execution {
public:
    CPUScaleInt8(const Op *op, Backend *bn);
    virtual ~CPUScaleInt8();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScaleBias;
    std::vector<float>      mOutputQuantInfo;
    std::vector<float>      mInputQuantInfo;
    int32_t mShiftBits;
};

} // namespace MNN
#endif /* CPUScaleInt8_hpp */
