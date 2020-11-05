//
//  TRTBinary.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTBinary_HPP
#define MNN_TRTBinary_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTDetectionPostProcess : public TRTCommonExecution {
public:
    TRTDetectionPostProcess(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTDetectionPostProcess() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
private:
    DetectionPostProcessParamT mParam;
};


} // namespace MNN

#endif // MNN_TRTBinary_HPP
