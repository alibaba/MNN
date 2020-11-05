//
// Created by alibaba on 2019/9/11.
//

#ifndef MNN_TRTSoftmax_HPP
#define MNN_TRTSoftmax_HPP

#include "TRTCommonExecution.hpp"
#include "TRTBackend.hpp"

namespace MNN {

class TRTSoftmax : public TRTCommonExecution {
public:
    TRTSoftmax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTSoftmax() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
private:
    int mAxis;
};

} // namespace MNN

#endif // MNN_TRTSoftmax_HPP
