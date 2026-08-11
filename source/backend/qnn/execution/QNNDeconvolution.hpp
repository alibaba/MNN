//
//  QNNDeconvolution.hpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNDECONVOLUTION_HPP
#define MNN_QNNDECONVOLUTION_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNDeconvolution : public QNNCommonExecution {
public:
    QNNDeconvolution(Backend* backend, const Op* op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    // Convert weight from MNN's OIHW layout to QNN's HWIO layout
    void convertWeightOIHWtoHWIO(const float* src, float* dst, int oc, int ic, int kh, int kw);
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // MNN_QNNDECONVOLUTION_HPP