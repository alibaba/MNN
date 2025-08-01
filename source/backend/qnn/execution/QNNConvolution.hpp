//
//  QNNConvolution.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCONVOLUTION_HPP
#define MNN_QNNCONVOLUTION_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {

class QNNConvolution : public QNNCommonExecution {
public:
    QNNConvolution(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode onEncodeQuant(Tensor * input, Tensor * output, int n, int h, int w, int ic, int oc, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon);

private:
    void createWeight(Qnn_DataType_t dataType, int oc, int ic, int kernelH, int kernelW, int group, const float * source);
    void createBias(Qnn_DataType_t dataType, int oc);
    void convertWeight(const float * src, float * dst, int oc, int ic, int kernelH, int kernelW);
};

} // end namespace QNN
} // end namespace MNN

#endif
