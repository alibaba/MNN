//
//  QNNConvDepthwise.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCONVDEPTHWISE_HPP
#define MNN_QNNCONVDEPTHWISE_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {

class QNNConvDepthwise : public QNNCommonExecution {
public:
    QNNConvDepthwise(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    void createWeight(Qnn_DataType_t dataType, int oc, int kernelH, int kernelW);
    void createBias(Qnn_DataType_t dataType, int oc);
    void convertWeight(const float * src, float * dst, int oc, int kernelH, int kernelW);
};

} // end namespace QNN
} // end namespace MNN

#endif
