//
//  QNNSoftmax.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNSOFTMAX_HPP
#define MNN_QNNSOFTMAX_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {

class QNNSoftmax : public QNNCommonExecution {
public:
    QNNSoftmax(Backend *backend, const Op *op, int axis) : QNNCommonExecution(backend, op) {mAxis = axis;}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    ErrorCode onEncodePermute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    int mAxis;
};

} // end namespace QNN
} // end namespace MNN

#endif
