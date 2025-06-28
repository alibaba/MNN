//
//  QNNBinary.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNBINARY_HPP
#define MNN_QNNBINARY_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {

class QNNBinary : public QNNCommonExecution {
public:
    QNNBinary(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // end namespace QNN
} // end namespace MNN

#endif // end MNN_BINARY_HPP

