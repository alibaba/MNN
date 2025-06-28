//
//  QNNUnary.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNUNARY_HPP
#define MNN_QNNUNARY_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNUnary : public QNNCommonExecution {
public:
    QNNUnary(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNUNARY_HPP
