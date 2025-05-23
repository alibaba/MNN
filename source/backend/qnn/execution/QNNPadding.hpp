//
//  QNNPadding.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNPADDING_HPP
#define MNN_QNNPADDING_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNPadding : public QNNCommonExecution {
public:
    QNNPadding(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNPADDING_HPP
