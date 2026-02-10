//
//  QNNQuant.hpp
//  MNN
//
//  Created by MNN on b'2025/05/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNQUANT_HPP
#define MNN_QNNQUANT_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNQuant : public QNNCommonExecution {
public:
    QNNQuant(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {};
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

class QNNDeQuant : public QNNCommonExecution {
public:
    QNNDeQuant(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {};
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNQUANT_HPP
