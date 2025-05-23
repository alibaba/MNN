//
//  QNNLayerNorm.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNLAYERNORM_HPP
#define MNN_QNNLAYERNORM_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNLayerNorm : public QNNCommonExecution {
public:
    QNNLayerNorm(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // end namespace MNN
} // end namespace QNN

#endif // end MNN_QNNLAYERNORM_HPP
