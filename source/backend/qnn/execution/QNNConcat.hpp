//
//  QNNConcat.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCONCAT_HPP
#define MNN_QNNCONCAT_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNConcat : public QNNCommonExecution {
public:
    QNNConcat(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNCONCAT_HPP


