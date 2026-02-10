//
//  QNNCast.hpp
//  MNN
//
//  Created by MNN on 2025/05/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCAST_HPP
#define MNN_QNNCAST_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNCast : public QNNCommonExecution {
public:
    QNNCast(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNCAST_HPP
