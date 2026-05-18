//
//  QNNCast.hpp
//  MNN
//
//  Created by MNN on 2026/02/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNTOPKV2_HPP
#define MNN_QNNTOPKV2_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNTopKV2 : public QNNCommonExecution {
public:
    QNNTopKV2(Backend* backend, const Op* op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNTOPKV2_HPP
