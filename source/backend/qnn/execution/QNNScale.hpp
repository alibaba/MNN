//
//  QNNScale.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNSCALE_HPP
#define MNN_QNNSCALE_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNScale : public QNNCommonExecution {
public:
    QNNScale(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    void mulWeight(Tensor * input);
    void addBias(Tensor * output);
};

} // end namespace QNN
} // end namespace MNN

#endif
