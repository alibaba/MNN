//
//  QNNFlatten.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNFLATTEN_HPP
#define MNN_QNNFLATTEN_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNFlatten : public QNNCommonExecution {
public:
    QNNFlatten(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    void NHWC2NCHW(const std::vector<Tensor *> &inputs);
    void Reshape(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    void NCHW2NHWC(const std::vector<Tensor *> &outputs);
};

} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNFLATTEN_HPP
