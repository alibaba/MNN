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
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNScale : public QNNCommonExecution {
public:
    QNNScale(Backend *backend, const Op *op);
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    void mulWeight(Tensor * input);
    void addBias(Tensor * output);
private:
    std::vector<float> mWeightData;
    std::vector<float> mBiasData;
    bool mNeedQuantDequant = false;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
