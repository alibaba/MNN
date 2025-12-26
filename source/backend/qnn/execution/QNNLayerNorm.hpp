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
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNLayerNorm : public QNNCommonExecution {
public:
    QNNLayerNorm(Backend *backend, const Op *op, Tensor * input);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    ErrorCode onEncodeNormWithPermute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
private:
    Qnn_DataType_t mQnnDataType;
    int mInputDim;
    Tensor::DimensionType mDimType;
    float mEpsilon;
    bool mUseRMSNorm;
    int mRealAxis;
    int mGammaBetaSize = 0;
    std::vector<float> mGammaData;
    std::vector<float> mBetaData;
};
#endif
} // end namespace MNN
} // end namespace QNN

#endif // end MNN_QNNLAYERNORM_HPP
