//
//  QNNConvDepthwise.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCONVDEPTHWISE_HPP
#define MNN_QNNCONVDEPTHWISE_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNConvDepthwise : public QNNCommonExecution {
public:
    QNNConvDepthwise(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode onEncodeQuantDequantDepthConv(Tensor *input, Tensor *output, const int n, const int ic, const int oc);
private:
template <typename T>
    void convertWeight(const T * src, T * dst, int oc, int kernelH, int kernelW) {
        for (int c = 0; c < oc; c++) {
            for (int h = 0; h < kernelH; h++) {
                for (int w = 0; w < kernelW; w++) {
                    int srcOffset = w + kernelW * (h + kernelH * c);
                    int dstOffset = c + oc * (w + kernelW * h);
                    dst[dstOffset] = src[srcOffset];
                }
            }
        }
    }
    void isWeightQuantSupported(const Tensor *input, const int oc);
    void createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input, int oc, int kernelH, int kernelW);
    std::vector<float> mScale;
    std::vector<Qnn_ScaleOffset_t> mScaleOffsetData;
    std::vector<Qnn_ScaleOffset_t> mBiasScaleOffsetData;
    std::vector<uint8_t> mBlockScale;
    float *mDequantAlpha = nullptr;
    bool mWeightQuant = false;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
