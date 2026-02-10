//
//  QNNConvolution.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCONVOLUTION_HPP
#define MNN_QNNCONVOLUTION_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNConvolution : public QNNCommonExecution {
public:
    QNNConvolution(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode onEncodeFpAIntBMatMul(Tensor * input, Tensor * output, int n, int h, int w, int ic, int oc);
    ErrorCode onEncodeQuantDequantConv(Tensor *input, Tensor *output, const int n, const int ic, const int oc);

private:
    template <typename T>
    void convertWeight(const T * src, T * dst, int oc, int ic, int kernelH, int kernelW) {
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        uint32_t srcOffset = w + kernelW * (h + kernelH * (i + ic * o));
                        uint32_t dstOffset = o + oc * (i + ic * (w + kernelW * h));
                        dst[dstOffset] = src[srcOffset];
                    }
                }
            }
        }
    }
    void isWeightQuantSupported(const Tensor *input, const int ic, const int oc);
    bool createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input, int oc, int ic, int kernelH, int kernelW, int group);
    void createBias(Qnn_DataType_t dataType, int oc, const Tensor *input, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon);
    std::vector<float> mScale;
    std::vector<Qnn_ScaleOffset_t> mScaleOffsetData;
    std::vector<Qnn_ScaleOffset_t> mBiasScaleOffsetData;
    std::vector<uint8_t> mBlockScale;
    Qnn_BlockwiseExpansion_t weightBlockwiseExpansionEncoding = QNN_BLOCKWISE_EXPANSION_INIT;
    float *mDequantAlpha = nullptr;
    int mBlockSize = 1;
    bool mWeightQuant = false;
    bool mIsMatMul = false;
    bool mIs1x1Conv = false;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
