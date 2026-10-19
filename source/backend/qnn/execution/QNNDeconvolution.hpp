//
//  QNNDeconvolution.hpp
//  MNN
//

#ifndef MNN_QNNDECONVOLUTION_HPP
#define MNN_QNNDECONVOLUTION_HPP

#include <vector>

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNDeconvolution : public QNNCommonExecution {
public:
    QNNDeconvolution(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;

private:
    ErrorCode onEncodeLegacy(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs);
    std::vector<Qnn_ScaleOffset_t> mWeightScaleOffsets;
    std::vector<Qnn_ScaleOffset_t> mBiasScaleOffsets;
};

#endif
} // end namespace QNN
} // end namespace MNN

#endif
