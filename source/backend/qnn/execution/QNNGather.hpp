#ifndef MNN_QNNGATHER_HPP
#define MNN_QNNGATHER_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNGather : public QNNCommonExecution {
public:
    QNNGather(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    ErrorCode onEncodeNHWCScalar(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onEncodeNHWCTensor(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onEncodeNCHWScalar(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onEncodeNCHWTensor(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    void addNodeGather(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input0, const Qnn_Tensor_t & input1, const Qnn_Param_t & paramAxis, const Qnn_Tensor_t & output);
    void addNodeReshape(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input, const Qnn_Tensor_t & output);

private:
    int mInputDim;
    int mOutputDim;
    Tensor::DimensionType mDimType;
    int mRawAxis;
    Qnn_DataType_t mQnnDataType;
    bool mFlagScalarIndices;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNGATHER_HPP
