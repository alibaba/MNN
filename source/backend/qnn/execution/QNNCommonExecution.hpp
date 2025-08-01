//
//  QNNCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2025/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCOMMONEXECUTION_HPP
#define MNN_QNNCOMMONEXECUTION_HPP
#include "QNNBackend.hpp"
#include "core/Execution.hpp"

#define MNN_QNN_NOT_SUPPORT_SPECIAL_CASE \
    do { \
        MNN_PRINT("MNN_QNN: Some special cases of %s is not supported currently.\n", MNN::EnumNameOpType(mOp->type())); \
        return NOT_SUPPORT; \
    } while(0)

#define MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT \
do { \
    MNN_PRINT("MNN_QNN: Op %d parameters violated the constraints of the Qnn Htp backend.\n", (int) mOp->type()); \
    return NOT_SUPPORT; \
} while(0)

#define CLEAR_BEFORE_ADDING_NODE \
    mNodeType.clear();           \
    mInputs.clear();             \
    mParams.clear();             \
    mOutputs.clear();

namespace MNN {
namespace QNN {

class QNNCommonExecution : public Execution {
public:
    QNNCommonExecution(Backend *backend, const Op *op);

    virtual ~QNNCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    void setNodeName(const Op * op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    void createStaticTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, const void * buffer, Qnn_QuantizeParams_t quantizeParam = DEFAULT_QUANTIZE_PARAMS);
    void createStaticFloatTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, const float * buffer, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    void createStageTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<int> & dimensions, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    void createStageTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    void createParamTensor(const std::string & paramName, Qnn_DataType_t dataType, const std::vector<uint32_t> & dims, void * data, std::string postName = "");
    void createParamScalar(const std::string & name, bool data);
    void createParamScalar(const std::string & name, uint32_t data);
    void createParamScalar(const std::string & name, int data);
    void createParamScalar(const std::string & name, float data);

    void addNodeCommon(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    void addNodeCommonPermute(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input, const Qnn_Param_t & paramPerm, const Qnn_Tensor_t & output);
    void addNodeCommonReshape(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input, const Qnn_Tensor_t & output);

    void clean();

public:
    QnnBackend* mBackend;
    const Op* mOp;

    // mandatory params for QnnBackend::addNodeToGraph()
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;

    // wrappers for <mParams>, <mInputs>, <mOutputs>
    std::vector<std::shared_ptr<QNNTensorWrapper>> mTempTensorWrappers;
    std::vector<std::shared_ptr<QNNParamTensorWrapper>> mParamTensorWrappers;
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mParamScalarWrappers;
};

} // end namespace QNN
} // end namespace MNN

#endif /* QNNCommonExecution_hpp */
