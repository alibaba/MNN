//
//  QNNWrapper.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNTENSORWARPPER_HPP
#define MNN_QNNTENSORWARPPER_HPP

#include "QnnInterface.h"
#include "QNNUtils.hpp"
#include <vector>
#include <string>
#include "MNN/MNNDefine.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

// Wrap 'Qnn_Tensor_t' for the convenience of memory management.
class QNNTensorWrapper {
public:
    static std::shared_ptr<QNNTensorWrapper> create(const std::string & name, Qnn_TensorType_t type, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    static std::shared_ptr<QNNTensorWrapper> create(const std::string & name, Qnn_TensorType_t type, Qnn_DataType_t dataType, const std::vector<int> & dimensions, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    static std::shared_ptr<QNNTensorWrapper> createStaticTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, const void * buffer, Qnn_QuantizeParams_t quantizeParam = DEFAULT_QUANTIZE_PARAMS);
    static std::shared_ptr<QNNTensorWrapper> createStaticFloatTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, const float * buffer, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    static std::shared_ptr<QNNTensorWrapper> createStaticFloatTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<int> & dimensions, const float * buffer, Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
    QNNTensorWrapper(const std::string & name, Qnn_TensorType_t type, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, Qnn_QuantizeParams_t quantize);
    ~QNNTensorWrapper();
    Qnn_Tensor_t * getNativeTensor();
    const Qnn_Tensor_t * getNativeTensor() const;
    void * alloc(Tensor::DimensionType dimType = gQnnTensorDimType);
    std::shared_ptr<Tensor> getDataContainer();
    const std::vector<uint32_t> * getDimension();

private:
    std::string mName;
    std::vector<uint32_t> mDimensions;
    std::shared_ptr<Tensor> mDataContainer;
    Qnn_Tensor_t mQnnTensor;
    bool mIsAlloc = false;

friend class QNNParamTensorWrapper;
};

class QNNParamTensorWrapper {
public:
    static std::shared_ptr<QNNParamTensorWrapper> create(const std::string & paramName, const std::string & tensorName, Qnn_DataType_t dataType, const std::vector<int32_t> & dimensions);
    static std::shared_ptr<QNNParamTensorWrapper> create(const std::string & paramName, const std::string & tensorName, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions);
    QNNParamTensorWrapper(const std::string & paramName, const std::string & tensorName, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions);
    ~QNNParamTensorWrapper();
    void * alloc();
    Qnn_Param_t * getNativeParam();
    Qnn_Tensor_t * getNativeTensor();


private:
    std::string mParamName;
    std::string mTensorName;
    std::vector<uint32_t> mDimensions;
    Qnn_Param_t mQnnParam{};
};

class QNNParamScalarWrapper {
public:
    template<typename T>
    static std::shared_ptr<QNNParamScalarWrapper> create(const std::string & name, T value) {return std::make_shared<QNNParamScalarWrapper>(name, value);};
    QNNParamScalarWrapper(const std::string & name, bool value);
    QNNParamScalarWrapper(const std::string & name, uint32_t value);
    QNNParamScalarWrapper(const std::string & name, int value);
    QNNParamScalarWrapper(const std::string & name, float value);
    Qnn_Param_t * getNativeParam();

private:
    std::string mName;
    Qnn_Param_t mQnnParam{};
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
