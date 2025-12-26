//
//  QNNWrapper.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNWrapper.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::create(const std::string & name, Qnn_TensorType_t type, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, Qnn_QuantizeParams_t quantize) {
    return std::make_shared<QNNTensorWrapper>(name, type, dataType, dimensions, quantize);
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::create(const std::string & name, Qnn_TensorType_t type, Qnn_DataType_t dataType, const std::vector<int> & dimensions, Qnn_QuantizeParams_t quantize) {
    std::vector<uint32_t> vec(dimensions.size());
    for (int i = 0; i < dimensions.size(); i++) {
        vec[i] = (uint32_t)dimensions[i];
    }
    return std::make_shared<QNNTensorWrapper>(name, type, dataType, vec, quantize);
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::createStaticTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, const void * buffer, Qnn_QuantizeParams_t quantizeParam) {
    MNN_ASSERT(!name.empty() && !dimensions.empty() && buffer);
    MNN_ASSERT(dataType == QNN_DATATYPE_SFIXED_POINT_8 || dataType == QNN_DATATYPE_INT_32 || dataType == QNN_DATATYPE_UINT_32 || dataType == QNN_DATATYPE_SFIXED_POINT_32 || dataType == QNN_DATATYPE_UFIXED_POINT_8);

    std::shared_ptr<QNNTensorWrapper> tensorWrapper = QNNTensorWrapper::create(name, QNN_TENSOR_TYPE_STATIC, dataType, dimensions, quantizeParam);
    uint32_t numElement = 1;
    for (int i = 0; i < dimensions.size(); i++) {
        numElement *= dimensions[i];
    }
    void * dst = tensorWrapper->alloc();
    uint32_t dataSize = gQnnTypeSize.find(dataType)->second;
    ::memcpy(dst, buffer, dataSize * numElement);
    return tensorWrapper;
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::createStaticFloatTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, const float * buffer, Qnn_QuantizeParams_t quantize) {
    MNN_ASSERT(!name.empty() && !dimensions.empty() && buffer != nullptr);
    MNN_ASSERT(dataType == QNN_DATATYPE_FLOAT_16 || dataType == QNN_DATATYPE_FLOAT_32);
    std::shared_ptr<QNNTensorWrapper> tensorWrapper = QNNTensorWrapper::create(name, QNN_TENSOR_TYPE_STATIC, dataType, dimensions, quantize);
    uint32_t numElement = 1;
    for (int i = 0; i < dimensions.size(); i++) {
        numElement *= dimensions[i];
    }

    if (dataType == QNN_DATATYPE_FLOAT_32) {
        float * dst = (float *)tensorWrapper->alloc();
        ::memcpy(dst, buffer, sizeof(float) * numElement);
    } else {
        void * dst = tensorWrapper->alloc();
        FLOAT_TO_HALF(buffer, (int16_t *)dst, numElement);
    }

    return tensorWrapper;
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::createStaticFloatTensor(const std::string & name, Qnn_DataType_t dataType, const std::vector<int> & dimensions, const float * buffer, Qnn_QuantizeParams_t quantize) {
    std::vector<uint32_t> vec(dimensions.size());
    for (int i = 0; i < dimensions.size(); i++) {
        vec[i] = (uint32_t)dimensions[i];
    }
    return QNNTensorWrapper::createStaticFloatTensor(name, dataType, vec, buffer, quantize);
}


QNNTensorWrapper::QNNTensorWrapper(const std::string & name, Qnn_TensorType_t type, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions, Qnn_QuantizeParams_t quantize) {
    mName = name;
    mDimensions = dimensions;

    mQnnTensor.version = QNN_TENSOR_VERSION_1;

    Qnn_TensorV1_t v1;
    v1.name = mName.c_str();
    v1.type = type;
    v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    v1.dataType = dataType;
    v1.quantizeParams = quantize;
    v1.rank = mDimensions.size();
    v1.dimensions = mDimensions.data();
    v1.memType = QNN_TENSORMEMTYPE_RAW;
    v1.clientBuf.data = nullptr;
    v1.clientBuf.dataSize = 0;

    mQnnTensor.v1 = v1;
}

QNNTensorWrapper::~QNNTensorWrapper() {

}

Qnn_Tensor_t * QNNTensorWrapper::getNativeTensor() {
    return &mQnnTensor;
}

const Qnn_Tensor_t * QNNTensorWrapper::getNativeTensor() const {
    return &mQnnTensor;
}

std::shared_ptr<Tensor> QNNTensorWrapper::getDataContainer() {
    MNN_ASSERT(mDataContainer.get() != nullptr);
    return mDataContainer;
}

void * QNNTensorWrapper::alloc(Tensor::DimensionType dimType) {
    MNN_ASSERT(mIsAlloc == false); // Realloc is not allowed.
    MNN_ASSERT(mQnnTensor.v1.type == QNN_TENSOR_TYPE_APP_READ || mQnnTensor.v1.type == QNN_TENSOR_TYPE_APP_WRITE || mQnnTensor.v1.type == QNN_TENSOR_TYPE_STATIC);

    std::vector<int> dims(mDimensions.size());
    for (int i = 0; i < mDimensions.size(); i++) {
        dims[i] = (int)mDimensions[i];
    }

    MNN_ASSERT(mQnnTensor.v1.dataType == QNN_DATATYPE_FLOAT_32 || mQnnTensor.v1.dataType == QNN_DATATYPE_FLOAT_16 \
        || mQnnTensor.v1.dataType == QNN_DATATYPE_INT_32 || mQnnTensor.v1.dataType == QNN_DATATYPE_UINT_32 \
        || mQnnTensor.v1.dataType == QNN_DATATYPE_SFIXED_POINT_8 \
        || mQnnTensor.v1.dataType == QNN_DATATYPE_SFIXED_POINT_32 \
        || mQnnTensor.v1.dataType == QNN_DATATYPE_UFIXED_POINT_8);
    halide_type_t halideType;

    halideType.lanes = 1;
    switch (mQnnTensor.v1.dataType) {
        case QNN_DATATYPE_FLOAT_32:
            halideType.code = halide_type_float;
            halideType.bits = 32;
            break;
        case QNN_DATATYPE_FLOAT_16:
            halideType.code = halide_type_float;
            halideType.bits = 16;
            break;
        case QNN_DATATYPE_INT_32:
            halideType.code = halide_type_int;
            halideType.bits = 32;
            break;
        case QNN_DATATYPE_SFIXED_POINT_8:
            halideType.code = halide_type_int;
            halideType.bits = 8;
            break;
        case QNN_DATATYPE_SFIXED_POINT_32:
            halideType.code = halide_type_int;
            halideType.bits = 32;
            break;
        case QNN_DATATYPE_UFIXED_POINT_8:
            halideType.code = halide_type_int;
            halideType.bits = 8;
            break;
        default:
            break;
    }

    mDataContainer.reset(Tensor::create(dims, halideType, nullptr, dimType));

    mQnnTensor.v1.clientBuf.data = mDataContainer->host<void>();
    mQnnTensor.v1.clientBuf.dataSize = mDataContainer->usize();
    mIsAlloc = true;

    return mQnnTensor.v1.clientBuf.data;
}

const std::vector<uint32_t> * QNNTensorWrapper::getDimension() {
    return &mDimensions;
}

std::shared_ptr<QNNParamTensorWrapper> QNNParamTensorWrapper::create(const std::string & paramName, const std::string & tensorName, Qnn_DataType_t dataType, const std::vector<int> & dimensions) {
    std::vector<uint32_t> vec(dimensions.size());
    for (int i = 0; i < dimensions.size(); i++) {
        vec[i] = (uint32_t)dimensions[i];
    }
    return std::make_shared<QNNParamTensorWrapper>(paramName, tensorName, dataType, vec);
}

std::shared_ptr<QNNParamTensorWrapper> QNNParamTensorWrapper::create(const std::string & paramName, const std::string & tensorName, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions) {
    return std::make_shared<QNNParamTensorWrapper>(paramName, tensorName, dataType, dimensions);
}

QNNParamTensorWrapper::QNNParamTensorWrapper(const std::string & paramName, const std::string & tensorName, Qnn_DataType_t dataType, const std::vector<uint32_t> & dimensions) {
    MNN_ASSERT(dataType == QNN_DATATYPE_INT_32 || dataType == QNN_DATATYPE_UINT_32);
    mParamName = paramName;
    mTensorName = tensorName;
    mDimensions = dimensions;
    // Fix parameters.
    mQnnParam.paramType = QNN_PARAMTYPE_TENSOR;
    mQnnParam.tensorParam.version = QNN_TENSOR_VERSION_1;
    mQnnParam.tensorParam.v1.id = 0;
    mQnnParam.tensorParam.v1.type = QNN_TENSOR_TYPE_STATIC;
    mQnnParam.tensorParam.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    mQnnParam.tensorParam.v1.quantizeParams = DEFAULT_QUANTIZE_PARAMS;
    mQnnParam.tensorParam.v1.memType = QNN_TENSORMEMTYPE_RAW;
    // Custom parameters.
    mQnnParam.name = mParamName.c_str();
    mQnnParam.tensorParam.v1.name = mTensorName.c_str();
    mQnnParam.tensorParam.v1.dataType = dataType;
    mQnnParam.tensorParam.v1.rank = mDimensions.size();
    mQnnParam.tensorParam.v1.dimensions = mDimensions.data();
    mQnnParam.tensorParam.v1.clientBuf = {.data = nullptr,
                                          .dataSize = 0};
}

QNNParamTensorWrapper::~QNNParamTensorWrapper() {
    MNN_ASSERT(mQnnParam.tensorParam.v1.clientBuf.data != nullptr);
    free(mQnnParam.tensorParam.v1.clientBuf.data);
}

void * QNNParamTensorWrapper::alloc() {
    uint32_t dataSize = gQnnTypeSize.find(mQnnParam.tensorParam.v1.dataType)->second;
    for (int i = 0; i < mQnnParam.tensorParam.v1.rank; i++) {
        dataSize *= mQnnParam.tensorParam.v1.dimensions[i];
    }
    #ifdef QNN_VERBOSE
    MNN_PRINT("QNNParamTensorWrapper size: %d\n", dataSize);
    #endif
    mQnnParam.tensorParam.v1.clientBuf.data = malloc(dataSize);
    MNN_ASSERT(mQnnParam.tensorParam.v1.clientBuf.data != nullptr);
    mQnnParam.tensorParam.v1.clientBuf.dataSize = dataSize;
    return mQnnParam.tensorParam.v1.clientBuf.data;
}

Qnn_Param_t * QNNParamTensorWrapper::getNativeParam() {
    return &(mQnnParam);
}

Qnn_Tensor_t * QNNParamTensorWrapper::getNativeTensor() {
    return &(mQnnParam.tensorParam);
}



QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string & name, bool value) {
    mName = name;
    mQnnParam.paramType = QNN_PARAMTYPE_SCALAR;
    mQnnParam.name        = mName.c_str();
    mQnnParam.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    mQnnParam.scalarParam.bool8Value = static_cast<uint8_t>(value);
}

QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string & name, uint32_t value) {
    mName = name;
    mQnnParam.paramType = QNN_PARAMTYPE_SCALAR;
    mQnnParam.name        = mName.c_str();
    mQnnParam.scalarParam.dataType = QNN_DATATYPE_UINT_32;
    mQnnParam.scalarParam.uint32Value = value;
}

QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string & name, int value) {
    mName = name;
    mQnnParam.paramType = QNN_PARAMTYPE_SCALAR;
    mQnnParam.name        = mName.c_str();
    mQnnParam.scalarParam.dataType = QNN_DATATYPE_INT_32;
    mQnnParam.scalarParam.uint32Value = value;
}

QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string & name, float value) {
    mName = name;
    mQnnParam.paramType = QNN_PARAMTYPE_SCALAR;
    mQnnParam.name        = mName.c_str();
    mQnnParam.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    mQnnParam.scalarParam.floatValue = value;
}

Qnn_Param_t * QNNParamScalarWrapper::getNativeParam() {
    return &(mQnnParam);
}
#endif
} // end namespace QNN
} // end namespace MNN
