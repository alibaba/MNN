//
//  QNNUtils.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNUtils.hpp"

namespace MNN {
namespace QNN {

#ifndef MNN_USE_ARMV82

void QnnFloatToHalf(const float* src, int16_t* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ((half_float::half *)dst)[i] = (half_float::half)(src[i]);
    }
    return;
}

void QnnHalfToFloat(const int16_t* src, float* dst, size_t size) {
    const size_t batchSize = 8;
    std::vector<half_float::half> halfBatch(batchSize);

    for (size_t i = 0; i < size; i += batchSize) {
        size_t currentBatchSize = std::min(batchSize, size - i);

        ::memcpy(halfBatch.data(), &(src[i]), currentBatchSize * sizeof(int16_t));

        for (size_t j = 0; j < currentBatchSize; ++j) {
            dst[i + j] = static_cast<float>(halfBatch[j]);
        }
    }
}

#endif

QnnInterface_getProviders_t QnnInterface_getProviders = nullptr;

bool loadQNNSymbol() {
    void * qnnLibHandle = nullptr;
    qnnLibHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    const char * errorOpen = dlerror();
    if (!qnnLibHandle) {
        MNN_PRINT("MNN_QNN: Failed to open QNN libs. Ensure that the libs related to the QNN HTP backend is available in your environment. dlerror() returns %s.\n", errorOpen);
        return false;
    }

    QnnInterface_getProviders = (QnnInterface_getProviders_t)dlsym(qnnLibHandle, "QnnInterface_getProviders");
    const char * errorSym = dlerror();
    if (!QnnInterface_getProviders) {
        MNN_PRINT("MNN_QNN: Failed to load symbol <QnnInterface_getProviders>. dlerror returns %s.\n", errorSym);
        dlclose(qnnLibHandle);
        return false;
    }

    return true;
}

void registerQNNOps() {
    ___QNNActivationCreator__OpType_ReLU__();
    ___QNNActivationCreator__OpType_ReLU6__();
    ___QNNActivationCreator__OpType_Sigmoid__();
    ___QNNActivationCreator__OpType_ELU__();
    ___QNNArgmaxCreator__OpType_ArgMax__();
    ___QNNArgmaxCreator__OpType_ArgMin__();
    ___QNNBinaryCreator__OpType_BinaryOp__();
    ___QNNBinaryCreator__OpType_Eltwise__();
    ___QNNConcatCreator__OpType_Concat__();
    ___QNNConvDepthwiseCreator__OpType_ConvolutionDepthwise__();
    ___QNNConvolutionCreator__OpType_Convolution__();
    ___QNNFlattenCreator__OpType_Flatten__();
    ___QNNLayerNormCreator__OpType_LayerNorm__();
    ___QNNPaddingCreator__OpType_Padding__();
    ___QNNPoolCreator__OpType_Pooling__();
    ___QNNPoolCreator__OpType_Pooling3D__();
    ___QNNReduceCreator__OpType_Reduction__();
    ___QNNReshapeCreator__OpType_Reshape__();
    ___QNNReshapeCreator__OpType_Squeeze__();
    ___QNNReshapeCreator__OpType_Unsqueeze__();
    ___QNNReshapeCreator__OpType_ConvertTensor__();
    ___QNNScaleCreator__OpType_Scale__();
    ___QNNSoftmaxCreator__OpType_Softmax__();
    ___QNNStridedSliceCreator__OpType_StridedSlice__();
    ___QNNUnaryCreator__OpType_UnaryOp__();
}

Tensor::DimensionType gQnnTensorDimType = Tensor::TENSORFLOW;

const std::map<Qnn_DataType_t, uint32_t> gQnnTypeSize = {
    {QNN_DATATYPE_INT_8, 1},
    {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},
    {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},
    {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},
    {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},
    {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_FLOAT_64, 8},
    {QNN_DATATYPE_BOOL_8, 1},
    // {QNN_DATATYPE_SFIXED_POINT_4, 0.5},
    {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2},
    {QNN_DATATYPE_SFIXED_POINT_32, 4},
    // {QNN_DATATYPE_UFIXED_POINT_4, 0.5},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},
    {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4},
};

int getNHWCAxis(const int axis, const int dim, const Tensor::DimensionType type) {
    MNN_ASSERT(dim >= 1 && axis >= 0 && axis < dim);

    if (dim <= 2) {
        return axis;
    }

    std::vector<int> axisMap(dim);
    switch (type) {
        case Tensor::TENSORFLOW:
            return axis;
        case Tensor::CAFFE:
        case Tensor::CAFFE_C4:
            axisMap[0] = 0;
            axisMap[1] = dim - 1;
            for (int i = 2; i < dim; i++) {
                axisMap[i] = i - 1;
            }
            break;
        default:
            MNN_ERROR("MNN_QNN: Not supports Tensor::DimensionType.\n");
            break;
    }

    return axisMap[axis];
}

int getNCHWAxis(const int axis, const int dim, const Tensor::DimensionType type) {
    MNN_ASSERT(dim >= 1 && axis >= 0 && axis < dim);

    if (dim <= 2) {
        return axis;
    }

    std::vector<int> axisMap(dim);
    switch (type) {
        case Tensor::CAFFE:
        case Tensor::CAFFE_C4:
            return axis;
        case Tensor::TENSORFLOW:
            axisMap[0] = 0;
            axisMap[dim - 1] = 1;
            for (int i = 2; i < dim; i++) {
                axisMap[i - 1] = i;
            }
            break;
        default:
            MNN_ERROR("MNN_QNN: Not supports Tensor::DimensionType.\n");
            break;
    }

    return axisMap[axis];
}

std::vector<uint32_t> getNHWCShape(const Tensor * tensor) {
    std::vector<int> rawShape = tensor->shape();
    if (rawShape.empty()) {
        return {1};
    }
    std::vector<uint32_t> tensorShape(rawShape.size());
    for (int i = 0; i < tensorShape.size(); i++) {
        tensorShape[i] = (uint32_t) rawShape[i];
    }
    Tensor::DimensionType dimType = tensor->getDimensionType();
    int dim = rawShape.size();
    MNN_ASSERT(dim >= 1);

    if (dim <=2) {
        return tensorShape;
    }

    std::vector<uint32_t> NHWCShape(dim);
    switch (dimType) {
        case Tensor::TENSORFLOW:
            return tensorShape;
        case Tensor::CAFFE:
        case Tensor::CAFFE_C4:
            NHWCShape[0] = tensorShape[0];
            NHWCShape[dim - 1] = tensorShape[1];
            for (int i = 1; i < dim - 1; i++) {
                NHWCShape[i] = tensorShape[i + 1];
            }
            break;
        default:
            break;
    }
    return NHWCShape;
}

bool checkCapability(QNN_INTERFACE_VER_TYPE qnnInterface, QnnProperty_Key_t key) {
    Qnn_ErrorHandle_t errorCode;
    errorCode = qnnInterface.propertyHasCapability(key);
    if (errorCode == QNN_PROPERTY_SUPPORTED) {
        return true;
    } else {
        return false;
    }
}

void printNHWCShape(const Tensor * tensor) {
    std::vector<uint32_t> shape = getNHWCShape(tensor);
    MNN_PRINT("NHWC shape is:");
    for (int i = 0; i < shape.size(); i++) {
        MNN_PRINT(" %u", shape[i]);
    }
    MNN_PRINT(".\n");
}

} // end namespace QNN
} // end namespace MNN
