//
//  QNNUtils.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNUTILS_HPP
#define MNN_QNNUTILS_HPP

#include "QnnInterface.h"
#include "QnnCommon.h"
#include "QnnLog.h"
#include "QnnTypes.h"
#include "HTP/QnnHtpDevice.h"
#include <MNN/HalideRuntime.h>
#include <map>
#include "core/TensorUtils.hpp"
#include <dlfcn.h>

#ifdef MNN_USE_ARMV82
// FP32 <--> FP16 Function
#include "backend/arm82/Arm82OptFunc.hpp"
#define FLOAT_TO_HALF MNNQuantizeFP16
#define HALF_TO_FLOAT MNNDequantizeFP16
#else
#include "half.hpp"
#define FLOAT_TO_HALF QnnFloatToHalf
#define HALF_TO_FLOAT QnnHalfToFloat
#endif // MNN_USE_ARMV82

#define CALL_QNN(apiCall)                                                       \
    do {                                                                        \
        int errorCode = ((apiCall) & 0xFFFF);                                   \
        if (errorCode != QNN_SUCCESS) {                                         \
            MNN_ERROR("Error in file %s, line %d: error code %d\n",             \
                    __FILE__, __LINE__, errorCode);                             \
            assert(errorCode == QNN_SUCCESS);                                   \
        }                                                                       \
    } while (0)

#define DEFAULT_QUANTIZE_PARAMS     (Qnn_QuantizeParams_t { \
                                        QNN_DEFINITION_UNDEFINED, \
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED, \
                                        {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}} \
                                    })

namespace MNN {
namespace QNN {

#ifndef MNN_USE_ARMV82

void QnnFloatToHalf(const float* src, int16_t* dst, size_t size);

void QnnHalfToFloat(const int16_t* src, float* dst, size_t size);

#endif

// the only symbol requiring dynamic loading
typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
extern QnnInterface_getProviders_t QnnInterface_getProviders;
bool loadQNNSymbol();

// op registration
extern void ___QNNActivationCreator__OpType_ReLU__();
extern void ___QNNActivationCreator__OpType_ReLU6__();
extern void ___QNNActivationCreator__OpType_Sigmoid__();
extern void ___QNNActivationCreator__OpType_ELU__();
extern void ___QNNArgmaxCreator__OpType_ArgMax__();
extern void ___QNNArgmaxCreator__OpType_ArgMin__();
extern void ___QNNBinaryCreator__OpType_BinaryOp__();
extern void ___QNNBinaryCreator__OpType_Eltwise__();
extern void ___QNNConcatCreator__OpType_Concat__();
extern void ___QNNConvDepthwiseCreator__OpType_ConvolutionDepthwise__();
extern void ___QNNConvolutionCreator__OpType_Convolution__();
extern void ___QNNFlattenCreator__OpType_Flatten__();
extern void ___QNNLayerNormCreator__OpType_LayerNorm__();
extern void ___QNNPaddingCreator__OpType_Padding__();
extern void ___QNNPoolCreator__OpType_Pooling__();
extern void ___QNNPoolCreator__OpType_Pooling3D__();
extern void ___QNNReduceCreator__OpType_Reduction__();
extern void ___QNNReshapeCreator__OpType_Reshape__();
extern void ___QNNReshapeCreator__OpType_Squeeze__();
extern void ___QNNReshapeCreator__OpType_Unsqueeze__();
extern void ___QNNReshapeCreator__OpType_ConvertTensor__();
extern void ___QNNScaleCreator__OpType_Scale__();
extern void ___QNNSoftmaxCreator__OpType_Softmax__();
extern void ___QNNStridedSliceCreator__OpType_StridedSlice__();
extern void ___QNNUnaryCreator__OpType_UnaryOp__();
void registerQNNOps();


extern Tensor::DimensionType gQnnTensorDimType;

extern const std::map<Qnn_DataType_t, uint32_t> gQnnTypeSize;

int getNHWCAxis(const int axis, const int dim, const Tensor::DimensionType type);

int getNCHWAxis(const int axis, const int dim, const Tensor::DimensionType type);

std::vector<uint32_t> getNHWCShape(const Tensor * tensor);

bool checkCapability(QNN_INTERFACE_VER_TYPE qnnInterface, QnnProperty_Key_t key);

void printNHWCShape(const Tensor * tensor);

} // end namespace QNN
} // end namespace MNN

#endif
