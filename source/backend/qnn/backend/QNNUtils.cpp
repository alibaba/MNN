#if defined(__aarch64__)
#include <arm_neon.h>
#endif
//
//  QNNUtils.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNUtils.hpp"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE LibHandle;
#else
#include <dlfcn.h>
typedef void* LibHandle;
#endif

namespace MNN {
namespace QNN {

void QnnFloatToHalf(const float* src, int16_t* dst, size_t size) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= size; i += 4)
        vst1_s16(dst + i, vreinterpret_s16_f16(vcvt_f16_f32(vld1q_f32(src + i))));
#endif
    for (; i < size; ++i) {
        const half_float::half value(src[i]);
        std::memcpy(dst + i, &value, sizeof(value));
    }
}

void QnnHalfToFloat(const int16_t* src, float* dst, size_t size) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= size; i += 4)
        vst1q_f32(dst + i, vcvt_f32_f16(vreinterpret_f16_s16(vld1_s16(src + i))));
#endif
    for (; i < size; ++i) {
        half_float::half value;
        std::memcpy(&value, src + i, sizeof(value));
        dst[i] = static_cast<float>(value);
    }
}

QnnInterface_getProviders_t QnnInterface_getProviders = nullptr;
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
QnnSystemInterface_getProviders_t QnnSystemInterface_getProviders = nullptr;
#endif

static LibHandle gQnnLibHandle = nullptr;
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
static LibHandle gQnnSystemHandle = nullptr;
#endif
static QnnBackendKind gLoadedQnnBackend = QnnBackendKind::None;

static std::string& loadedQnnLibraryPath() {
    static auto* path = new std::string;
    return *path;
}

static const char* backendName(QnnBackendKind backend) {
    switch (backend) {
        case QnnBackendKind::Htp:
            return "HTP";
        case QnnBackendKind::Dsp:
            return "DSP V66";
        default:
            return "none";
    }
}

static std::string libraryPath(const std::string& directory,
                               const char* libraryName) {
    if (directory.empty()) return libraryName;
    if (directory.back() == '/') return directory + libraryName;
    return directory + "/" + libraryName;
}

QnnBackendKind getLoadedQNNBackend() {
    return gLoadedQnnBackend;
}

const char* getLoadedQNNBackendName() {
    return backendName(gLoadedQnnBackend);
}

bool isLoadedQNNLibraryCompatible(QnnBackendKind backend,
                                  const std::string& libraryDirectory) {
    if (gQnnLibHandle == nullptr) {
        return true;
    }
    if (backend != QnnBackendKind::None && backend != gLoadedQnnBackend) {
        return false;
    }
    if (libraryDirectory.empty()) {
        return true;
    }
    const char* basename = gLoadedQnnBackend == QnnBackendKind::Dsp
                               ? "libQnnDsp.so"
                               : "libQnnHtp.so";
    return loadedQnnLibraryPath() == libraryPath(libraryDirectory, basename);
}

static void closeQnnBackendLibrary() {
    QnnInterface_getProviders = nullptr;
    gLoadedQnnBackend = QnnBackendKind::None;
    loadedQnnLibraryPath().clear();
    if (gQnnLibHandle == nullptr) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(gQnnLibHandle);
#else
    dlclose(gQnnLibHandle);
#endif
    gQnnLibHandle = nullptr;
}

bool loadQNNSymbol(QnnBackendKind backend,
                   const std::string& libraryDirectory) {
    if (backend == QnnBackendKind::None) {
        return false;
    }
#ifndef MNN_QNN_DSP_RUNTIME
    if (backend == QnnBackendKind::Dsp) {
        return false;
    }
#endif
    if (gQnnLibHandle != nullptr && gLoadedQnnBackend == backend &&
        QnnInterface_getProviders != nullptr &&
        isLoadedQNNLibraryCompatible(backend, libraryDirectory)) {
        return true;
    }
    closeQnnBackendLibrary();

#ifdef _WIN32
    if (backend != QnnBackendKind::Htp) {
        MNN_PRINT("MNN_QNN: QNN DSP backend selection is not supported on Windows.\n");
        return false;
    }
    gQnnLibHandle = LoadLibraryA("QnnHtp.dll");
    if (!gQnnLibHandle) {
        MNN_PRINT("MNN_QNN: Failed to open QnnHtp.dll.\n");
        return false;
    }

    QnnInterface_getProviders = (QnnInterface_getProviders_t)GetProcAddress(
        gQnnLibHandle, "QnnInterface_getProviders");
    if (!QnnInterface_getProviders) {
        MNN_PRINT("MNN_QNN: Failed to load symbol <QnnInterface_getProviders>.\n");
        closeQnnBackendLibrary();
        return false;
    }
#else
    const char* libraryBasename =
        backend == QnnBackendKind::Dsp ? "libQnnDsp.so" : "libQnnHtp.so";
    const std::string libraryName =
        libraryPath(libraryDirectory, libraryBasename);
    dlerror();
    gQnnLibHandle = dlopen(libraryName.c_str(), RTLD_NOW | RTLD_LOCAL);
    const char * errorOpen = dlerror();
    if (!gQnnLibHandle) {
        MNN_PRINT("MNN_QNN: Failed to open %s for the %s backend: %s.\n",
                  libraryName.c_str(), backendName(backend),
                  errorOpen == nullptr ? "unknown dlopen error" : errorOpen);
        return false;
    }

    dlerror();
    QnnInterface_getProviders = (QnnInterface_getProviders_t)dlsym(
        gQnnLibHandle, "QnnInterface_getProviders");
    const char * errorSym = dlerror();
    if (!QnnInterface_getProviders) {
        MNN_PRINT("MNN_QNN: Failed to load symbol <QnnInterface_getProviders>. dlerror returns %s.\n", errorSym);
        closeQnnBackendLibrary();
        return false;
    }
#endif

    gLoadedQnnBackend = backend;
#ifdef _WIN32
    loadedQnnLibraryPath() = "QnnHtp.dll";
#else
    loadedQnnLibraryPath() = libraryPath(
        libraryDirectory,
        backend == QnnBackendKind::Dsp ? "libQnnDsp.so" : "libQnnHtp.so");
#endif
    MNN_PRINT("MNN_QNN: Loaded %s backend.\n", backendName(backend));
    return true;
}

bool loadQNNSystemSymbol(const std::string& libraryDirectory) {
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
    if (gQnnSystemHandle != nullptr &&
        QnnSystemInterface_getProviders != nullptr) {
        return true;
    }
#ifdef _WIN32
    gQnnSystemHandle = LoadLibraryA("QnnSystem.dll");
    if (gQnnSystemHandle == nullptr) {
        MNN_PRINT("MNN_QNN: Failed to open QnnSystem.dll.\n");
        return false;
    }
    QnnSystemInterface_getProviders =
        (QnnSystemInterface_getProviders_t)GetProcAddress(
            gQnnSystemHandle, "QnnSystemInterface_getProviders");
    if (QnnSystemInterface_getProviders == nullptr) {
        MNN_PRINT(
            "MNN_QNN: Failed to load symbol "
            "<QnnSystemInterface_getProviders>.\n");
        FreeLibrary(gQnnSystemHandle);
        gQnnSystemHandle = nullptr;
        return false;
    }
#else
    dlerror();
    const std::string libraryName =
        libraryPath(libraryDirectory, "libQnnSystem.so");
    gQnnSystemHandle =
        dlopen(libraryName.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (gQnnSystemHandle == nullptr) {
        const char* errorOpen = dlerror();
        MNN_PRINT("MNN_QNN: Failed to open %s: %s.\n",
                  libraryName.c_str(),
                  errorOpen == nullptr ? "unknown dlopen error" : errorOpen);
        return false;
    }
    dlerror();
    QnnSystemInterface_getProviders =
        (QnnSystemInterface_getProviders_t)dlsym(
            gQnnSystemHandle, "QnnSystemInterface_getProviders");
    if (QnnSystemInterface_getProviders == nullptr) {
        const char* errorSym = dlerror();
        MNN_PRINT(
            "MNN_QNN: Failed to load symbol "
            "<QnnSystemInterface_getProviders>: %s.\n",
            errorSym == nullptr ? "unknown dlsym error" : errorSym);
        dlclose(gQnnSystemHandle);
        gQnnSystemHandle = nullptr;
        return false;
    }
#endif
    return true;
#else
    MNN_PRINT("MNN_QNN: offline Context support was not compiled.\n");
    return false;
#endif
}

bool loadQNNSymbol() {
    // Automatic selection is process-scoped. Reuse the backend that already
    // owns live QNN handles instead of unloading it merely to probe the other
    // candidate again on a later Runtime registration.
    if (gQnnLibHandle != nullptr &&
        gLoadedQnnBackend != QnnBackendKind::None &&
        QnnInterface_getProviders != nullptr) {
        return true;
    }
    if (loadQNNSymbol(QnnBackendKind::Htp)) {
        return true;
    }
#ifdef MNN_QNN_DSP_RUNTIME
    return loadQNNSymbol(QnnBackendKind::Dsp);
#else
    return false;
#endif
}

static bool probeQNNBackendLibrary(QnnBackendKind backend) {
#ifdef _WIN32
    if (backend != QnnBackendKind::Htp) {
        return false;
    }
    LibHandle handle = LoadLibraryA("QnnHtp.dll");
    if (handle == nullptr) {
        return false;
    }
    const bool available = GetProcAddress(handle, "QnnInterface_getProviders") != nullptr;
    FreeLibrary(handle);
    return available;
#else
    const char* libraryName = backend == QnnBackendKind::Dsp ? "libQnnDsp.so" : "libQnnHtp.so";
    LibHandle handle = dlopen(libraryName, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return false;
    }
    const bool available = dlsym(handle, "QnnInterface_getProviders") != nullptr;
    dlclose(handle);
    return available;
#endif
}

bool isDefaultQNNRuntimeAvailable() {
    if (gQnnLibHandle != nullptr && QnnInterface_getProviders != nullptr) {
        return gLoadedQnnBackend == QnnBackendKind::Htp;
    }
    return probeQNNBackendLibrary(QnnBackendKind::Htp);
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

#ifdef ENABLE_QNN_ONLINE_FINALIZE

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
    ___QNNConcatCreator__OpType_Pack__();
    ___QNNConcatCreator__OpType_Unpack__();
    ___QNNConvDepthwiseCreator__OpType_ConvolutionDepthwise__();
    ___QNNConvolutionCreator__OpType_Convolution__();
    ___QNNDeconvolutionCreator__OpType_Deconvolution__();
    ___QNNFlattenCreator__OpType_Flatten__();
    ___QNNLayerNormCreator__OpType_LayerNorm__();
    ___QNNPaddingCreator__OpType_Padding__();
    ___QNNPoolCreator__OpType_Pooling__();
    ___QNNPoolCreator__OpType_Pooling3D__();
    ___QNNReduceCreator__OpType_Reduction__();
    ___QNNFlattenCreator__OpType_Reshape__();
    ___QNNFlattenCreator__OpType_Squeeze__();
    ___QNNFlattenCreator__OpType_Unsqueeze__();
    ___QNNReshapeCreator__OpType_ConvertTensor__();
    ___QNNScaleCreator__OpType_Scale__();
    ___QNNSoftmaxCreator__OpType_Softmax__();
    ___QNNStridedSliceCreator__OpType_StridedSlice__();
    ___QNNStridedSliceCreator__OpType_Slice__();
    ___QNNUnaryCreator__OpType_UnaryOp__();
    ___QNNCastCreator__OpType_Cast__();
    ___QNNPermuteCreator__OpType_Permute__();
    ___QNNPermuteCreator__OpType_Transpose__();
    ___QNNGatherCreator__OpType_GatherV2__();
    ___QNNGatherCreator__OpType_GatherElements__();
    ___QNNBroadcastToCreator__OpType_BroadcastTo__();
    ___QNNMatMulCreator__OpType_MatMul__();
    #ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    ___QNNAttentionCreator__OpType_Attention__();
    #endif
    ___QNNQuantCreator__OpType_FloatToInt8__();
    ___QNNDeQuantCreator__OpType_Int8ToFloat__();
    ___QNNTopKV2Creator__OpType_TopKV2__();
    ___QNNInterpCreator__OpType_Interp__();
}

Tensor::DimensionType gQnnTensorDimType = Tensor::TENSORFLOW;

const std::map<Qnn_DataType_t, uint32_t>& qnnTypeSizes() {
    static const std::map<Qnn_DataType_t, uint32_t> sizes = {
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
    return sizes;
}

const char gParamMarker[] = "PARAM";

std::vector<uint32_t> getNHWCShape(const Tensor * tensor) {
    std::vector<int> rawShape = tensor->shape();
    if (rawShape.empty()) {
        return {1};
    }
    std::vector<uint32_t> tensorShape(rawShape.size());
    for (int i = 0; i < tensorShape.size(); i++) {
        tensorShape[i] = (uint32_t) rawShape[i];
    }
    auto dataFormat = TensorUtils::getDescribe(tensor)->dimensionFormat;
    int dim = rawShape.size();
    MNN_ASSERT(dim >= 1);

    if (dim <=2) {
        return tensorShape;
    }

    std::vector<uint32_t> NHWCShape(dim);
    // only Tensor::CAFFE_C4 convert to Tensor::TENSORFLOW on qnn
    switch (dataFormat) {
        case MNN_DATA_FORMAT_NCHW:
        case MNN_DATA_FORMAT_NHWC:
            return tensorShape;
        case MNN_DATA_FORMAT_NC4HW4:
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

void printNHWCShape(const Tensor * tensor) {
    std::vector<uint32_t> shape = getNHWCShape(tensor);
    MNN_PRINT("NHWC shape is:");
    for (int i = 0; i < shape.size(); i++) {
        MNN_PRINT(" %u", shape[i]);
    }
    MNN_PRINT(".\n");
}
#endif
} // end namespace QNN
} // end namespace MNN
