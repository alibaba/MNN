/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 */
/* MediaTek Inc. (C) 2019. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY ACKNOWLEDGES
 * THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES
 * CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK
 * SOFTWARE RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER WILL BE,
 * AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT ISSUE,
 * OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER TO
 * MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek Software")
 * have been modified by MediaTek Inc. All revisions are subject to any receiver's
 * applicable license agreements with MediaTek Inc.
 */

#ifndef __MTK_NEUROPILOT_TFLITE_SHIM_H__
#define __MTK_NEUROPILOT_TFLITE_SHIM_H__

#include <dlfcn.h>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif

#define TFLITE_TENSOR_MAX_DIMENSTIONS 4

#ifdef __ANDROID__
#define TFLITE_LOG_D(format, ...) \
    __android_log_print(ANDROID_LOG_DEBUG, "NeuroPilotTFLiteShim", format "\n", ##__VA_ARGS__);
#else
#define TFLITE_LOG_D(format, ...) \
    printf(format "\n", ##__VA_ARGS__)
#endif

#ifdef __ANDROID__
#define TFLITE_LOG_E(format, ...) \
    __android_log_print(ANDROID_LOG_ERROR, "NeuroPilotTFLiteShim", format "\n", ##__VA_ARGS__);
#else
#define TFLITE_LOG_D(format, ...) \
    printf(format "\n", ##__VA_ARGS__)
#endif

#define LOAD_TFLITE_FUNCTION(name) \
    static name##_fn fn = reinterpret_cast<name##_fn>(loadTFLiteFunction(#name));

#define EXECUTE_TFLITE_FUNCTION(...) \
    if (fn != nullptr) {             \
        fn(__VA_ARGS__);             \
    }

#define EXECUTE_TFLITE_FUNCTION_RETURN_INT(...) \
    return fn != nullptr ? fn(__VA_ARGS__) : ANEURALNETWORKS_BAD_STATE;

#define EXECUTE_TFLITE_FUNCTION_RETURN_BOOL(...) return fn != nullptr ? fn(__VA_ARGS__) : false;

#define EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(...) \
    return fn != nullptr ? fn(__VA_ARGS__) : nullptr;

/************************************************************************************************/

typedef struct ANeuralNetworksTFLite ANeuralNetworksTFLite;
typedef struct NeuronModel NeuronModel;
typedef struct ANeuralNetworksTFLiteOptions ANeuralNetworksTFLiteOptions;
typedef struct ANeuralNetworksTFLiteTensor ANeuralNetworksTFLiteTensor;
typedef struct TfLiteContext TfLiteContext;

typedef enum {
    kUndefined = -1,
    kLowPower = 0,
    kFastSingleAnswer = 1,
    kSustainedSpeed = 2,
} ExecutionPreference;

typedef enum { kTfLiteOk = 0, kTfLiteError = 1 } TfLiteStatus;
typedef enum {
    TFLITE_BUFFER_TYPE_INPUT = 0,
    TFLITE_BUFFER_TYPE_OUTPUT = 1,
} NpTFLiteBufferType;

typedef uint32_t TFLiteBufferType;

typedef enum {
    TFLITE_TENSOR_TYPE_NONE = 0,
    TFLITE_TENSOR_TYPE_FLOAT = 1,
    TFLITE_TENSOR_TYPE_UINT8 = 2,
    TFLITE_TENSOR_TYPE_INT32 = 3,
    TFLITE_TENSOR_TYPE_INT64 = 4,
    TFLITE_TENSOR_TYPE_STRING = 5,
    TFLITE_TENSOR_TYPE_BOOL = 6,
    TFLITE_TENSOR_TYPE_INT16 = 7,
    TFLITE_TENSOR_TYPE_COMPLEX64 = 8,
    TFLITE_TENSOR_TYPE_INT8 = 9,
    TFLITE_TENSOR_TYPE_FLOAT16 = 10,
} NpTFLiteTensorType;

typedef uint32_t TFLiteTensorType;

typedef enum {
    NP_INFERENCE_TYPE_NONE = 0,
    NP_INFERENCE_TYPE_QNAUT = 1,
    NP_INFERENCE_TYPE_FLOAT = 2,
} NpInferenceType;

typedef uint32_t InferenceType;

typedef enum {
    // Use CPU to inference the model
    NP_ACCELERATION_CPU = 0,
    // Turns on Android NNAPI for hardware acceleration when it is available.
    NP_ACCELERATION_NNAPI = 1,
    // Use Neuron Delegate
    NP_ACCELERATION_NEURON = 2,
    // Use TFLITE GPU delegate
    NP_ACCELERATION_GPU = 9999,
} NpAccelerationMode;

typedef uint32_t AccelerationMode;

typedef struct TfLiteNode TfLiteNode;
typedef struct {
    int size;
    int data[];
} TfLiteIntArray;

typedef struct {
    int size;
    float data[];
} TfLiteFloatArray;

// SupportedQuantizationTypes.
typedef enum {
  // No quantization.
  TFLITE_NO_QUANTIZATION = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to TfLiteAffineQuantization.
  TFLITE_AFFINE_QUANTIZATION = 1,
} TfLiteQuantizationType;

// Structure specifying the quantization used by the tensor, if-any.
typedef struct {
  // The type of quantization held by params.
  TfLiteQuantizationType type;
  // Holds an optional reference to a quantization param structure. The actual
  // type depends on the value of the `type` field (see the comment there for
  // the values and corresponding types).
  void* params;
} TfLiteQuantization;

typedef struct {
  TfLiteFloatArray* scale;
  TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;

typedef struct {
    // The data type specification for data stored in `data`. This affects
    // what member of `data` union should be used.
    TFLiteTensorType type;
    // Tensor shapes
    int dimsSize;
    int dims[TFLITE_TENSOR_MAX_DIMENSTIONS];
    // Data pointer. The appropriate type should be used for a typed
    // tensor based on `type`.
    // The memory pointed by this data pointer is managed by ANeuralNetworksTFLite instance.
    // Caller should not try to free this pointer.
    void* buffer;

    // Correct the error naming from TFLiteTensor, this is actual buffer size in byte.
    size_t bufferSize;
} TFLiteTensorExt;

typedef struct {
    const char* op_name;
    const char* target_name;
    const char* vendor_name;
    void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
    void (*free)(TfLiteContext* context, void* buffer);
    TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
    TfLiteStatus (*add_params)(void*, ANeuralNetworksModel*, std::vector<uint32_t>&, uint32_t&);
} TFLiteCustomOpExt;

/*************************************************************************************************/
typedef int (*ANeuroPilotTFLiteOptions_create_fn)(ANeuralNetworksTFLiteOptions** options);

typedef int (*ANeuroPilotTFLiteOptions_free_fn)(ANeuralNetworksTFLiteOptions* options);

typedef int (*ANeuroPilotTFLiteOptions_setLowLatency_fn)(ANeuralNetworksTFLiteOptions* options,
                                                         bool enableLowLatency);

typedef int (*ANeuroPilotTFLiteOptions_setDeepFusion_fn)(ANeuralNetworksTFLiteOptions* options,
                                                         bool enableDeepDusion);

typedef int (*ANeuroPilotTFLiteOptions_setBatchProcessing_fn)(ANeuralNetworksTFLiteOptions* options,
                                                              bool enableBatchProcessing);

typedef int (*ANeuroPilotTFLiteOptions_setWarmupRuns_fn)(ANeuralNetworksTFLiteOptions* options,
                                                         uint32_t warmupRuns);

typedef int (*ANeuroPilotTFLiteOptions_setBoostHint_fn)(ANeuralNetworksTFLiteOptions* options,
                                                        uint8_t boostValue);

typedef int (*ANeuroPilotTFLiteOptions_setBoostDuration_fn)(ANeuralNetworksTFLiteOptions* options,
                                                        uint32_t boostDuration);

typedef int (*ANeuroPilotTFLiteOptions_setUseAhwb_fn)(ANeuralNetworksTFLiteOptions* options,
                                                        bool use_ahwb);
typedef int (*ANeuroPilotTFLiteOptions_setAllowExtremePerformance_fn)(
    ANeuralNetworksTFLiteOptions* options, bool allow, uint32_t duration);

typedef int (*ANeuroPilotTFLiteOptions_setAllowFp16PrecisionForFp32_fn)(
    ANeuralNetworksTFLiteOptions* options, bool allow);

typedef int (*ANeuroPilotTFLiteOptions_resizeInputTensor_fn)(ANeuralNetworksTFLiteOptions* options,
                                                             int32_t index, const int* dims,
                                                             int32_t dimsSize);

typedef int (*ANeuroPilotTFLiteOptions_setAccelerationMode_fn)(
    ANeuralNetworksTFLiteOptions* options, AccelerationMode mode);

typedef int (*ANeuroPilotTFLiteOptions_setEncryptionLevel_fn)(ANeuralNetworksTFLiteOptions* options,
                                                              int encryption_level);

typedef int (*ANeuroPilotTFLiteOptions_setCacheDir_fn)(ANeuralNetworksTFLiteOptions* options,
                                                       const char* cache_dir);

typedef int (*ANeuroPilotTFLiteOptions_setPreference_fn)(ANeuralNetworksTFLiteOptions* options,
                                                         int execution_preference);

typedef int (*ANeuroPilotTFLiteOptions_setDisallowNnApiCpu_fn)(
    ANeuralNetworksTFLiteOptions* options, bool disallow_nnapi_cpu);

typedef int (*ANeuroPilotTFLiteOptions_setCacheableIonBuffer_fn)(
    ANeuralNetworksTFLiteOptions* options, bool cacheable_ion_buffer);

typedef int (*ANeuroPilotTFLiteOptions_setUseIon_fn)(
    ANeuralNetworksTFLiteOptions* options, bool use_ion);

typedef int (*ANeuroPilotTFLiteOptions_setNoSupportedOperationCheck_fn)(
    ANeuralNetworksTFLiteOptions* options, bool no_supported_operation_check);

typedef int (*ANeuroPilotTFLiteOptions_setAcceleratorName_fn)(ANeuralNetworksTFLiteOptions* options,
                                                              const char* accelerator_name);

typedef int (*ANeuroPilotTFLiteOptions_setExecutionPriority_fn)(
    ANeuralNetworksTFLiteOptions* options, int execution_priority);

typedef int (*ANeuroPilotTFLiteOptions_setMaxCompilationTimeout_fn)(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_compilation_timeout_duration_ns);

typedef int (*ANeuroPilotTFLiteOptions_setMaxNumberDelegatedPartitions_fn)(
    ANeuralNetworksTFLiteOptions* options, uint32_t max_number_delegated_partitions);

typedef int (*ANeuroPilotTFLiteOptions_setMaxExecutionTimeout_fn)(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_timeout_duration_ns);

typedef int (*ANeuroPilotTFLiteOptions_setMaxExecutionLoopTimeout_fn)(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_loop_timeout_duration_ns);

typedef int (*ANeuroPilotTFLite_setBufferHandle_fn)(ANeuralNetworksTFLite* tflite,
        void** memory_data, TFLiteBufferType btype, int index, bool cacheable, int buffer_size);

typedef int (*ANeuroPilotTFLite_setAhwb_fn)(ANeuralNetworksTFLite* tflite,
        AHardwareBuffer* buffer, TFLiteBufferType btype, int index);

typedef int (*ANeuroPilotTFLiteOptions_setCompileOptionByString_fn) (ANeuralNetworksTFLiteOptions* options,
        const char* compileOptions);

typedef int (*ANeuroPilotTFLiteOptions_setComplieExtensionAttribute_fn) (
        ANeuralNetworksTFLiteOptions* options, const char* complieExtensionAttribute);

typedef int (*ANeuroPilotTFLiteOptions_setGpuExecutionPreference_fn)(ANeuralNetworksTFLiteOptions* options,
        int execution_preference);

typedef int (*ANeuroPilotTFLiteOptions_setGpuExecutionPriority_fn)(ANeuralNetworksTFLiteOptions* options,
        int priority_index, int priority_setting);

typedef int (*ANeuroPilotTFLite_create_fn)(ANeuralNetworksTFLite** tflite, const char* modelPath);

typedef int (*ANeuroPilotTFLite_createAdv_fn)(ANeuralNetworksTFLite** tflite, const char* modelPath,
                                              ANeuralNetworksTFLiteOptions* options);

typedef int (*ANeuroPilotTFLite_createWithBuffer_fn)(ANeuralNetworksTFLite** tflite,
                                                     const char* buffer, size_t bufferSize);
typedef int (*ANeuroPilotTFLite_createNeuronModelWithBuffer_fn)(NeuronModel** neuron_model,
                                                              const char* buffer,
                                                              const size_t bufferSize,
                                                              uint32_t* neuron_input_index,
                                                              uint32_t* neuron_output_index,
                                                              uint32_t* current_neuron_index);

typedef int (*ANeuroPilotTFLite_createAdvWithBuffer_fn)(ANeuralNetworksTFLite** tflite,
                                                        const char* buffer, size_t bufferSize,
                                                        ANeuralNetworksTFLiteOptions* options);

typedef int (*ANeuroPilotTFLite_createCustom_fn)(
    ANeuralNetworksTFLite** tflite, const char* modelPath,
    const std::vector<TFLiteCustomOpExt>& customOperations);

typedef int (*ANeuroPilotTFLite_createAdvCustom_fn)(
    ANeuralNetworksTFLite** tflite, const char* modelPath,
    const std::vector<TFLiteCustomOpExt>& customOperations, ANeuralNetworksTFLiteOptions* options);

typedef int (*ANeuroPilotTFLite_createCustomWithBuffer_fn)(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    const std::vector<TFLiteCustomOpExt>& customOperations);

typedef int (*ANeuroPilotTFLite_createAdvCustomWithBuffer_fn)(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    const std::vector<TFLiteCustomOpExt>& customOperations, ANeuralNetworksTFLiteOptions* options);

typedef int (*ANeuroPilotTFLite_getTensorCount_fn)(ANeuralNetworksTFLite* tflite,
                                                   TFLiteBufferType btype, int32_t* count);

typedef int (*ANeuroPilotTFLite_getTensorRank_fn)(ANeuralNetworksTFLite* tflite,
                                                  TFLiteBufferType btype, int index, int* rank);

typedef int (*ANeuroPilotTFLite_getTensorDimensions_fn)(ANeuralNetworksTFLite* tflite,
                                                        TFLiteBufferType btype, int index,
                                                        int* dimensions);

typedef int (*ANeuroPilotTFLite_getTensorByteSize_fn)(ANeuralNetworksTFLite* tflite,
                                                      TFLiteBufferType btype, int index,
                                                      size_t* size);

typedef int (*ANeuroPilotTFLite_getTensorType_fn)(ANeuralNetworksTFLite* tflite,
                                                  TFLiteBufferType btype, int index,
                                                  TFLiteTensorType* ttype);

typedef int (*ANeuroPilotTFLite_getTensorQuantizeParams_fn)(ANeuralNetworksTFLite* tflite,
                                                            TFLiteBufferType btype, int index,
                                                            TfLiteQuantization* quantization);

typedef int (*ANeuroPilotTFLite_setTensorBuffer_fn)(ANeuralNetworksTFLite* tflite, int index,
                                                    char* data);

typedef int (*ANeuroPilotTFLite_setInputTensorData_fn)(ANeuralNetworksTFLite* tflite, int index,
                                                       const void* data, size_t size);

typedef int (*ANeuroPilotTFLite_getOutputTensorData_fn)(ANeuralNetworksTFLite* tflite, int index,
                                                        void* data, size_t size);

typedef int (*ANeuroPilotTFLite_getDequantizedOutputByIndex_fn)(ANeuralNetworksTFLite* tflite,
                                                                void* buffer, size_t bufferByteSize,
                                                                int tensorIndex);

typedef int (*ANeuroPilotTFLite_invoke_fn)(ANeuralNetworksTFLite* tflite);

typedef int (*ANeuroPilotTFLite_free_fn)(ANeuralNetworksTFLite* tflite);

typedef int (*ANeuroPilotTFLite_setAllowFp16PrecisionForFp32_fn)(ANeuralNetworksTFLite* tflite,
                                                                 bool allow);

typedef int (*ANeuroPilot_getInferencePreference_fn)(void);

typedef int (*ANeuroPilotTFLiteCustomOp_getIntAttribute_fn)(const char* buffer, size_t length,
                                                            const char* attr, int32_t* outValue);

typedef int (*ANeuroPilotTFLiteCustomOp_getFloatAttribute_fn)(const char* buffer, size_t length,
                                                              const char* attr, float* outValue);

typedef void* (*ANeuroPilotTFLiteCustomOp_getUserData_fn)(TfLiteNode* node);

typedef int (*ANeuroPilotTFLiteCustomOp_getInput_fn)(TfLiteContext* context, TfLiteNode* node,
                                                     int index, TFLiteTensorExt* tfliteTensor);

typedef int (*ANeuroPilotTFLiteCustomOp_getOutput_fn)(TfLiteContext* context, TfLiteNode* node,
                                                      int index, TFLiteTensorExt* tfliteTensor);

typedef int (*ANeuroPilotTFLiteCustomOp_resizeOutput_fn)(TfLiteContext* context, TfLiteNode* node,
                                                         int index, TfLiteIntArray* new_size);

typedef TfLiteIntArray* (*ANeuroPilotTFLite_createIntArray_fn)(int size);

typedef int (*ANeuroPilotTFLite_freeIntArray_fn)(TfLiteIntArray* v);

/*************************************************************************************************/
inline int32_t GetAndroidSdkVersion() {
#ifdef __ANDROID__
    const char* sdkProp = "ro.build.version.sdk";
    char sdkVersion[PROP_VALUE_MAX];
    int length = __system_property_get(sdkProp, sdkVersion);
    if (length != 0) {
        int32_t result = 0;
        for (int i = 0; i < length; ++i) {
            int digit = sdkVersion[i] - '0';
            if (digit < 0 || digit > 9) {
                // Non-numeric SDK version, assume it's higher than expected;
                return 0xffff;
            }
            result = result * 10 + digit;
        }
        // TODO(levp): remove once SDK gets updated to 29th level
        // Upgrade SDK version for pre-release Q to be able to test functionality
        // available from SDK level 29.
        if (result == 28) {
            char versionCodename[PROP_VALUE_MAX];
            const char* versionCodenameProp = "ro.build.version.codename";
            length = __system_property_get(versionCodenameProp, versionCodename);
            if (length != 0) {
                if (versionCodename[0] == 'Q') {
                    return 29;
                }
            }
        }
        return result;
    }
#else
    return 32; // for non-Android platform set to version T.
#endif // _ANDROID__
    return 0;
}
// For add-on
static void* sTFLiteHandle;
inline void* loadTFLiteLibrary(const char* name) {
    sTFLiteHandle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (sTFLiteHandle == nullptr) {
        TFLITE_LOG_E("TFLite error: unable to open library %s", name);
    } else {
        TFLITE_LOG_D("TFLite : open library %s", name);
    }
    return sTFLiteHandle;
}

inline void* getTFLiteLibraryHandle() {
    if (sTFLiteHandle == nullptr) {
        // Load library for platform level development
        sTFLiteHandle = loadTFLiteLibrary("libtflite_mtk.mtk.so");
    }
    if (sTFLiteHandle == nullptr) {
        // Load library for platform level development
        sTFLiteHandle = loadTFLiteLibrary("libtflite_mtk.so");
    }
    int32_t sdk_version = GetAndroidSdkVersion();
    if (sdk_version < 30) {
        if (sTFLiteHandle == nullptr) {
            // Load library for APK JNI level development
            sTFLiteHandle = loadTFLiteLibrary("libtflite_mtk_static.so");
        }
    } else {
        if (sTFLiteHandle == nullptr) {
            // Load library for APK JNI level development
            sTFLiteHandle = loadTFLiteLibrary("libtflite_static_mtk.so");
        }
    }
    return sTFLiteHandle;
}

inline void* loadTFLiteFunction(const char* name) {
    void* fn = nullptr;
    if (getTFLiteLibraryHandle() != nullptr) {
        fn = dlsym(getTFLiteLibraryHandle(), name);
    }

    if (fn == nullptr) {
        TFLITE_LOG_E("TFLite error: unable to open function %s", name);
    }

    return fn;
}

/**
 * Create an {@link ANeuralNetworksTFLiteOptions} with default options.
 *
 * <p>{@link ANeuroPilotTFLiteOptionWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuralNetworksTFLiteOptions_create(ANeuralNetworksTFLiteOptions** options) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_create);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options);
}

/**
 * Specifies whether {@link ANeuralNetworksTFLiteOptions} is allowed to be calculated
 * with range and/or precision as low as that of the IEEE 754 16-bit
 * floating-point format.
 * This function is only used with float model.
 * A float model is calculated with FP16 precision by default.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param allow True to allow FP16 precision if possible.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(
    ANeuralNetworksTFLiteOptions* options, bool allow) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setAllowFp16PrecisionForFp32);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, allow);
}

/**
 * Create a copy of an array passed as `src`.
 * Developers are expected to free memory with ANeuroPilotTFLiteWrapper_freeIntArray.
 *
 * @param size The array size to be created.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline TfLiteIntArray* ANeuroPilotTFLiteWrapper_createIntArray(int size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createIntArray);
    EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(size);
}

/**
 * Free memory of array `v`.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_freeIntArray(TfLiteIntArray* v) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_freeIntArray);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(v);
}

/**
 * Change the dimensionality of a given input tensor.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param index The index of the input tensor.
 * @param dims List of the dimensions.
 * @param dimsSize Number of the dimensions.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_resizeInputTensor(ANeuralNetworksTFLiteOptions* options,
                                                          int32_t index, const int* dims,
                                                          int32_t dimsSize) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_resizeInputTensor);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, index, dims, dimsSize);
}

/**
 * Set preferred acceleration mode.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param mode Refer to {@link NpAccelerationMode} enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setAccelerationMode(ANeuralNetworksTFLiteOptions* options,
                                                            AccelerationMode mode) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setAccelerationMode);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, mode);
}

/**
 * Set compilation cache directory.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param user define cache directory.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setCacheDir(ANeuralNetworksTFLiteOptions* options,
                                                    const char* cache_dir) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setCacheDir);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, cache_dir);
}

/**
 * Set Execution Preference.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param execution preference refer to {@link ExecutionPreference} enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setPreference(ANeuralNetworksTFLiteOptions* options,
                                                      int execution_preference) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setPreference);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, execution_preference);
}

/**
 * Set Disallow NnApi Cpu.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param disallow nnapi cpu.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setDisallowNnApiCpu(ANeuralNetworksTFLiteOptions* options,
                                                            bool disallow_nnapi_cpu) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setDisallowNnApiCpu);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, disallow_nnapi_cpu);
}

/** This API is deprecated
 * Set cacheable ion buffer
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param cacheable ion buffer
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setCacheableIonBuffer(ANeuralNetworksTFLiteOptions* options,
                                                            bool cacheable_ion_buffer) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setCacheableIonBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, cacheable_ion_buffer);
}

/**
 * Set use Ion
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param use Ion
 *
 * Available only in Neuron Delegate.
 * Available in API level 30.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setUseIon(ANeuralNetworksTFLiteOptions* options,
                                                            bool use_ion) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setUseIon);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, use_ion);
}

/** This API is deprecated
 * Set no supported operation check
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param no supported operation check
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setNoSupportedOperationCheck(ANeuralNetworksTFLiteOptions* options,
                                                            bool no_supported_operation_check) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setNoSupportedOperationCheck);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, no_supported_operation_check);
}

/**
 * Set Accelerator Name.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param accelerator name.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setAcceleratorName(ANeuralNetworksTFLiteOptions* options,
                                                           const char* accelerator_name) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setAcceleratorName);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, accelerator_name);
}

/**
 * Set Execution Priority.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param execution prioriy refer to enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setExecutionPriority(ANeuralNetworksTFLiteOptions* options,
                                                             int execution_priority) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setExecutionPriority);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, execution_priority);
}

/**
 * Set Max Compilation Timeout in NNAPI acceleration mode.{@link NpAccelerationMode}.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max compilation timeout.
 *
 * Available only in NNAPI Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setMaxCompilationTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_compilation_timeout_duration_ns) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setMaxCompilationTimeout);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, max_compilation_timeout_duration_ns);
}

/**
 * Set Max number delegated partition in NNAPI.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max number delegates partitions.
 *
 * Available only in NNAPI Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setMaxNumberDelegatedPartitions(
    ANeuralNetworksTFLiteOptions* options, uint32_t max_number_delegated_partitions) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setMaxNumberDelegatedPartitions);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, max_number_delegated_partitions);
}

/**
 * Set Max Execution Timeout in NNAPI acceleration mode.{@link NpAccelerationMode}.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max execution timeout.
 *
 * Available only in NNAPI Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setMaxExecutionTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_timeout_duration_ns) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setMaxExecutionTimeout);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, max_execution_timeout_duration_ns);
}

/**
 * Set Max Execution Loop Timeout in NNAPI acceleration mode.{@link NpAccelerationMode}.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max execution loop timeout.
 *
 * Available only in NNAPI Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setMaxExecutionLoopTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_loop_timeout_duration_ns) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setMaxExecutionLoopTimeout);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, max_execution_loop_timeout_duration_ns);
}

/** This API is deprecated
 * Set encryption level.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param encryption level refer to {@link NpEncryptionLevel} enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setEncryptionLevel(ANeuralNetworksTFLiteOptions* options,
                                                           int encryption_level) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setEncryptionLevel);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, encryption_level);
}

/**
 * Set the model optimization hint in Neuron acceleration mode.{@link NpAccelerationMode}.
 * Allow to maximize the bandwidth utilization for low latency.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param enableLowLatency True to allow low latency if possible.
 *
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setLowLatency(ANeuralNetworksTFLiteOptions* options,
                                                      bool enableLowLatency) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setLowLatency);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, enableLowLatency);
}

/**
 * Set the model optimization hint in Neuron acceleration mode.{@link NpAccelerationMode}.
 * Allows deep fusion optimization. This may increase the model initialzation time.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param enableDeepFusion True to allow deep fusion if possible.
 *
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setDeepFusion(ANeuralNetworksTFLiteOptions* options,
                                                      bool enableDeepFusion) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setDeepFusion);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, enableDeepFusion);
}

/**
 * Set Neuron compile options.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param compileOptions The string of compile options.
 *
 * Available since API level 31.
 *
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
inline int ANeuralNetworksTFLiteOptions_setCompileOptionByString(ANeuralNetworksTFLiteOptions* options,
        const char* compileOptions) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setCompileOptionByString);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, compileOptions);
}


/**
 * Set the model optimization hint in Neuron acceleration mode.{@link NpAccelerationMode}.
 * Allows batch optimization of models with an N dimension greater than 1. This may increase the
 * model initialzation time.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param enableDeepFusion True to allow deep fusion if possible.
 *
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setBatchProcessing(ANeuralNetworksTFLiteOptions* options,
                                                           bool enableBatchProcessing) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setBatchProcessing);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, enableBatchProcessing);
}

/**
 * Set the number of warm up runs to do after the {@link ANeuroPilotTFLite} instance is created.
 * This may increase the model initialzation time.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param warmupRuns The number of warmup runs.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setWarmupRuns(ANeuralNetworksTFLiteOptions* options,
                                                      uint32_t warmupRuns) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setWarmupRuns);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, warmupRuns);
}

/**
 * Set the model execution boost hint in Neuron acceleration mode.{@link NpAccelerationMode}.
 *
 * For the execution preference set as NEURON_PREFER_SUSTAINED_SPEED, the executing boost value
 * would equal to the boost value hint.
 * On the other hand, for the execution preference set as
 * NEURON_PREFER_LOW_POWER, the executing boost value would not exceed the boost value hint to save
 * power.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param boostValue The hint for the device frequency, ranged between 0 (lowest) to 100 (highest).
 *
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setBoostHint(ANeuralNetworksTFLiteOptions* options,
                                                     uint8_t boostValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setBoostHint);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, boostValue);
}

/**
 * Set the model execution boost duration.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param boostDuration Set boost duration in ms.
 *
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setBoostDuration(ANeuralNetworksTFLiteOptions* options,
                                                     uint32_t boostDuration) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setBoostDuration);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, boostDuration);
}

/**
 * Set use AhardwareBuffer.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param set True to use AhardwareBuffer.
 *
 * Available only in Neuron Delegate.
 * Available since API level 31.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setUseAhwb(ANeuralNetworksTFLiteOptions* options,
                                                     bool use_ahwb) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setUseAhwb);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, use_ahwb);
}

/**
 * Set Extension Attribute To The Compilation Object In NNAPI.
 * To use complie extension attribute, the ANeuralNetworksTFLiteOptions_setAcceleratorName
 * must has been set and numDevices = 1.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param complie extension attribute.
 *
 * Available only in NNAPI Delegate.
 * Available since API level 33.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksTFLiteOptions_setComplieExtensionAttribute(
    ANeuralNetworksTFLiteOptions* options, const char* complieExtensionAttribute) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setComplieExtensionAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, complieExtensionAttribute);
}

/** This API is deprecated
 * Specifies whether to allow extreme performance acceleration of model execution in Neuron
 * acceleration mode {@link NpAccelerationMode} + fast-single-answer {@link ExecutionPreference} by
 * acquiring other system resources at the cost of increased power consumption.
 *
 * This option is enabled by default and apply extreme performance for 2 seconds.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param allow True to apply extreme performance if possible.
 * @param duration Apply extreme performance for the duration in milliseconds.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setAllowExtremePerformance(
    ANeuralNetworksTFLiteOptions* options, bool allow, uint32_t duration) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setAllowExtremePerformance);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, allow, duration);
}

/**
 * Set GPU Execution Preference.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param execution preference refer to enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setGpuExecutionPreference(ANeuralNetworksTFLiteOptions* options,
                                                                  int execution_preference) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setGpuExecutionPreference);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, execution_preference);
}

/**
 * Set GPU Execution Priority.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param priority index means set the setting to priority 1 or 2 or 3.
 * @param priority setting refer to enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksTFLiteOptions_setGpuExecutionPriority(ANeuralNetworksTFLiteOptions* options,
                                                               int priority_index, int priority_setting) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_setGpuExecutionPriority);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options, priority_index, priority_setting);
}

/**
 * Delete a {@link ANeuralNetworksTFLiteOptions} object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} object to be freed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuralNetworksTFLiteOptions_free(ANeuralNetworksTFLiteOptions* options) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteOptions_free);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(options);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeTFLite(ANeuralNetworksTFLite** tflite,
                                               const char* modelPath) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_create);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 * @param customOperations Custom defined operation list.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeCustomTFLite(
    ANeuralNetworksTFLite** tflite, const char* modelPath,
    const std::vector<TFLiteCustomOpExt>& customOperations) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createCustom);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath, customOperations);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 * @param option Option of the {@link ANeuralNetworksTFLite} object.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeAdvTFLite(ANeuralNetworksTFLite** tflite,
                                                  const char* modelPath,
                                                  ANeuralNetworksTFLiteOptions* options) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createAdv);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath, options);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 * @param customOperations Custom defined operation list.
 * @param setting Setting of the {@link ANeuralNetworksTFLite} object.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeCustomTFLite(
    ANeuralNetworksTFLite** tflite, const char* modelPath,
    const std::vector<TFLiteCustomOpExt>& customOperations, ANeuralNetworksTFLiteOptions* options) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createAdvCustom);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, modelPath, customOperations, options);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(ANeuralNetworksTFLite** tflite,
                                                         const char* buffer, size_t bufferSize) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize);
}

/**
 * Make pre-created neuron model add tflite ops and return new neuron model and neuron index.
 *
 * This API is used in CV+NN use case.
 *
 * @param neuron_model Pre-created neuron model.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 * @param neuron_input_index Neuron indexs of input tflite model.
 * @param neuron_output_index Neuron indexs of output tflite model.
 * @param current_neuron_index Return final neuron index after add tflite ops.
 *
 * Available since API level 31.
 * Available only in Neuron Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeNeuronModelWithBuffer(NeuronModel** neuron_model,
                                                              const char* buffer,
                                                              const size_t bufferSize,
                                                              uint32_t* neuron_input_index,
                                                              uint32_t* neuron_output_index,
                                                              uint32_t* current_neuron_index) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createNeuronModelWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(neuron_model, buffer, bufferSize,
            neuron_input_index, neuron_output_index, current_neuron_index);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 * @param customOperations Custom defined operation list.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeCustomTFLiteWithBuffer(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    const std::vector<TFLiteCustomOpExt>& customOperations) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createCustomWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize, customOperations);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 * @param option Option of the {@link ANeuralNetworksTFLite} object.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(ANeuralNetworksTFLite** tflite,
                                                            const char* buffer, size_t bufferSize,
                                                            ANeuralNetworksTFLiteOptions* options) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createAdvWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize, options);
}

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a data buffer pointer.
 * The data buffer will be duplicated in ANeuralNetworksTFLite instance.
 * Caller could free the input data buffer after calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLiteWrapper_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLiteWrapper_free} should be called once the object
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 * @param customOperations Custom defined operation list.
 * @param setting Setting of the {@link ANeuralNetworksTFLite} object.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the model can't be parsed correctly.
 */
inline int ANeuroPilotTFLiteWrapper_makeCustomTFLiteWithBuffer(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    const std::vector<TFLiteCustomOpExt>& customOperations, ANeuralNetworksTFLiteOptions* options) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_createAdvCustomWithBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferSize, customOperations, options);
}

/**
 * Store dequantized contents of the given output tensor to user-allocated buffer.
 * This function is only used with quantized model.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to get dequantized data from the output tensor.
 * @param buffer The pointer to the user-allocated buffer for storing dequantized contents.
 * @param bufferByteSize Specifies the buffer size in bytes.
 * @param tensorIndex Zero-based index of the output tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuroPilotTFLiteWrapper_getDequantizedOutputByIndex(ANeuralNetworksTFLite* tflite,
                                                                void* buffer, size_t bufferByteSize,
                                                                int tensorIndex) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getDequantizedOutputByIndex);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, bufferByteSize, tensorIndex);
}

/**
 * Invoke inference. (run the whole graph in dependency order).
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to invoke inference.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the operation is failed.
 */
inline int ANeuroPilotTFLiteWrapper_invoke(ANeuralNetworksTFLite* tflite) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_invoke);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite);
}

/**
 * Set input/ouput with Ahardwarebuffer and get buffer virtual address.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param memory_data Get Ahardwarebuffer virtual address.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param cacheable Decide cacheable/non-cacheable buffer.
 * @param buffer_size Set buffer size.
 *
 * Available since API level 31.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_setBufferHandle(ANeuralNetworksTFLite* tflite,
        void** memory_data, TFLiteBufferType btype, int index, bool cacheable, int buffer_size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_setBufferHandle);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, memory_data, btype, index, cacheable, buffer_size);
}

/**
 * Set input/ouput with Ahardwarebuffer
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param buffer Set Ahardwarebuffer.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 *
 * Available since API level 31.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_setAhwb(ANeuralNetworksTFLite* tflite,
        AHardwareBuffer* buffer, TFLiteBufferType btype, int index) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_setAhwb);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, buffer, btype, index);
}

/**
 * Delete a {@link ANeuralNetworksTFLite} object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The {@link ANeuralNetworksTFLite} object to be freed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_free(ANeuralNetworksTFLite* tflite) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_free);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite);
}

/**
 * Get the number of input/output tensors associated with the model.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param btype Input or output tensor.
 * @param count the number of input/output tensors.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorCount(ANeuralNetworksTFLite* tflite,
                                                   TFLiteBufferType btype, int32_t* count) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorCount);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, count);
}

/**
 * Get the dimensional information of the input/output tensor with the given index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param rank The rank of the tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorRank(ANeuralNetworksTFLite* tflite,
                                                  TFLiteBufferType btype, int index, int* rank) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorRank);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, index, rank);
}

/**
 * Get the dimensional information of the input/output tensor with the given index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param dimensions The dimension array to be filled. The size of the array
 *                   must be exactly as large as the rank.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorDimensions(ANeuralNetworksTFLite* tflite,
                                                        TFLiteBufferType btype, int index,
                                                        int* dimensions) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorDimensions);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, index, dimensions);
}

/**
 * Get the size of the underlying data in bytes.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param size The tensor's size in bytes.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorByteSize(ANeuralNetworksTFLite* tflite,
                                                      TFLiteBufferType btype, int index,
                                                      size_t* size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorByteSize);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, index, size);
}

/**
 * Get the data type information of the input/output tensor with the given index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param ttpte The tensor's data type.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorType(ANeuralNetworksTFLite* tflite,
                                                  TFLiteBufferType btype, int index,
                                                  TFLiteTensorType* ttype) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorType);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, index, ttype);
}

/**
 * Get the quantization information of the input/output tensor with the given index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param quantization The quantization information of the tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getTensorQuantizeParams(
       ANeuralNetworksTFLite* tflite, TFLiteBufferType btype, int index, TfLiteQuantization* quantization) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getTensorQuantizeParams);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, btype, index, quantization);
}

/**
 * Get the data type information of the input/output tensor with the given
 * index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output
 * tensor.
 * @param btype Input or output tensor.
 * @param tensorIndex Zero-based index of tensor.
 * @param data The buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_setTensorBuffer(ANeuralNetworksTFLite* tflite, int index,
                                                    char* data) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_setTensorBuffer);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, index, data);
}

/**
 * Copies from the provided input buffer into the input tensor's buffer.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param index Zero-based index of the input tensor.
 * @param data The input buffer.
 * @param size The input buffer's size in bytes.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_setInputTensorData(ANeuralNetworksTFLite* tflite, int index,
                                                       const void* data, size_t size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_setInputTensorData);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, index, data, size);
}

/**
 * Copies to the provided output buffer from the output tensor's buffer.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the output tensor.
 * @param index Zero-based index of the output tensor.
 * @param data The output buffer.
 * @param size The output buffer's size in bytes.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
inline int ANeuroPilotTFLiteWrapper_getOutputTensorData(ANeuralNetworksTFLite* tflite, int index,
                                                        void* data, size_t size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLite_getOutputTensorData);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(tflite, index, data, size);
}

/**
 * Get inference preference of current platform.
 *
 * @return NP_INFERENCE_TYPE_NONE if NeuroPilot is not supported.
 *         NP_INFERENCE_TYPE_QNAUT if quantization inference is preferred.
 *         NP_INFERENCE_TYPE_FLOAT if float inference is preferred.
 */

inline int ANeuroPilotWrapper_getInferencePreference(void) {
    LOAD_TFLITE_FUNCTION(ANeuroPilot_getInferencePreference);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT();
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpIntAttribute(const char* buffer, size_t length,
                                                            const char* attr, int32_t* outValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getIntAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(buffer, length, attr, outValue);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpFloatAttribute(const char* buffer, size_t length,
                                                              const char* attr, float* outValue) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getFloatAttribute);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(buffer, length, attr, outValue);
}

inline void* ANeuroPilotTFLiteWrapper_getCustomOpUserData(TfLiteNode* node) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getUserData);
    EXECUTE_TFLITE_FUNCTION_RETURN_POINTER(node);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpInput(TfLiteContext* context, TfLiteNode* node,
                                                     int index, TFLiteTensorExt* tfliteTensor) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getInput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, tfliteTensor);
}

inline int ANeuroPilotTFLiteWrapper_getCustomOpOutput(TfLiteContext* context, TfLiteNode* node,
                                                      int index, TFLiteTensorExt* tfliteTensor) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_getOutput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, tfliteTensor);
}

inline int ANeuroPilotTFLiteWrapper_resizeCustomOpOutput(TfLiteContext* context, TfLiteNode* node,
                                                         int index, TfLiteIntArray* new_size) {
    LOAD_TFLITE_FUNCTION(ANeuroPilotTFLiteCustomOp_resizeOutput);
    EXECUTE_TFLITE_FUNCTION_RETURN_INT(context, node, index, new_size);
}

#endif  // __MTK_NEUROPILOT_TFLITE_SHIM_H__
