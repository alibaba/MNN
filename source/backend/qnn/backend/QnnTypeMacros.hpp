//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
// TODO: remove once the SNPE build for QNN core is sorted out
#pragma once
#include "QnnTypes.h"
#define QNN_OP_CFG_VALID(opConfig) ((opConfig).version == QNN_OPCONFIG_VERSION_1)
/**
 * @brief Verifies the tensor object passed is of supported Qnn_Tensor_t API version
 *
 * @param[in] tensor Qnn_Tensor_t object to validate
 *
 * @return Error code
 */
inline bool validateTensorVersion(Qnn_Tensor_t tensor) {
  return !(tensor.version != QNN_TENSOR_VERSION_1 && tensor.version != QNN_TENSOR_VERSION_2);
}
/**
 * @brief Verifies the tensor object passed is of supported Qnn_OpConfig_t API version
 *
 * @param[in] tensor Qnn_OpConfig_t object to validate
 *
 * @return Error code
 */
inline bool validateOpConfigVersion(Qnn_OpConfig_t opConfig) {
  return !(opConfig.version != QNN_OPCONFIG_VERSION_1);
}
inline Qnn_OpConfig_t createQnnOpConfig(const Qnn_OpConfigVersion_t version) {
  Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
  opConfig.version        = version;
  if (version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1 = QNN_OPCONFIG_V1_INIT;
  }
  return opConfig;
}
inline const char* getQnnOpConfigName(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.name;
  }
  return NULL;
}
inline const char* getQnnOpConfigName(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigName(*opConfig);
}
inline const char* getQnnOpConfigPackageName(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.packageName;
  }
  return NULL;
}
inline const char* getQnnOpConfigPackageName(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigPackageName(*opConfig);
}
inline const char* getQnnOpConfigTypeName(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.typeName;
  }
  return NULL;
}
inline const char* getQnnOpConfigTypeName(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigTypeName(*opConfig);
}
inline uint32_t getQnnOpConfigNumParams(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.numOfParams;
  }
  return 0u;
}
inline uint32_t getQnnOpConfigNumParams(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigNumParams(*opConfig);
}
inline Qnn_Param_t* getQnnOpConfigParams(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.params;
  }
  return NULL;
}
inline Qnn_Param_t* getQnnOpConfigParams(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigParams(*opConfig);
}
inline uint32_t getQnnOpConfigNumInputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.numOfInputs;
  }
  return 0u;
}
inline uint32_t getQnnOpConfigNumInputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigNumInputs(*opConfig);
}
inline Qnn_Tensor_t* getQnnOpConfigInputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.inputTensors;
  }
  return NULL;
}
inline Qnn_Tensor_t* getQnnOpConfigInputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigInputs(*opConfig);
}
inline uint32_t getQnnOpConfigNumOutputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.numOfOutputs;
  }
  return 0u;
}
inline uint32_t getQnnOpConfigNumOutputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigNumOutputs(*opConfig);
}
inline Qnn_Tensor_t* getQnnOpConfigOutputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.outputTensors;
  }
  return NULL;
}
inline Qnn_Tensor_t* getQnnOpConfigOutputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigOutputs(*opConfig);
}
inline void setQnnOpConfigName(Qnn_OpConfig_t& opConfig, const char* name) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.name = name;
  }
}
inline void setQnnOpConfigName(Qnn_OpConfig_t* opConfig, const char* name) {
  setQnnOpConfigName(*opConfig, name);
}
inline void setQnnOpConfigPackageName(Qnn_OpConfig_t& opConfig, const char* packageName) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.packageName = packageName;
  }
}
inline void setQnnOpConfigPackageName(Qnn_OpConfig_t* opConfig, const char* packageName) {
  setQnnOpConfigPackageName(*opConfig, packageName);
}
inline void setQnnOpConfigTypeName(Qnn_OpConfig_t& opConfig, const char* typeName) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.typeName = typeName;
  }
}
inline void setQnnOpConfigTypeName(Qnn_OpConfig_t* opConfig, const char* typeName) {
  setQnnOpConfigTypeName(*opConfig, typeName);
}
inline void setQnnOpConfigParams(Qnn_OpConfig_t& opConfig,
                                 uint32_t numOfParams,
                                 Qnn_Param_t* params) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.numOfParams = numOfParams;
    opConfig.v1.params      = params;
  }
}
inline void setQnnOpConfigParams(Qnn_OpConfig_t* opConfig,
                                 uint32_t numOfParams,
                                 Qnn_Param_t* params) {
  setQnnOpConfigParams(*opConfig, numOfParams, params);
}
inline void setQnnOpConfigInputs(Qnn_OpConfig_t& opConfig,
                                 uint32_t numOfInputs,
                                 Qnn_Tensor_t* inputTensors) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.numOfInputs  = numOfInputs;
    opConfig.v1.inputTensors = inputTensors;
  }
}
inline void setQnnOpConfigInputs(Qnn_OpConfig_t* opConfig,
                                 uint32_t numOfInputs,
                                 Qnn_Tensor_t* inputTensors) {
  setQnnOpConfigInputs(*opConfig, numOfInputs, inputTensors);
}
inline void setQnnOpConfigOutputs(Qnn_OpConfig_t& opConfig,
                                  uint32_t numOfOutputs,
                                  Qnn_Tensor_t* outputTensors) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.numOfOutputs  = numOfOutputs;
    opConfig.v1.outputTensors = outputTensors;
  }
}
inline void setQnnOpConfigOutputs(Qnn_OpConfig_t* opConfig,
                                  uint32_t numOfOutputs,
                                  Qnn_Tensor_t* outputTensors) {
  setQnnOpConfigOutputs(*opConfig, numOfOutputs, outputTensors);
}
inline Qnn_Tensor_t createQnnTensor(const Qnn_TensorVersion_t version) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version      = version;
  if (version == QNN_TENSOR_VERSION_1) {
    tensor.v1 = QNN_TENSOR_V1_INIT;
  } else if (version == QNN_TENSOR_VERSION_2) {
    tensor.v2 = QNN_TENSOR_V2_INIT;
  }
  return tensor;
}
inline uint32_t getQnnTensorId(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.id;
}
inline uint32_t getQnnTensorId(const Qnn_Tensor_t* tensor) { return getQnnTensorId(*tensor); }
inline const char* getQnnTensorName(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.name;
}
inline const char* getQnnTensorName(const Qnn_Tensor_t* tensor) {
  return getQnnTensorName(*tensor);
}
inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.type;
}
inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t* tensor) {
  return getQnnTensorType(*tensor);
}
inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.dataFormat;
}
inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t* tensor) {
  return getQnnTensorDataFormat(*tensor);
}
inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.dataType;
}
inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t* tensor) {
  return getQnnTensorDataType(*tensor);
}
inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.quantizeParams;
}
inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t* const tensor) {
  if (tensor != nullptr) {
    return getQnnTensorQuantParams(*tensor);
  }
  return QNN_QUANTIZE_PARAMS_INIT;
}
inline uint32_t getQnnTensorRank(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.rank;
}
inline uint32_t getQnnTensorRank(const Qnn_Tensor_t* const tensor) {
  if (tensor != nullptr) {
    return getQnnTensorRank(*tensor);
  }
  return 0u;
}
inline uint32_t* getQnnTensorDimensions(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.dimensions;
}
inline uint32_t* getQnnTensorDimensions(const Qnn_Tensor_t* tensor) {
  return getQnnTensorDimensions(*tensor);
}
inline uint8_t* getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.isDynamicDimensions;
  }
  return NULL;
}
inline uint8_t* getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t* tensor) {
  return getQnnTensorIsDynamicDimensions(*tensor);
}
inline Qnn_SparseParams_t getQnnTensorSparseParams(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.sparseParams;
  }
  return QNN_SPARSE_PARAMS_INIT;
}
inline Qnn_SparseParams_t getQnnTensorSparseParams(const Qnn_Tensor_t* tensor) {
  return getQnnTensorSparseParams(*tensor);
}
inline Qnn_TensorMemType_t getQnnTensorMemType(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.memType;
}
inline Qnn_TensorMemType_t getQnnTensorMemType(const Qnn_Tensor_t* tensor) {
  return getQnnTensorMemType(*tensor);
}
inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.clientBuf;
}
inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t* tensor) {
  return getQnnTensorClientBuf(*tensor);
}
inline Qnn_MemHandle_t getQnnTensorMemHandle(const Qnn_Tensor_t& tensor) {
  // TensorCompatTest justifies no need to check version
  return tensor.v1.memHandle;
}
inline Qnn_MemHandle_t getQnnTensorMemHandle(const Qnn_Tensor_t* tensor) {
  return getQnnTensorMemHandle(*tensor);
}
inline void setQnnTensorId(Qnn_Tensor_t& tensor, const uint32_t id) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.id = id;
}
inline void setQnnTensorId(Qnn_Tensor_t* tensor, uint32_t id) { setQnnTensorId(*tensor, id); }
inline void setQnnTensorName(Qnn_Tensor_t& tensor, const char* const name) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.name = name;
}
inline void setQnnTensorName(Qnn_Tensor_t* tensor, const char* name) {
  setQnnTensorName(*tensor, name);
}
inline void setQnnTensorType(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.type = type;
}
inline void setQnnTensorType(Qnn_Tensor_t* tensor, Qnn_TensorType_t type) {
  setQnnTensorType(*tensor, type);
}
inline void setQnnTensorDataFormat(Qnn_Tensor_t& tensor, const Qnn_TensorDataFormat_t dataFormat) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.dataFormat = dataFormat;
}
inline void setQnnTensorDataFormat(Qnn_Tensor_t* tensor, Qnn_TensorDataFormat_t format) {
  setQnnTensorDataFormat(*tensor, format);
}
inline void setQnnTensorDataType(Qnn_Tensor_t& tensor, const Qnn_DataType_t dataType) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.dataType = dataType;
}
inline void setQnnTensorDataType(Qnn_Tensor_t* tensor, Qnn_DataType_t dataType) {
  setQnnTensorDataType(*tensor, dataType);
}
inline void setQnnTensorQuantParams(Qnn_Tensor_t& tensor,
                                    const Qnn_QuantizeParams_t quantizeParams) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.quantizeParams = quantizeParams;
}
inline void setQnnTensorQuantParams(Qnn_Tensor_t* tensor, Qnn_QuantizeParams_t params) {
  setQnnTensorQuantParams(*tensor, params);
}
inline void setQnnTensorRank(Qnn_Tensor_t& tensor, const uint32_t rank) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.rank = rank;
}
inline void setQnnTensorRank(Qnn_Tensor_t* tensor, uint32_t rank) {
  setQnnTensorRank(*tensor, rank);
}
inline void setQnnTensorDimensions(Qnn_Tensor_t& tensor, uint32_t* const dimensions) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.dimensions = dimensions;
}
inline void setQnnTensorDimensions(Qnn_Tensor_t* tensor, uint32_t* dims) {
  setQnnTensorDimensions(*tensor, dims);
}
inline void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t& tensor, uint8_t* isDynamic) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.isDynamicDimensions = isDynamic;
  }
}
inline void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t* tensor, uint8_t* isDynamic) {
  setQnnTensorIsDynamicDimensions(*tensor, isDynamic);
}
inline void setQnnTensorSparseParams(Qnn_Tensor_t& tensor, Qnn_SparseParams_t sparseParams) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.sparseParams = sparseParams;
  }
}
inline void setQnnTensorSparseParams(Qnn_Tensor_t* tensor, Qnn_SparseParams_t sparseParams) {
  setQnnTensorSparseParams(*tensor, sparseParams);
}
inline void setQnnTensorMemType(Qnn_Tensor_t& tensor, const Qnn_TensorMemType_t memType) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.memType = memType;
}
inline void setQnnTensorMemType(Qnn_Tensor_t* tensor, Qnn_TensorMemType_t memType) {
  setQnnTensorMemType(*tensor, memType);
}
inline void setQnnTensorClientBuf(Qnn_Tensor_t& tensor, const Qnn_ClientBuffer_t clientBuf) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.clientBuf = clientBuf;
}
inline void setQnnTensorClientBuf(Qnn_Tensor_t* tensor, Qnn_ClientBuffer_t clientBuf) {
  setQnnTensorClientBuf(*tensor, clientBuf);
}
inline void setQnnTensorMemHandle(Qnn_Tensor_t& tensor, const Qnn_MemHandle_t memHandle) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.memHandle = memHandle;
}
inline void setQnnTensorMemHandle(Qnn_Tensor_t* tensor, Qnn_MemHandle_t handle) {
  setQnnTensorMemHandle(*tensor, handle);
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t& tensor,
                                          Qnn_TensorRetrieveRaw_t* const retrieve) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.retrieveRaw = retrieve;
  }
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t* const tensor,
                                          Qnn_TensorRetrieveRaw_t* const retrieve) {
  setQnnTensorClientBufRetrieve(*tensor, retrieve);
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t& tensor, Qnn_TensorRetrieveRaw_t& retrieve) {
  setQnnTensorClientBufRetrieve(tensor, &retrieve);
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t* const tensor,
                                          Qnn_TensorRetrieveRaw_t& retrieve) {
  setQnnTensorClientBufRetrieve(*tensor, &retrieve);
}
inline Qnn_TensorRetrieveRaw_t* getQnnTensorClientBufRetrieve(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.retrieveRaw;
  }
  return nullptr;
}
inline Qnn_TensorRetrieveRaw_t* getQnnTensorClientBufRetrieve(const Qnn_Tensor_t* const tensor) {
  return getQnnTensorClientBufRetrieve(*tensor);
}
inline Qnn_TensorSet_t createQnnTensorSet(const Qnn_TensorSetVersion_t version) {
  Qnn_TensorSet_t tensorSet = QNN_TENSOR_SET_INIT;
  tensorSet.version         = version;
  if (version == QNN_TENSOR_SET_VERSION_1) {
    tensorSet.v1 = QNN_TENSOR_SET_V1_INIT;
  }
  return tensorSet;
}
inline uint32_t getQnnTensorSetNumInputs(const Qnn_TensorSet_t& tensorSet) {
  if (tensorSet.version == QNN_TENSOR_SET_VERSION_1) {
    return tensorSet.v1.numInputs;
  }
  return 0;
}
inline uint32_t getQnnTensorSetNumInputs(const Qnn_TensorSet_t* tensorSet) {
  return getQnnTensorSetNumInputs(*tensorSet);
}
inline Qnn_Tensor_t* getQnnTensorSetInputTensors(const Qnn_TensorSet_t& tensorSet) {
  if (tensorSet.version == QNN_TENSOR_SET_VERSION_1) {
    return tensorSet.v1.inputs;
  }
  return 0;
}
inline Qnn_Tensor_t* getQnnTensorSetInputTensors(const Qnn_TensorSet_t* tensorSet) {
  return getQnnTensorSetInputTensors(*tensorSet);
}
inline uint32_t getQnnTensorSetNumOutputs(const Qnn_TensorSet_t& tensorSet) {
  if (tensorSet.version == QNN_TENSOR_SET_VERSION_1) {
    return tensorSet.v1.numOutputs;
  }
  return 0;
}
inline uint32_t getQnnTensorSetNumOutputs(const Qnn_TensorSet_t* tensorSet) {
  return getQnnTensorSetNumOutputs(*tensorSet);
}
inline Qnn_Tensor_t* getQnnTensorSetOutputTensors(const Qnn_TensorSet_t& tensorSet) {
  if (tensorSet.version == QNN_TENSOR_SET_VERSION_1) {
    return tensorSet.v1.outputs;
  }
  return 0;
}
inline Qnn_Tensor_t* getQnnTensorSetOutputTensors(const Qnn_TensorSet_t* tensorSet) {
  return getQnnTensorSetOutputTensors(*tensorSet);
}
inline void setQnnTensorSetInputTensors(Qnn_TensorSet_t& tensorSet,
                                        Qnn_Tensor_t* inputTensors,
                                        uint32_t const numInputs) {
  if (tensorSet.version == QNN_TENSOR_SET_VERSION_1) {
    tensorSet.v1.inputs    = inputTensors;
    tensorSet.v1.numInputs = numInputs;
  }
}
inline void setQnnTensorSetInputTensors(Qnn_TensorSet_t* tensorSet,
                                        Qnn_Tensor_t* inputTensors,
                                        uint32_t const numInputs) {
  setQnnTensorSetInputTensors(*tensorSet, inputTensors, numInputs);
}
inline void setQnnTensorSetOutputTensors(Qnn_TensorSet_t& tensorSet,
                                         Qnn_Tensor_t* outputTensors,
                                         const uint32_t numOutputs) {
  if (tensorSet.version == QNN_TENSOR_SET_VERSION_1) {
    tensorSet.v1.outputs    = outputTensors;
    tensorSet.v1.numOutputs = numOutputs;
  }
}
inline void setQnnTensorSetOutputTensors(Qnn_TensorSet_t* tensorSet,
                                         Qnn_Tensor_t* outputTensors,
                                         const uint32_t numOutputs) {
  setQnnTensorSetOutputTensors(*tensorSet, outputTensors, numOutputs);
}
// Validation
#define VALIDATE_TENSOR_VERSION(tensor, err) validateTensorVersion(tensor)
#define VALIDATE_OP_CONFIG_VERSION(op, err)  validateOpConfigVersion(op)
// Creator for QNN Op Config
#define QNN_OP_CFG_CREATE(version) createQnnOpConfig(version)
// Accessors for QNN Op Config
#define QNN_OP_CFG_GET_NAME(opConfig)         getQnnOpConfigName(opConfig)
#define QNN_OP_CFG_GET_PACKAGE_NAME(opConfig) getQnnOpConfigPackageName(opConfig)
#define QNN_OP_CFG_GET_TYPE_NAME(opConfig)    getQnnOpConfigTypeName(opConfig)
#define QNN_OP_CFG_GET_NUM_PARAMS(opConfig)   getQnnOpConfigNumParams(opConfig)
#define QNN_OP_CFG_GET_PARAMS(opConfig)       getQnnOpConfigParams(opConfig)
#define QNN_OP_CFG_GET_NUM_INPUTS(opConfig)   getQnnOpConfigNumInputs(opConfig)
#define QNN_OP_CFG_GET_INPUTS(opConfig)       getQnnOpConfigInputs(opConfig)
#define QNN_OP_CFG_GET_NUM_OUTPUTS(opConfig)  getQnnOpConfigNumOutputs(opConfig)
#define QNN_OP_CFG_GET_OUTPUTS(opConfig)      getQnnOpConfigOutputs(opConfig)
// Modifiers for QNN Op Config
#define QNN_OP_CFG_SET_NAME(opConfig, value)         setQnnOpConfigName(opConfig, value)
#define QNN_OP_CFG_SET_PACKAGE_NAME(opConfig, value) setQnnOpConfigPackageName(opConfig, value)
#define QNN_OP_CFG_SET_TYPE_NAME(opConfig, value)    setQnnOpConfigTypeName(opConfig, value)
#define QNN_OP_CFG_SET_PARAMS(opConfig, numOfParams, params) \
  setQnnOpConfigParams(opConfig, numOfParams, params)
#define QNN_OP_CFG_SET_INPUTS(opConfig, numOfInputs, inputTensors) \
  setQnnOpConfigInputs(opConfig, numOfInputs, inputTensors)
#define QNN_OP_CFG_SET_OUTPUTS(opConfig, numOfOutputs, outputTensors) \
  setQnnOpConfigOutputs(opConfig, numOfOutputs, outputTensors)
// Creator for QNN Tensor
#define QNN_TENSOR_CREATE(version) createQnnTensor(version)
// Accessors for QNN Tensor
#define QNN_TENSOR_GET_ID(tensor)                    getQnnTensorId(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                  getQnnTensorName(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                  getQnnTensorType(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)           getQnnTensorDataFormat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)             getQnnTensorDataType(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)          getQnnTensorQuantParams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                  getQnnTensorRank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)            getQnnTensorDimensions(tensor)
#define QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor) getQnnTensorIsDynamicDimensions(tensor)
#define QNN_TENSOR_GET_SPARSE_PARAMS(tensor)         getQnnTensorSparseParams(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)              getQnnTensorMemType(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor)            getQnnTensorClientBuf(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor)            getQnnTensorMemHandle(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF_RETRIEVE(tensor)   getQnnTensorClientBufRetrieve(tensor)
// Modifiers for QNN Tensor
#define QNN_TENSOR_SET_ID(tensor, value)           setQnnTensorId(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)         setQnnTensorName(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)         setQnnTensorType(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)  setQnnTensorDataFormat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)    setQnnTensorDataType(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value) setQnnTensorQuantParams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)         setQnnTensorRank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)   setQnnTensorDimensions(tensor, value)
#define QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(tensor, value) \
  setQnnTensorIsDynamicDimensions(tensor, value)
#define QNN_TENSOR_SET_SPARSE_PARAMS(tensor, value) setQnnTensorSparseParams(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)      setQnnTensorMemType(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)    setQnnTensorClientBuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)    setQnnTensorMemHandle(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF_RETRIEVE(tensor, value) \
  setQnnTensorClientBufRetrieve(tensor, value)
// Creator for QNN Tensor Set
#define QNN_TENSORSET_CREATE(version) createQnnTensorSet(version)
// Accessors for QNN Tensor Set
#define QNN_TENSORSET_GET_NUM_INPUTS(tensorSet)     getQnnTensorSetNumInputs(tensorSet)
#define QNN_TENSORSET_GET_INPUT_TENSORS(tensorSet)  getQnnTensorSetInputTensors(tensorSet)
#define QNN_TENSORSET_GET_NUM_OUTPUTS(tensorSet)    getQnnTensorSetNumOutputs(tensorSet)
#define QNN_TENSORSET_GET_OUTPUT_TENSORS(tensorSet) getQnnTensorSetOutputTensors(tensorSet)
// Modifiers for QNN Tensor Set
#define QNN_TENSORSET_SET_INPUT_TENSORS(tensorSet, inputTensors, numInputs) \
  setQnnTensorSetInputTensors(tensorSet, inputTensors, numInputs)
#define QNN_TENSORSET_SET_OUTPUT_TENSORS(tensorSet, outputTensors, numOutputs) \
  setQnnTensorSetOutputTensors(tensorSet, outputTensors, numOutputs)
inline bool isQnnTensorV1Compatible(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    if (tensor.v2.isDynamicDimensions != NULL) {
      return false;
    }
    if (tensor.v2.dataFormat == QNN_TENSOR_DATA_FORMAT_SPARSE) {
      return false;
    }
  }
  return true;
}
inline bool isQnnTensorV1Compatible(const Qnn_Tensor_t* const tensor) {
  return isQnnTensorV1Compatible(*tensor);
}
inline bool isQnnTensorV1Compatible(const Qnn_OpConfig_t& opConfig) {
  if ((QNN_OP_CFG_GET_INPUTS(opConfig) != NULL) && (QNN_OP_CFG_GET_NUM_INPUTS(opConfig) > 0u)) {
    for (uint32_t tensorIdx = 0u; tensorIdx < QNN_OP_CFG_GET_NUM_INPUTS(opConfig); tensorIdx++) {
      if (!isQnnTensorV1Compatible(QNN_OP_CFG_GET_INPUTS(opConfig)[tensorIdx])) {
        return false;
      }
    }
  }
  if ((QNN_OP_CFG_GET_OUTPUTS(opConfig) != NULL) && (QNN_OP_CFG_GET_NUM_OUTPUTS(opConfig) > 0u)) {
    for (uint32_t tensorIdx = 0u; tensorIdx < QNN_OP_CFG_GET_NUM_OUTPUTS(opConfig); tensorIdx++) {
      if (!isQnnTensorV1Compatible(QNN_OP_CFG_GET_OUTPUTS(opConfig)[tensorIdx])) {
        return false;
      }
    }
  }
  if ((QNN_OP_CFG_GET_PARAMS(opConfig) != NULL) && (QNN_OP_CFG_GET_NUM_PARAMS(opConfig) > 0)) {
    for (uint32_t paramIdx = 0u; paramIdx < QNN_OP_CFG_GET_NUM_PARAMS(opConfig); paramIdx++) {
      const Qnn_Param_t& param = QNN_OP_CFG_GET_PARAMS(opConfig)[paramIdx];
      if (QNN_PARAMTYPE_TENSOR == param.paramType) {
        if (!isQnnTensorV1Compatible(param.tensorParam)) {
          return false;
        }
      }
    }
  }
  return true;
}
inline bool isQnnTensorV1Compatible(const Qnn_OpConfig_t* const opConfig) {
  return isQnnTensorV1Compatible(*opConfig);
}