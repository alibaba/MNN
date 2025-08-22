//
//  QNNBackend.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBackend.hpp"
#include "QnnTypeMacros.hpp"
#include "core/MNNFileUtils.h"

namespace MNN {
namespace QNN {
struct QnnContext {
    QNN_INTERFACE_VER_TYPE interface{};
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    int soc_id;
    int dsp_arch;
};

static QnnContext gContext;
}
}
#ifdef MNN_WITH_PLUGIN

#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "shape/SizeComputer.hpp"

namespace MNN {
namespace plugin {

namespace shape_inference {
class PluginShapeRaw : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};
bool computeIndex(PluginContext* ctx, int & index) {
    const std::vector<Tensor *> & inputs = ctx->inputs();
    std::string inputShapeStr = "";
    for (int i = 0; i < inputs.size(); i++) {
        if (i > 0) {
            inputShapeStr += "_";
        }
        std::vector<int> inputShape = inputs[i]->shape();
        for (int j = 0; j < inputShape.size(); j++) {
            if (j > 0) {
                inputShapeStr += "x";
            }
            inputShapeStr += std::to_string(inputShape[j]);
        }
    }

    auto attrAllShape = ctx->getAttr("allInputShape");
    std::vector<std::string> vec;
    if (nullptr != attrAllShape && nullptr != attrAllShape->list() && nullptr != attrAllShape->list()->s()) {
        for (int i = 0; i < attrAllShape->list()->s()->size(); ++i) {
            vec.push_back(attrAllShape->list()->s()->GetAsString(i)->str());
        }
    } else {
        MNN_ERROR("MNN_QNN: Incorrect Plugin Op.\n");
        return false;
    }

    auto it = std::find(vec.begin(), vec.end(), inputShapeStr);
    if (it != vec.end()) {
        index = std::distance(vec.begin(), it);
        return true;
    }

    MNN_ERROR("MNN_QNN: Failed to find inputShape %s in Plugin Op.\n", inputShapeStr.c_str());
    return false;
}

bool PluginShapeRaw::compute(InferShapeContext* ctx) {
    if (ctx->hasAttr("op")) {
        auto attr = ctx->getAttr("op");
        if (nullptr != attr->tensor() && nullptr != attr->tensor()->int8s()) {
            auto realop = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
            return SizeComputer::computeOutputSize(realop, ctx->inputs(), ctx->outputs());
        }
    } else {
        int shapeIndex = 0;
        if (!(computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to compute shape for Plugin Op.\n");
            return false;
        }

        std::string prefix = "o_" + std::to_string(shapeIndex) + "_";
        for (int i=0; i<ctx->outputs().size(); ++i) {
            auto dst = ctx->output(i);
            std::string key = prefix + std::to_string(i);
            auto attr = ctx->getAttr(key.c_str());

            if (nullptr == attr || nullptr == attr->tensor()) {
                MNN_ERROR("MNN_QNN: Failed to find raw shape %s.\n", key.c_str());
                return false;
            }
            auto blob = attr->tensor();
            dst->setType(blob->dataType());
            if (nullptr != blob->dims()) {
                dst->buffer().dimensions = blob->dims()->size();
                for (int j=0; j<blob->dims()->size(); ++j) {
                    dst->setLength(j, blob->dims()->data()[j]);
                }
            } else {
                dst->buffer().dimensions = 0;
            }
            TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
        }
        return true;
    }
    return false;
}
}

namespace backend {
static bool freeQnnTensor(Qnn_Tensor_t &tensor) {
  // free all pointer allocations in struct
  free((void *)QNN_TENSOR_GET_NAME(tensor));
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
  free(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor));

  auto quant    = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  auto encoding = quant.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    free(quant.axisScaleOffsetEncoding.scaleOffset);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    free(quant.bwAxisScaleOffsetEncoding.scales);
    if (quant.bwAxisScaleOffsetEncoding.offsets != nullptr) {
      free(quant.bwAxisScaleOffsetEncoding.offsets);
    }
  }
  return true;
}

static bool freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensor(tensors[i]);
  }
  free(tensors);

  return true;
}

struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
};

static bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src) {
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version           = src->version;
  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, ::strdup(tensorName));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      ::memcpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                             QNN_TENSOR_GET_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
          dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
      ::memcpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
                             QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

static bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                                 Qnn_Tensor_t *&tensorWrappers,
                                 uint32_t tensorsCount) {
  auto returnStatus = true;
  tensorWrappers    = (Qnn_Tensor_t *)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    MNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
      #ifdef QNN_VERBOSE
      MNN_PRINT("Extracting tensorInfo for tensor Idx: %d.\n", (int) tIdx);
      #endif
      tensorWrappers[tIdx] = QNN_TENSOR_INIT;
      deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }
  }
  return returnStatus;
}
static bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t *graphInfoSrc,
                                  GraphInfo *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName =
        ::strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

static bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t *graphInfoSrc,
                                  GraphInfo *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName =
        strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

static bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                                const uint32_t numGraphs,
                                GraphInfo **&graphsInfo) {
  if (!graphsInput) {
    MNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (GraphInfo **)calloc(numGraphs, sizeof(GraphInfo *));
  GraphInfo *graphInfoArr =
      (GraphInfo *)calloc(numGraphs, sizeof(GraphInfo));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    MNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
      #ifdef QNN_VERBOSE
      MNN_PRINT("Extracting graphsInfo for graph Idx: %d", (int) gIdx);
      #endif
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
      } else if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        copyGraphsInfoV3(&graphsInput[gIdx].graphInfoV3, &graphInfoArr[gIdx]);
      }
      graphsInfo[gIdx] = graphInfoArr + gIdx;
    }
  }
  if (true != returnStatus) {
    MNN_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensors(graphsInfo[gIdx]->inputTensors,
                                          graphsInfo[gIdx]->numInputTensors);
          freeQnnTensors(graphsInfo[gIdx]->outputTensors,
                                          graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }
  return true;
}

static bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                                          GraphInfo **&graphsInfo,
                                          uint32_t &graphsCount) {
  if (nullptr == binaryInfo) {
    MNN_ERROR("binaryInfo is nullptr.");
    return false;
  }
  graphsCount = 0;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    if (binaryInfo->contextBinaryInfoV1.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                          binaryInfo->contextBinaryInfoV1.numGraphs,
                          graphsInfo)) {
        MNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    if (binaryInfo->contextBinaryInfoV2.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                          binaryInfo->contextBinaryInfoV2.numGraphs,
                          graphsInfo)) {
        MNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    if (binaryInfo->contextBinaryInfoV3.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV3.graphs,
                          binaryInfo->contextBinaryInfoV3.numGraphs,
                          graphsInfo)) {
        MNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV3.numGraphs;
      return true;
    }
  }
  MNN_ERROR("Unrecognized system context binary info version.");
  return false;
}

static bool freeGraphsInfo(GraphInfo ***graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    free((*graphsInfo)[i]->graphName);
    freeQnnTensors((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
    freeQnnTensors((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
  }
  free(**graphsInfo);
  free(*graphsInfo);
  *graphsInfo = nullptr;
  return true;
}

class RawExecutorWrapper {
private:
    Qnn_ContextHandle_t mQnnContextHandle = nullptr;
    const QnnContext_Config_t** mQnnContextConfig = nullptr;
    std::vector<Qnn_GraphHandle_t> mQnnGraphHandleVec = {};
    QnnHtpGraph_CustomConfig_t mQnnHtpGraphCustomConfig{};
    QnnGraph_Config_t mQnnGraphConfig{};
    int mInputNumber;
    int mOutputNumber;
    GraphInfo **mGraphsInfo = nullptr;
    uint32_t mGraphCount = 0;
    std::string mPath;
    std::unique_ptr<QNN::QNNPerf> mPerf;
public:
    RawExecutorWrapper(int inputNumber, int outputNumber) {
        mInputNumber = inputNumber;
        mOutputNumber = outputNumber;
        mPerf = QNN::QNNPerf::create(&QNN::gContext.interface);
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    ~ RawExecutorWrapper() {
        if (nullptr != mQnnContextHandle) {
            CALL_QNN(QNN::gContext.interface.contextFree(mQnnContextHandle, nullptr));
        }
        freeGraphsInfo(&mGraphsInfo, mGraphCount);
    }
    bool compileModel(const std::string & path) {
        mPath = path;
        file_t file = MNNOpenFile(path.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
        auto size = MNNGetFileSize(file);
        void* buffer = MNNMmapFile(file, size);

        // 1. Set mGraphsInfo and mGraphCount from the buffer.
        {
            QnnSystemContext_Handle_t systemContextHandle = nullptr;
            if (QNN_SUCCESS != QNN::gContext.systemInterface.systemContextCreate(&systemContextHandle)) {
                MNN_ERROR("Could not create system context handle.");
                return false;
            }
            Qnn_ContextBinarySize_t binarySize = 0;
            const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
            CALL_QNN(QNN::gContext.systemInterface.systemContextGetBinaryInfo(systemContextHandle, buffer, size, &binaryInfo,&binarySize));
            copyMetadataToGraphsInfo(binaryInfo, mGraphsInfo, mGraphCount);
            if (QNN_SUCCESS != QNN::gContext.systemInterface.systemContextFree(systemContextHandle)) {
                MNN_ERROR("Could not free system context handle.");
                return false;
            }
        }


        // 2. Retrieve graphs.
        {
            auto error = QNN::gContext.interface.contextValidateBinary(QNN::gContext.backendHandle, QNN::gContext.deviceHandle, mQnnContextConfig, buffer, size);
            if (QNN_SUCCESS != error) {
                MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
                return false;
            }
            CALL_QNN(QNN::gContext.interface.contextCreateFromBinary(QNN::gContext.backendHandle, QNN::gContext.deviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, nullptr));

            mQnnGraphHandleVec.resize(mGraphCount, nullptr);
            // [此处排序]
            for (int i = 0; i < mGraphCount; i++) {
                CALL_QNN(QNN::gContext.interface.graphRetrieve(mQnnContextHandle, mGraphsInfo[i]->graphName, &(mQnnGraphHandleVec[i])));
            }
        }

        MNNUnmapFile(buffer, file);
        MNNCloseFile(file);

        return true;
    }

    void invokModel(const std::vector<std::pair<const MNN::Tensor *, std::string>>& inputs, std::vector<std::pair<const MNN::Tensor *, std::string>>& outputs, const std::string & graphName) {
        // 根据graphName索引对应的graph
        GraphInfo* graph = nullptr;
        Qnn_GraphHandle_t qnnGraphHandle = nullptr;

        for (int i = 0; i < mGraphCount; i++) {
            #ifdef QNN_VERBOSE
            MNN_PRINT("graph name: %s %s\n", graphName.c_str(),mGraphsInfo[i]->graphName );
            #endif
            if (graphName == std::string(mGraphsInfo[i]->graphName)) {
                graph = mGraphsInfo[i];
                qnnGraphHandle = mQnnGraphHandleVec[i];
                break;
            }
        }
        MNN_ASSERT(graph && qnnGraphHandle);

        //MNN_PRINT("%s, Input:%d, output:%d\n", mPath.c_str(), inputs.size(), outputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            auto t = inputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numInputTensors; ++j) {
                auto& dstT = graph->inputTensors[j];
                #ifdef QNN_VERBOSE
                MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
                #endif
                if (inputs[i].second == dstT.v1.name) {
                    dstT.v1.clientBuf.data = t->host<void>();
                    if (t->getType().code == halide_type_float) {
                        dstT.v1.clientBuf.dataSize = t->usize() / 2;
                    } else {
                        dstT.v1.clientBuf.dataSize = t->usize();
                    }
                    find = true;
                    break;
                }
            }
            if (!find) {
                FUNC_PRINT(i);
            }
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto t = outputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numOutputTensors; ++j) {
                auto& dstT = graph->outputTensors[j];
                #ifdef QNN_VERBOSE
                MNN_PRINT("output name: %s %s\n", outputs[i].second.c_str(), dstT.v1.name);
                #endif
                if (outputs[i].second == dstT.v1.name) {
                    dstT.v1.clientBuf.data = t->host<void>();
                    if (t->getType().code == halide_type_float) {
                        dstT.v1.clientBuf.dataSize = t->usize() / 2;
                    } else {
                        dstT.v1.clientBuf.dataSize = t->usize();
                    }
                    find = true;
                    break;
                }
            }
            if (!find) {
                FUNC_PRINT(i);
            }
        }
        CALL_QNN(QNN::gContext.interface.graphExecute(qnnGraphHandle, graph->inputTensors,graph->numInputTensors, graph->outputTensors, graph->numOutputTensors, nullptr, nullptr));
    }
};
class PluginExecuteRaw : public CPUComputeKernel {
private:
    std::unique_ptr<RawExecutorWrapper> mRawExecutor;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mInputs;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mOutputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealInputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealOutputs;
public:
    ~ PluginExecuteRaw() {
        mRealInputs.clear();
        mRealOutputs.clear();
        mRawExecutor.reset();
    }
    bool init(CPUKernelContext* ctx) override {
        // TODO: Support relative path
        auto path = ctx->dir_path() + ctx->getAttr("path")->s()->str();
        #ifdef QNN_VERBOSE
        MNN_PRINT("MNN QNN Model dir:%s\nFile path:%s\n", ctx->dir_path().c_str(), path.c_str());
        #endif
        auto inputs = ctx->getAttr("inputs")->list();
        auto inputTensor = ctx->inputs();
        MNN_ASSERT(inputs->s()->size() == inputTensor.size());
        mInputs.resize(inputs->s()->size());
        mRealInputs.resize(inputTensor.size());
        for (int i=0; i<inputs->s()->size(); ++i) {
            mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::TENSORFLOW));
            mInputs[i].second = inputs->s()->GetAsString(i)->str();
            mInputs[i].first = mRealInputs[i].get();
        }
        auto outputs = ctx->getAttr("outputs")->list();
        auto outputTensor = ctx->outputs();
        mOutputs.resize(outputs->s()->size());
        MNN_ASSERT(outputs->s()->size() == outputTensor.size());
        mRealOutputs.resize(outputTensor.size());
        for (int i=0; i<outputs->s()->size(); ++i) {
            mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::TENSORFLOW));
            mOutputs[i].second = outputs->s()->GetAsString(i)->str();
            mOutputs[i].first = mRealOutputs[i].get();
        }
        mRawExecutor.reset(new RawExecutorWrapper(mInputs.size(), mOutputs.size()));

        return mRawExecutor->compileModel(path);
    }
    bool compute(CPUKernelContext* ctx) override {
        int shapeIndex = 0;
        if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
            return false;
        }
        std::string graphName = ctx->getAttr("allGraphName")->list()->s()->GetAsString(shapeIndex)->str();

        #ifdef QNN_VERBOSE
        MNN_PRINT("Graph name:%s, %d\n", graphName.c_str(), shapeIndex);
        #endif
        auto inputTensor = ctx->inputs();
        auto outputTensor = ctx->outputs();
        // 这里是默认了ctx->inputs()的顺序和ctx->getAttr("inputs")的顺序是一致的
        for (int i=0; i<mInputs.size(); ++i) {
            CPUTensorConverter::convert(inputTensor[i], mRealInputs[i].get());
            if (inputTensor[i]->getType().code == halide_type_float) {
                FLOAT_TO_HALF(mRealInputs[i]->host<float>(), mRealInputs[i]->host<int16_t>(), mRealInputs[i]->elementSize());
            }
        }
        mRawExecutor->invokModel(mInputs, mOutputs, graphName);
        for (int i=0; i<mOutputs.size(); ++i) {
            if (mRealOutputs[i]->getType().code == halide_type_float) {
                ::memcpy(mRealOutputs[i]->host<int16_t>() + mRealOutputs[i]->elementSize(), mRealOutputs[i]->host<int16_t>(),  mRealOutputs[i]->elementSize() * sizeof(int16_t));
                HALF_TO_FLOAT(mRealOutputs[i]->host<int16_t>() + mRealOutputs[i]->elementSize(), mRealOutputs[i]->host<float>(), mRealOutputs[i]->elementSize());
            }
            CPUTensorConverter::convert(mRealOutputs[i].get(), outputTensor[i]);
        }
        return true;
    }
};
} // namespace backend
}
}

#endif

namespace MNN {
namespace QNN {
// #define QNN_PROFILE_OP
// #define QNN_PROFILE_SUMMARIZE
// #define QNN_VERBOSE
QnnBackend::QnnBackend(const QnnRuntime* runtime) : Backend(MNNForwardType::MNN_FORWARD_NN), mPower(runtime->mPower) {
    mRuntime = runtime;
    mUseFP16 = (runtime->mPrecision != BackendConfig::Precision_High) ? true : false;
    mPerf = QNNPerf::create(&mRuntime->mQnnInterface);
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }

    // Set mQnnGraphConfig.
    mQnnHtpGraphCustomConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    mQnnHtpGraphCustomConfig.precision = QNN_PRECISION_FLOAT16;
    mQnnGraphConfig.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    mQnnGraphConfig.customConfig = &mQnnHtpGraphCustomConfig;
}

QnnBackend::~QnnBackend() {
    clean();
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBalanced();
    }
}

static inline std::map<OpType, QnnBackend::Creator*>* getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, QnnBackend::Creator*>* ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, QnnBackend::Creator*>; });
    return ret;
}

Execution* QnnBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    MNN_ASSERT(op != nullptr);
    auto map = getCreatorMap();
    auto iter = map->find(op->type());

    // MNN_PRINT("MNN_QNN::onCreate Type %d, Name %s.\n", op->type(), op->name()->c_str());

    if (iter == map->end()) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Not registered type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Not registered type %d.\n", op->type());
        }
        return nullptr;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    if (nullptr == exe) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Don't support type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Don't support type %d.\n", op->type());
        }
        return nullptr;
    }

    return exe;
}

bool QnnBackend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("MNN_QNN: %d type has be added.\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}


void QnnBackend::onExecuteBegin() const {
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    return;
}

#ifdef QNN_PROFILE_SUMMARIZE
static std::string getOpTypeFromName(const std::string& nodeName) {
    // The pattern is usually "OpType_..."
    size_t pos = nodeName.find('_');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // Fallback for names without '_', like "Input OpId_2 (cycles)"
    pos = nodeName.find(' ');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // If no delimiter is found, return the whole name as the type
    return nodeName;
}
#endif

void QnnBackend::startProfile() const{
#ifdef QNN_PROFILE_OP
    if (mQnnProfileHandle) {
        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = mRuntime->mQnnInterface.profileGetEvents(mQnnProfileHandle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        MNN_PRINT("\n--- QNN Node-level Performance Report ---\n");
        bool foundNodeData = false;

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            mRuntime->mQnnInterface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) {
                MNN_PRINT("Found EXECUTE event. Total time: %llu us. Querying sub-events...\n", (unsigned long long)eventData.value);
                
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;

                // 3. GetSubEvents
                auto get_sub_err = mRuntime->mQnnInterface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) {
                    MNN_PRINT("[QNN Profile] Failed to get sub-events for EXECUTE event. Error: %d\n", (int)get_sub_err);
                    continue;
                }

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    mRuntime->mQnnInterface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        foundNodeData = true;
                        const char* nodeName = subEventData.identifier;
                        uint64_t value = subEventData.value;

                        switch (subEventData.unit) {
                            case QNN_PROFILE_EVENTUNIT_MICROSEC:
                                MNN_PRINT("Node: %-45s | Time: %10llu us (%.3f ms)\n", 
                                        nodeName, (unsigned long long)value, (double)value / 1000.0);
                                break;
                            case QNN_PROFILE_EVENTUNIT_CYCLES:
                                MNN_PRINT("Node: %-45s | Cycles: %.2f*10^6\n", nodeName, (double)value / 1000000.0);
                                break;
                            // ... other dealing ...
                            default:
                                MNN_PRINT("Node: %-45s | Value: %10llu (Unit: %u - Unknown)\n",
                                        nodeName, (unsigned long long)value, subEventData.unit);
                                break;
                        }
                    }
                }
            }
        }

        if (!foundNodeData) {
            MNN_PRINT("No node-specific performance data found. Please ensure you have set:\n");
            MNN_PRINT("1. Profile level to QNN_PROFILE_LEVEL_DETAILED.\n");
            MNN_PRINT("2. HTP graph config with QNN_HTP_GRAPH_CONFIG_OPTION_PERF_PROFILE (if available).\n");
        }
        MNN_PRINT("-----------------------------------------\n");
    }
#endif

#ifdef QNN_PROFILE_SUMMARIZE
    if (mQnnProfileHandle) {
        std::map<std::string, uint64_t> opCycleStats;
        uint64_t totalNodeCycles = 0;

        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = mRuntime->mQnnInterface.profileGetEvents(mQnnProfileHandle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            mRuntime->mQnnInterface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) { // == QNN_PROFILE_EVENTTYPE_EXECUTE) {
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;
                auto get_sub_err = mRuntime->mQnnInterface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) continue;

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    mRuntime->mQnnInterface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        if (subEventData.identifier) {
                            std::string opType = getOpTypeFromName(subEventData.identifier);
                            opCycleStats[opType] += subEventData.value;
                            totalNodeCycles += subEventData.value;
                        }
                    }
                }
            }
        }
        
        if (!opCycleStats.empty()) {
            MNN_PRINT("\n--- QNN Operator-wise Performance Summary ---\n");
            MNN_PRINT("%-20s | %15s | %s\n", "Operator Type", "Total Cycles", "Percentage");
            MNN_PRINT("--------------------------------------------------\n");

            std::vector<std::pair<std::string, uint64_t>> sortedStats(opCycleStats.begin(), opCycleStats.end());
            std::sort(sortedStats.begin(), sortedStats.end(), [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
                return a.second > b.second; // sort by large -> small
            });

            for (const auto& pair : sortedStats) {
                double percentage = (totalNodeCycles > 0) ? ((double)pair.second * 100.0 / totalNodeCycles) : 0.0;
                MNN_PRINT("%-20s | %15llu | %.2f%%\n", pair.first.c_str(), pair.second, percentage);
            }
            MNN_PRINT("--------------------------------------------------\n");
            MNN_PRINT("%-20s | %15llu | 100.00%%\n", "Total", totalNodeCycles);
        }
    }
    // =========================================================
#endif
}

void QnnBackend::onExecuteEnd() const {
    executeGraph();
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBalanced();
    }
    startProfile();
    return;
}

void QnnBackend::onResizeBegin() {
    clean();
    createContextAndGraph();
    return;
}

ErrorCode QnnBackend::onResizeEnd() {
    #ifdef QNN_VERBOSE
    MNN_PRINT("start finalize\n");
    #endif
    buildOutputDequant();
    finalizeGraph();
    for(auto func : mReleaseFunc){
        func();
    }
    mReleaseFunc.clear();
    #ifdef QNN_VERBOSE
    MNN_PRINT("end finalize\n");
    #endif
    return NO_ERROR;
}


Backend::MemObj* QnnBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);
    if (TensorUtils::getDescribe(tensor)->index >= 0) {
        tName = std::string("t") + std::to_string(TensorUtils::getDescribe(tensor)->index);
    }
    
    bool isInput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    MNN_ASSERT(!isConst);

    Qnn_TensorType_t tType = QNN_TENSOR_TYPE_NATIVE;
    if (isInput) {
        tType = QNN_TENSOR_TYPE_APP_WRITE;
    }
    if (isOutput) {
        tType = QNN_TENSOR_TYPE_APP_READ;
    }

    Qnn_DataType_t tDataType;
    Qnn_QuantizeParams_t tQuantizeParams{};
    tQuantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    Qnn_ScaleOffset_t tScaleOffsetEncoding;
    tScaleOffsetEncoding.scale = 0.0f;
    tScaleOffsetEncoding.offset = 0;
    auto quant = TensorUtils::getDescribe(tensor)->quantAttr.get();
    //MNN_ASSERT((tensor->getType().code == halide_type_float) || (tensor->getType().code == halide_type_int && tensor->getType().bits == 32));
    if (mUseFP16 && tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_16;
    } else if (tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_32;
    } else if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
        tDataType = QNN_DATATYPE_INT_32;
    } else {
        MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
        return nullptr;
    }
    if(quant != nullptr && TensorUtils::getDescribe(tensor)->type == DataType_DT_INT8) {
        tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        tScaleOffsetEncoding.scale = quant->scale;
        if(quant->zero != 0){
            MNN_PRINT("MNN_QNN: Not supported asymmetric quant in <QnnBackend::onAcquire>.\n");
            return nullptr;
        }
        tDataType = QNN_DATATYPE_SFIXED_POINT_8;
        if (isOutput) {
            tType = QNN_TENSOR_TYPE_NATIVE;
        }
    }
    tQuantizeParams.scaleOffsetEncoding = tScaleOffsetEncoding;

    std::unique_ptr<Tensor> tempTensor(new Tensor(tensor, gQnnTensorDimType, false));
    std::vector<int> tDims;
    if (!(tempTensor->shape().empty())) {
        tDims = tempTensor->shape();
    } else {
        tDims = {1};
    }

    std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper = QNNTensorWrapper::create(tName, tType, tDataType, tDims, tQuantizeParams);

    Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
    mQNNTensorWrappers.push_back(qnnTensorWrapper);
    mTensorMap.insert({TensorUtils::getDescribe(tensor), mTensorCounter});

    if (isInput) {
        mInputTensorIndexes.push_back(mTensorCounter);
        qnnTensorWrapper->alloc();
    }
    if (isOutput) {
        if(quant != nullptr && TensorUtils::getDescribe(tensor)->type == DataType_DT_INT8){
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, gQnnTensorDimType));
            tName = "QnnTensor_" + std::to_string(mTensorCounter);
            tType = QNN_TENSOR_TYPE_APP_READ;
            if (mUseFP16 && tensor->getType().code == halide_type_float) {
                tDataType = QNN_DATATYPE_FLOAT_16;
            } else if (tensor->getType().code == halide_type_float) {
                tDataType = QNN_DATATYPE_FLOAT_32;
            } else {
                MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnOutputTensorWrapper = QNNTensorWrapper::create(tName, tType, tDataType, tDims, tQuantizeParamstmp);
            CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnOutputTensorWrapper->getNativeTensor()));
            mDeQuantOutputTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnOutputTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mOutputTensorIndexes.push_back(mTensorCounter);
            qnnOutputTensorWrapper->alloc();
        } else{
            mOutputTensorIndexes.push_back(mTensorCounter);
            qnnTensorWrapper->alloc();
        }
    }

    mTensorCounter += 1;
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif
    return new Backend::MemObj();
}


bool QnnBackend::onClearBuffer() {
    return true;
}


void QnnBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    bool isInput = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    // MNN_ASSERT(!isConst);

    if (isConst) {
        MNN_ASSERT(isInput);
    }

    MNN_ASSERT(isInput || isOutput);

    if (isInput) {
        inputIO(srcTensor, dstTensor);
    } else if (isOutput) {
        outputIO(srcTensor, dstTensor);
    } else {
        // Not support.
    }
}

void QnnBackend::inputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    int dstIndex = getTensorIdx(dstTensor);
    std::shared_ptr<QNNTensorWrapper> dstQnnTensorWrapper = mQNNTensorWrappers[dstIndex];
    std::shared_ptr<Tensor> dstDataContainer = dstQnnTensorWrapper->getDataContainer();

    bool valid0 = srcTensor->getType().code == halide_type_float;
    bool valid1 = srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    if (valid1) {
        auto code = CPUTensorConverter::convert(srcTensor, dstDataContainer.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
        return;
    }

    if (mUseFP16) {
        std::unique_ptr<Tensor> stageTensor(Tensor::create<float>(dstDataContainer->shape(), nullptr, TensorUtils::getDimType(dstDataContainer.get())));
        auto code = CPUTensorConverter::convert(srcTensor, stageTensor.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
        FLOAT_TO_HALF(stageTensor->host<float>(), dstDataContainer->host<int16_t>(), dstDataContainer->elementSize());
    } else {
        auto code = CPUTensorConverter::convert(srcTensor, dstDataContainer.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}

void QnnBackend::outputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto iter = mDeQuantOutputTensorMap.find(TensorUtils::getDescribe(srcTensor));
    int srcIndex = -1;
    if(iter != mDeQuantOutputTensorMap.end()){
        srcIndex = getTensorIdx(iter->second.second.get());
    } else{
        srcIndex = getTensorIdx(srcTensor);
    }
    std::shared_ptr<QNNTensorWrapper> srcQnnTensorWrapper = mQNNTensorWrappers[srcIndex];
    std::shared_ptr<Tensor> srcDataContainer = srcQnnTensorWrapper->getDataContainer();

    // Currently, support float output only.
    bool valid0 = dstTensor->getType().code == halide_type_float;
    bool valid1 = dstTensor->getType().code == halide_type_int && dstTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    if (valid1) {
        auto code = CPUTensorConverter::convert(srcDataContainer.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
        return;
    }

    if (mUseFP16) {
        std::unique_ptr<Tensor> stageTensor(Tensor::create<float>(srcDataContainer->shape(), nullptr, TensorUtils::getDimType(srcDataContainer.get())));
        HALF_TO_FLOAT(srcDataContainer->host<int16_t>(), stageTensor->host<float>(), srcDataContainer->elementSize());
        auto code = CPUTensorConverter::convert(stageTensor.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    } else {
        auto code = CPUTensorConverter::convert(srcDataContainer.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}
bool QnnBackend::useCache() const {
    return mRuntime->mUseCache;
}

void QnnBackend::createContextAndGraph() {
    mRuntime->allocContext();
    const QnnGraph_Config_t * pGraphConfig[] = {&mQnnGraphConfig, nullptr};
    if (mRuntime->mUseCache) {
        CALL_QNN(mRuntime->mQnnInterface.graphRetrieve(mRuntime->mQnnContextHandle, mQnnGraphName.c_str(), &mQnnGraphHandle));
    } else {
        CALL_QNN(mRuntime->mQnnInterface.graphCreate(mRuntime->mQnnContextHandle, mQnnGraphName.c_str(), pGraphConfig, &mQnnGraphHandle));
    }
    MNN_ASSERT(mQnnGraphHandle != nullptr);
}

void QnnBackend::finalizeGraph() {
    // [TODO] Fix this. Add the following branch for empty resize.
    if (mTensorCounter == 0) {
        return;
    }
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif

    #if defined(QNN_PROFILE_SUMMARIZE) || defined(QNN_PROFILE_OP)
    if (mQnnProfileHandle == nullptr) {
        // set QNN_PROFILE_LEVEL_DETAILED
        QnnProfile_Level_t profileLevel = QNN_PROFILE_LEVEL_DETAILED;
        MNN_PRINT("[QNN Profile] Creating QNN Profile Handle with DETAILED level.\n");
        auto profile_err = mRuntime->mQnnInterface.profileCreate(mRuntime->mQnnContextHandle, profileLevel, &mQnnProfileHandle);
        if (profile_err != QNN_SUCCESS || mQnnProfileHandle == nullptr) {
            MNN_ERROR("[QNN Profile] Failed to create QNN Profile Handle, error: %d\n", (int)profile_err);
            mQnnProfileHandle = nullptr;
        }
    }
    #endif
    CALL_QNN(mRuntime->mQnnInterface.graphFinalize(mQnnGraphHandle, mQnnProfileHandle, mQnnSignalHandle));
}

void QnnBackend::executeGraph() const {
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    for (int i = 0; i <  mInputTensorIndexes.size(); i++) {
        inputs.push_back(*(mQNNTensorWrappers[mInputTensorIndexes[i]]->getNativeTensor()));
    }
    for (int j = 0 ; j < mOutputTensorIndexes.size(); j++) {
        outputs.push_back(*(mQNNTensorWrappers[mOutputTensorIndexes[j]]->getNativeTensor()));
    }

    CALL_QNN(mRuntime->mQnnInterface.graphExecute(mQnnGraphHandle, inputs.data(), mInputTensorIndexes.size(), outputs.data(), mOutputTensorIndexes.size(), mQnnProfileHandle, mQnnSignalHandle));
}

void QnnBackend::freeContextAndGraph() {
    if (mTensorCounter != 0) {
        mQnnGraphHandle = nullptr;
    }
    mRuntime->freeContext();
}

void QnnBackend::addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs) {
    MNN_ASSERT(nodeName != nullptr && packageName != nullptr && nodeType != nullptr && !(inputs.empty()) && !(outputs.empty()));

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = version;
    opConfig.v1.name = nodeName;
    opConfig.v1.packageName = packageName;
    opConfig.v1.typeName = nodeType;
    opConfig.v1.numOfParams = params.size();
    opConfig.v1.params = params.data();
    opConfig.v1.numOfInputs = inputs.size();
    opConfig.v1.inputTensors = inputs.data();
    opConfig.v1.numOfOutputs = outputs.size();
    opConfig.v1.outputTensors = outputs.data();

    CALL_QNN(mRuntime->mQnnInterface.backendValidateOpConfig(mRuntime->mQnnBackendHandle, opConfig));

    CALL_QNN(mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig));
}

int QnnBackend::getTensorIdx(const Tensor * tensor) const {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    int idx = -1;
    if (iter == mTensorMap.end()) {
        std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);;
        if (TensorUtils::getDescribe(tensor)->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
            MNN_PRINT("Tensor usage is %d.\n", (int) TensorUtils::getDescribe(tensor)->usage);
        }
        #ifdef QNN_VERBOSE
        MNN_PRINT("qnn tenor usage:%d, dimension:%d\n", TensorUtils::getDescribe(tensor)->usage, tensor->dimensions());
        #endif
        MNN_ASSERT(TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        // MNN_ASSERT(tensor->dimensions() <= 2);
        std::vector<uint32_t> tDims = getNHWCShape(tensor);
        Qnn_DataType_t tDataType;
        std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper;
        if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
            tDataType = QNN_DATATYPE_INT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticTensor(tName, tDataType, tDims, tensor->host<int>());
        } else if (tensor->getType().code == halide_type_float) {
            tDataType = mUseFP16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticFloatTensor(tName, tDataType, tDims, tensor->host<float>());
        } else {
            MNN_ASSERT(false);
        }
        Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
        CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
        mQNNTensorWrappers.push_back(qnnTensorWrapper);
        mTensorMap.insert({tensorKey, mTensorCounter});
        idx = mTensorCounter;
        mTensorCounter += 1;
    } else {
        idx = iter->second;
    }
    return idx;
}

void QnnBackend::addStaticTensorToGraph(Qnn_Tensor_t * staticTensor) {
    MNN_ASSERT(staticTensor->v1.type == QNN_TENSOR_TYPE_STATIC);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, staticTensor));
}

void QnnBackend::addStageTensorToGraph(Qnn_Tensor_t * stageTensor) {
    MNN_ASSERT(stageTensor->v1.type == QNN_TENSOR_TYPE_NATIVE);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, stageTensor));
}

Qnn_Tensor_t * QnnBackend::getNativeTensor(const Tensor * tensor) {
    int idx = getTensorIdx(tensor);
    return mQNNTensorWrappers[idx]->getNativeTensor();
}

std::shared_ptr<QNNTensorWrapper> QnnBackend::getTensorWrapper(const Tensor * tensor) {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    MNN_ASSERT(iter != mTensorMap.end());
    return mQNNTensorWrappers[iter->second];
}

bool QnnBackend::getUseFP16() const {
    return mUseFP16;
}

void QnnBackend::clean() {
    if (mQnnProfileHandle) {
        mRuntime->mQnnInterface.profileFree(mQnnProfileHandle);
        mQnnProfileHandle = nullptr;
    }
    freeContextAndGraph(); // This function must be called first.
    mTensorCounter = 0;
    mQNNTensorWrappers.clear();
    mTensorMap.clear();
    mInputTensorIndexes.clear();
    mOutputTensorIndexes.clear();
    mDeQuantOutputTensorMap.clear();
}
void QnnBackend::buildOutputDequant(){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mDeQuantOutputTensorMap){
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = "Dequantize_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));;
        mInputs.push_back(*(getNativeTensor(iter.second.first))); // input
        mOutputs.push_back(*(getNativeTensor(iter.second.second.get()))); // output
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

QnnRuntime::QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle) {
    // MNN_PRINT("QnnRuntime is constructing.\n");
    mInfo = info;
    // Default setting
    mPower = BackendConfig::Power_Normal;
    mMemory = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    // User setting
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
    }
    mQnnInterface = qnnInterface;
    mQnnLogHandle = qnnLogHandle;
    mQnnBackendHandle = qnnBackendHandle;
    mQnnDeviceHandle = qnnDeviceHandle;
}

QnnRuntime::~QnnRuntime() {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
    }
}
bool QnnRuntime::onSetCache(const void* buffer, size_t size) {
    // TODO: Fix bug and complete
    return false;
    if (nullptr == buffer) {
        return false;
    }
    auto error = mQnnInterface.contextValidateBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size);
    if (QNN_SUCCESS != error) {
        MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
        return false;
    }
    freeContext();
    CALL_QNN(mQnnInterface.contextCreateFromBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, nullptr));
    mUseCache = true;
    return true;
}
void QnnRuntime::allocContext() const {
    CALL_QNN(mQnnInterface.contextCreate(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, &mQnnContextHandle));
    MNN_ASSERT(mQnnContextHandle != nullptr);
}
void QnnRuntime::freeContext() const {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
        mBinaryBuffer.clear();
    }    
}

std::pair<const void*, size_t> QnnRuntime::onGetCache() {
    return std::make_pair(nullptr, 0);
    if (!mBinaryBuffer.empty()) {
        return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
    }
    if (nullptr == mQnnContextHandle) {
        return std::make_pair(nullptr, 0);
    }
    Qnn_ContextBinarySize_t size = 0;
    CALL_QNN(mQnnInterface.contextGetBinarySize(mQnnContextHandle, &size));
    FUNC_PRINT(size);
    if (0 == size) {
        return std::make_pair(nullptr, 0);
    }
    mBinaryBuffer.resize(size);
    Qnn_ContextBinarySize_t writesize = 0;
    CALL_QNN(mQnnInterface.contextGetBinary(mQnnContextHandle, mBinaryBuffer.data(), size, &writesize));
    return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
}

Backend* QnnRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new QnnBackend(this);
}

QnnRuntime* QnnRuntime::create(const Backend::Info& info) {
    // Create Interface.
    return new QnnRuntime(info, gContext.interface, gContext.logHandle, gContext.backendHandle, gContext.deviceHandle);
}

// Do nothing
void QnnRuntime::onGabageCollect(int level) {}

Runtime::CompilerType QnnRuntime::onGetCompilerType() const {
    return Compiler_Origin;
}

bool QnnRuntime::onSetCachePath(const char* path, int mode) {
#ifdef ENABLE_QNN_CONVERT_MODE
    MNN_ASSERT(path != nullptr);
    QNNConvertor::OutputDir = std::string(path);
#endif
    return true;
}

bool QnnRuntime::registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target) {
    if (QNN_GET_ERROR_CODE(qnnInterface.backendRegisterOpPackage(backendHandle, path.c_str(), interfaceProvider.c_str(), target.c_str())) != QNN_SUCCESS) {
        MNN_PRINT("MNN_QNN: Failed to register the Op Package: %s.\n", path.c_str());
        return false;
    }
    return true;
}

class QnnRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        return QnnRuntime::create(info);
    }
    virtual bool onValid(Backend::Info& info) const override {
        return true;
    }
    virtual bool onGetDeviceInfo(const std::string& deviceKey, std::string& deviceValue) const override {
        if(deviceKey == "soc_id" && gContext.soc_id != 0) {
            deviceValue = std::to_string(gContext.soc_id);
            return true;
        }
        if(deviceKey == "dsp_arch" && gContext.dsp_arch != 0) {
            deviceValue = "v" + std::to_string(gContext.dsp_arch);
            return true;
        }
        return false;
    }
};

} // end namespace QNN

void registerQNNRuntimeCreator() {
#ifndef ENABLE_QNN_CONVERT_MODE
    // check whether the qnn lib is available
    if (!QNN::loadQNNSymbol()) {
        return;
    }
#endif

    QNN_INTERFACE_VER_TYPE qnnInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    {
        QnnInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return;
        }
        bool foundValidInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return;
        }
    }
#else
    qnnInterface = QNN::gQnnConvertorInterface;
#endif

    // Create Log.
    Qnn_LogHandle_t logHandle = nullptr;
    {
        QnnLog_Callback_t logCallback = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, QNN_LOG_LEVEL_ERROR, &logHandle)) != QNN_SUCCESS) ||
            (logHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to initialize logging in the backend.\n");
            return;
        }
    }

    // Create Backend.
    Qnn_BackendHandle_t backendHandle = nullptr;
    {
        const QnnBackend_Config_t** backendConfig = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS) ||
            (backendHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to create the backend.\n");
            return;
        }
    }

    // Create Device.
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    QnnHtpDevice_Arch_t dspArch = QNN_HTP_DEVICE_ARCH_NONE;
    uint32_t socId = 0;
    {
        // Check whether the device API is supported.
        bool supportDevice = QNN::checkCapability(qnnInterface, QNN_PROPERTY_GROUP_DEVICE);
        if (supportDevice) {
            const QnnDevice_Config_t ** deviceConfig = nullptr;
            auto qnnStatus = qnnInterface.deviceCreate(logHandle, deviceConfig, &deviceHandle);
            if(qnnStatus != QNN_SUCCESS || (deviceHandle == nullptr)) {
                MNN_PRINT("MNN_QNN: Failed to create the device, error:%lu\n", (unsigned long)qnnStatus);
                return;
            }

            if (qnnInterface.deviceGetPlatformInfo == nullptr) {
                MNN_PRINT("[Warning]: No QnnDevice_getPlatformInfo API");
            } else {
                // QnnDevice_PlatformInfo_t platformInfo = QNN_DEVICE_PLATFORM_INFO_INIT;
                const QnnDevice_PlatformInfo_t* backendPlatformInfoPtr = nullptr;
                qnnStatus = qnnInterface.deviceGetPlatformInfo(logHandle, &backendPlatformInfoPtr);
                if(qnnStatus != QNN_SUCCESS || backendPlatformInfoPtr == nullptr) {
                    MNN_PRINT("[Warning]: deviceGetPlatformInfo Failed to query platform info");
                } else {
                    QnnDevice_HardwareDeviceInfo_t* hwDeviceInfo = backendPlatformInfoPtr->v1.hwDevices;
                    dspArch = hwDeviceInfo->v1.deviceInfoExtension->onChipDevice.arch;
                    socId = hwDeviceInfo->v1.deviceInfoExtension->onChipDevice.socModel;
                }
            }
        } else {
            MNN_PRINT("MNN_QNN: Not supporting device API.\n");
            return;
        }
    }

    // Create System Interface
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    #ifdef MNN_WITH_PLUGIN
    {
        QnnSystemInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return;
        }
        bool foundValidSystemInterface{false};
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_SYSTEM_API_VERSION_MAJOR == interfaceProviders[pIdx]->systemApiVersion.major &&
                QNN_SYSTEM_API_VERSION_MINOR <= interfaceProviders[pIdx]->systemApiVersion.minor) {
                foundValidSystemInterface = true;
                systemInterface = interfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidSystemInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return;
        }
    }
    #endif
#else
    systemInterface = QNN::gQnnConvertorSystemInterface;
#endif


    QNN::gContext.interface = qnnInterface;
    QNN::gContext.systemInterface = systemInterface;
    QNN::gContext.backendHandle = backendHandle;
    QNN::gContext.deviceHandle = deviceHandle;
    QNN::gContext.logHandle = logHandle;
    QNN::gContext.soc_id = socId;
    QNN::gContext.dsp_arch = dspArch;

    QNN::registerQNNOps();
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, new QNN::QnnRuntimeCreator, false);

#ifdef MNN_WITH_PLUGIN
    plugin::InferShapeKernelRegister::add("QNN", []() { // NOLINT
        return new plugin::shape_inference::PluginShapeRaw;               // NOLINT
    });
    plugin::ComputeKernelRegistry<plugin::backend::PluginExecuteRaw::KernelT>::add("QNN", []() {
        return new plugin::backend::PluginExecuteRaw;
    });
#endif
}

} // end namespace MNN
