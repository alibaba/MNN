#include "QNNConvertor.hpp"
#include "core/MNNFileUtils.h"
#include <cctype>

#define APPEND_VECTOR(vec1, vec2) (vec1.insert(vec1.end(), std::make_move_iterator(vec2.begin()), std::make_move_iterator(vec2.end())))
#define TENSOR_NAME_SYMBOL(cName) ("tensor_" + std::string(cName))
#define PARAM_NAME_SYMBOL(cName) ("param_" + std::string(cName) + "_" + nodeName)

namespace MNN {
namespace QNN {

std::string QNNConvertor::OutputDir = "";
std::string QNNTranslator::GraphNameSymbol = "";
FILE * QNNConvertor::CppFilePointer = nullptr;
std::string QNNConvertor::CppBuffer = "";
const uint32_t QNNConvertor::CppBufferSize = 4096; // 4KB

std::string GetLastDirName(const std::string& path) {
    if (path.empty()) {
        MNN_ERROR("MNN_QNN: Invalid output dir for QNNConvertor.\n");
        return "";
    }

    std::string result = path;

    // Remove '/' and '\\' at the end.
    while (!result.empty() && (result.back() == '/' || result.back() == '\\')) {
        result.pop_back();
    }
    if (result.empty()) {
        MNN_ERROR("MNN_QNN: Invalid output dir for QNNConvertor.\n");
        return "";
    }

    size_t lastSeparator = result.find_last_of("/\\");
    // No '/' and '\\' in the path.
    if (lastSeparator != std::string::npos) {
        result = result.substr(lastSeparator + 1);
    }

    // Check whether result is a legal cpp symbol.
    if (std::isdigit(result[0])) {
        MNN_ERROR("MNN_QNN: Invalid cache path.\n");
        return "";
    }
    for (size_t i = 0; i < result.size(); ++i) {
        if (!(std::isalpha(result[i]) || std::isdigit(result[i]) || result[i] == '_')) {
            MNN_ERROR("MNN_QNN: Invalid cache path.\n");
            return "";
        }
    }

    return result;
}

void QNNConvertor::RecordBegin(const char* graphName) {
    MNN_ASSERT(!(QNNConvertor::OutputDir.empty()));
    QNNTranslator::GraphNameSymbol = GetLastDirName(QNNConvertor::OutputDir);
    MNN_ASSERT(!(QNNTranslator::GraphNameSymbol.empty()));

    QNNConvertor::CppBuffer.reserve(QNNConvertor::CppBufferSize);

    std::string cppFilePath = MNNFilePathConcat(QNNConvertor::OutputDir, QNNTranslator::GraphNameSymbol + ".cpp");
    QNNConvertor::CppFilePointer = std::fopen(cppFilePath.c_str(), "w");
    if (!QNNConvertor::CppFilePointer) {
        MNN_ERROR("MNN_QNN: Failed to open file %s.\n", cppFilePath.c_str());
        return;
    }

    QNNCommand cmd = {};
    cmd.type = QNNCommandTypeBegin;

    QNNConvertor::Translate(cmd);

    return;
}

void QNNConvertor::RecordTensor(const Qnn_Tensor_t * tensor) {
    QNNCommand cmd;

    cmd.type = QNNCommandTypeTensor;

    const Qnn_TensorV1_t & t = tensor->v1;
    cmd.commandTensor.name = t.name;
    switch (t.type) {
        case QNN_TENSOR_TYPE_APP_WRITE:
            cmd.commandTensor.type = TENSOR_INPUT;
            break;
        case QNN_TENSOR_TYPE_APP_READ:
            cmd.commandTensor.type = TENSOR_OUTPUT;
            break;
        case QNN_TENSOR_TYPE_NATIVE:
            cmd.commandTensor.type = TENSOR_NATIVE;
            break;
        case QNN_TENSOR_TYPE_STATIC: {
            std::string sname(t.name);
            bool isParam = false;
            if (sname.size() >= 5) {
                isParam = (sname.compare(sname.size() - 5, 5, gParamMarker) == 0);
            }
            cmd.commandTensor.type = isParam ? TENSOR_PARAM : TENSOR_STATIC;
            break;
        }
        default:
            MNN_ERROR("MNN_QNN: Unknown Qnn_Convertor_Tensor_t.\n");
            return;
    }
    cmd.commandTensor.dataType = t.dataType;
    cmd.commandTensor.quantizeParams = t.quantizeParams;
    cmd.commandTensor.rank = t.rank;
    cmd.commandTensor.dimensions = t.dimensions;
    cmd.commandTensor.clientBuf = t.clientBuf;

    QNNConvertor::Translate(cmd);

    if (cmd.commandTensor.type == Qnn_Convertor_Tensor_t::TENSOR_STATIC) {
        QNNConvertor::DumpBuffer(cmd.commandTensor.name, cmd.commandTensor.clientBuf.data, cmd.commandTensor.clientBuf.dataSize);
    }

    return;
}

void QNNConvertor::RecordNode(const Qnn_OpConfig_t & opConfig) {
    QNNCommand cmd;

    cmd.type = QNNCommandTypeNode;

    const Qnn_OpConfigV1_t & op = opConfig.v1;

    cmd.commandNode.name = op.name;
    cmd.commandNode.packageName = op.packageName;
    cmd.commandNode.typeName = op.typeName;
    cmd.commandNode.numOfParams = op.numOfParams;
    cmd.commandNode.params = op.params;
    cmd.commandNode.numOfInputs = op.numOfInputs;
    cmd.commandNode.inputTensors = op.inputTensors;
    cmd.commandNode.numOfOutputs = op.numOfOutputs;
    cmd.commandNode.outputTensors = op.outputTensors;

    QNNConvertor::Translate(cmd);

    return;
}

void QNNConvertor::RecordEnd() {
    QNNCommand cmd = {};
    cmd.type = QNNCommandTypeEnd;

    QNNConvertor::Translate(cmd);

    if (std::fclose(QNNConvertor::CppFilePointer) != 0) {
        MNN_ERROR("MNN_QNN: Failed to close the cpp file for QNNConvertor.\n");
    }

    return;
}

void QNNConvertor::Translate(const QNNCommand & cmd) {
    std::vector<std::string> cppLines = QNNTranslator::TranslateCommand(cmd);
    for (const std::string& line : cppLines) {
        QNNConvertor::CppBuffer.append(line);
        QNNConvertor::CppBuffer.push_back('\n');
    }
    size_t written = std::fwrite(QNNConvertor::CppBuffer.data(), 1, QNNConvertor::CppBuffer.size(), QNNConvertor::CppFilePointer);
    if (written != QNNConvertor::CppBuffer.size()) {
        MNN_ERROR("MNN_QNN: Failed to write to the Cpp File of QNNConvertor.\n");
    }
    QNNConvertor::CppBuffer.clear();
    return;
}

void QNNConvertor::DumpBuffer(const char * name, const void * buffer, size_t size) {
    std::string dataPath = MNNFilePathConcat(QNNConvertor::OutputDir, std::string(name) + ".raw");

    FILE* fp = std::fopen(dataPath.c_str(), "wb");
    if (!fp) {
        MNN_ERROR("MNN_QNN: Failed to open file %s.\n", dataPath.c_str());
        return;
    }

    size_t written = std::fwrite(buffer, 1, size, fp);
    if (written != size) {
        MNN_ERROR("MNN_QNN: Failed to write to file %s. Written: %zu, Expected: %zu\n", dataPath.c_str(), written, size);
    }

    int code = std::fclose(fp);
    if (code != 0) {
        MNN_ERROR("MNN_QNN: Failed to close file %s.\n", dataPath.c_str());
    }

    return;
}

std::vector<std::string> QNNTranslator::TranslateCommand(const QNNCommand & cmd) {
    switch (cmd.type) {
        case QNNCommandTypeBegin:
            return QNNTranslator::TranslateBegin();
        case QNNCommandTypeTensor:
            return QNNTranslator::TranslateTensor(cmd.commandTensor);
        case QNNCommandTypeNode:
            return QNNTranslator::TranslateNode(cmd.commandNode);
        case QNNCommandTypeEnd:
            return QNNTranslator::TranslateEnd();
        default:
            MNN_PRINT("MNN_QNN: Unknown QNNCommandType.\n");
            return {};
    }
}

std::vector<std::string> QNNTranslator::TranslateBegin() {
    std::vector<std::string> result;

    result.push_back("#include \"QnnModel.hpp\"");
    result.push_back("#include \"QnnOpDef.h\"");
    result.push_back("");
    result.push_back("// Flag to determine if Backend should node validation for each opNode added");
    result.push_back("#define DO_GRAPH_NODE_VALIDATIONS 1");
    result.push_back("");
    result.push_back("using namespace qnn_wrapper_api;");
    result.push_back("extern \"C\" {");
    result.push_back("QNN_API");
    result.push_back("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    result.push_back("                                    QNN_INTERFACE_VER_TYPE interface,");
    result.push_back("                                    Qnn_ContextHandle_t contextHandle,");
    result.push_back("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    result.push_back("                                    const uint32_t numGraphsConfigInfo,");
    result.push_back("                                    GraphInfoPtr_t** graphsInfo,");
    result.push_back("                                    uint32_t* numGraphsInfo,");
    result.push_back("                                    bool debug,");
    result.push_back("                                    QnnLog_Callback_t logCallback,");
    result.push_back("                                    QnnLog_Level_t maxLogLevel) {");
    result.push_back("  ModelError_t err = MODEL_NO_ERROR;");
    result.push_back("");
    result.push_back("  /* model/graph for " + QNNTranslator::GraphNameSymbol + "*/");
    result.push_back("  QnnModel " + QNNTranslator::GraphNameSymbol + ";");
    result.push_back("  const QnnGraph_Config_t** graphConfigs = nullptr;");
    result.push_back("  VALIDATE(getQnnGraphConfigFromInfo(");
    result.push_back("               \"" + QNNTranslator::GraphNameSymbol + "\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs),");
    result.push_back("           err);");
    result.push_back("  VALIDATE(" + QNNTranslator::GraphNameSymbol + ".initialize(backendHandle,");
    result.push_back("                                    interface,");
    result.push_back("                                    contextHandle,");
    result.push_back("                                    \"" + QNNTranslator::GraphNameSymbol + "\",");
    result.push_back("                                    debug,");
    result.push_back("                                    DO_GRAPH_NODE_VALIDATIONS,");
    result.push_back("                                    graphConfigs),");
    result.push_back("           err);");
    result.push_back("");

    return result;
}

std::vector<std::string> QNNTranslator::TranslateTensor(const QNNCommandTensor& cmdT) {
    std::string sName = cmdT.name;
    std::string tensorNameSymbol = TENSOR_NAME_SYMBOL(cmdT.name);
    std::string dimensionsNameSymbol = "dimensions_" + sName;
    std::string dataNameSymbol = "data_" + sName;

    bool isParam = (cmdT.type == Qnn_Convertor_Tensor_t::TENSOR_PARAM) ? true : false;
    bool hasClientBuf = (cmdT.clientBuf.data != nullptr) ? true : false;
    bool hasQuant = (cmdT.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED) ? true : false;
    bool shouldBeAdded = (cmdT.type == Qnn_Convertor_Tensor_t::TENSOR_INPUT) || (cmdT.type == Qnn_Convertor_Tensor_t::TENSOR_STATIC);

    std::vector<std::string> result;

    result.push_back("");
    result.push_back("  // Adding Tensor for " + sName + ".");
    result.push_back(QNNTranslator::TranslateDimensionsArray(dimensionsNameSymbol, cmdT.rank, cmdT.dimensions));
    if (isParam) {
        result.push_back(QNNTranslator::TranslateParamDataArray(dataNameSymbol, cmdT.dataType, cmdT.clientBuf));
    }
    if(hasQuant){
        std::vector<std::string> linesQuantScaleOffset = TranslateQuantizeScaleOffsetDataArray(tensorNameSymbol, cmdT.quantizeParams, cmdT.rank, cmdT.dimensions);
        APPEND_VECTOR(result, linesQuantScaleOffset);
    }
    result.push_back("  Qnn_Tensor_t " + tensorNameSymbol +   " = QNN_TENSOR_INIT;");
    result.push_back("  {");
    result.push_back("  " + tensorNameSymbol + ".version = QNN_TENSOR_VERSION_1;");
    result.push_back("  " + tensorNameSymbol + ".v1.id = 0;");
    result.push_back("  " + tensorNameSymbol + ".v1.name = \"" + sName +"\";");
    result.push_back("  " + tensorNameSymbol + ".v1.type = " + QNNTranslator::MapTensorType(cmdT.type) + ";");
    result.push_back("  " + tensorNameSymbol + ".v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;");
    result.push_back("  " + tensorNameSymbol + ".v1.dataType = " + QNNTranslator::MapDataType(cmdT.dataType) + ";");
    std::vector<std::string> linesQuant = QNNTranslator::TranslateTensorQuantizeParams(tensorNameSymbol, cmdT.quantizeParams);
    APPEND_VECTOR(result, linesQuant);
    result.push_back("  " + tensorNameSymbol + ".v1.rank = " + std::to_string(cmdT.rank) + ";");
    result.push_back("  " + tensorNameSymbol + ".v1.dimensions = " + dimensionsNameSymbol + ";");
    result.push_back("  " + tensorNameSymbol + ".v1.memType = QNN_TENSORMEMTYPE_RAW;");
    std::vector<std::string> linesClientBuf = QNNTranslator::TranslateTensorClientBuf(tensorNameSymbol, dataNameSymbol, sName, cmdT.clientBuf, hasClientBuf, isParam);
    APPEND_VECTOR(result, linesClientBuf);
    result.push_back("  }");

    if (shouldBeAdded) {
        result.push_back("  VALIDATE(" + QNNTranslator::GraphNameSymbol + ".addTensor(\"" + sName + "\", " + tensorNameSymbol + "), err);");
    }

    result.push_back("");

    return result;
}

std::vector<std::string> QNNTranslator::TranslateNode(const QNNCommandNode& cmdN) {
    std::string sName = cmdN.name;
    std::string paramArraySymbol = "params_" + sName;
    std::string inputArraySymbol = "inputs_" + sName;
    std::string outputArraySymbol = "outputs_" + sName;

    std::vector<std::string> result;

    result.push_back("");
    result.push_back("  // Adding Node for " + sName + ".");

    std::vector<std::string> linesParamArray = QNNTranslator::TranslateNodeParamArray(sName, paramArraySymbol, cmdN.numOfParams, cmdN.params);
    APPEND_VECTOR(result, linesParamArray);
    std::vector<std::string> linesInputArray = QNNTranslator::TranslateNodeInputArray(inputArraySymbol, cmdN.numOfInputs, cmdN.inputTensors);
    APPEND_VECTOR(result, linesInputArray);
    std::vector<std::string> linesOutputArray = QNNTranslator::TranslateNodeOutputArray(outputArraySymbol, cmdN.numOfOutputs, cmdN.outputTensors);
    APPEND_VECTOR(result, linesOutputArray);
    result.push_back("  VALIDATE(" + QNNTranslator::GraphNameSymbol + ".addNode(QNN_OPCONFIG_VERSION_1, \"" + sName + "\", \"" + std::string(cmdN.packageName) + "\", \"" + std::string(cmdN.typeName) + "\",");
    result.push_back("                                        " + paramArraySymbol + ", " + std::to_string(cmdN.numOfParams) + ",");
    result.push_back("                                        " + inputArraySymbol + ", " + std::to_string(cmdN.numOfInputs) + ",");
    result.push_back("                                        " + outputArraySymbol + ", " + std::to_string(cmdN.numOfOutputs) + "),");
    result.push_back("                                        err);");

    result.push_back("");

    return result;
}

std::vector<std::string> QNNTranslator::TranslateEnd() {
    std::vector<std::string> result;

    result.push_back("  // Add all models to array to get graphsInfo");
    result.push_back("  QnnModel* models[] = {&" + GraphNameSymbol + "};");
    result.push_back("  uint32_t numModels = 1;");
    result.push_back("");
    result.push_back("  // Populate the constructed graphs in provided output variables");
    result.push_back("  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
    result.push_back("  *numGraphsInfo = numModels;");
    result.push_back("");
    result.push_back("  return err;");
    result.push_back("");
    result.push_back("}  // PREPARE_GRAPHS");
    result.push_back("");
    result.push_back("QNN_API");
    result.push_back("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphs, uint32_t numGraphsInfo) {");
    result.push_back("  return qnn_wrapper_api::freeGraphsInfo(graphs, numGraphsInfo);");
    result.push_back("}  // FREEGRAPHINFO");
    result.push_back("}");

    return result;
}

std::string QNNTranslator::MapTensorType(Qnn_Convertor_Tensor_t type) {
    switch (type) {
        case TENSOR_INPUT:     return "QNN_TENSOR_TYPE_APP_WRITE";
        case TENSOR_STATIC:    return "QNN_TENSOR_TYPE_STATIC";
        case TENSOR_PARAM:     return "QNN_TENSOR_TYPE_STATIC";
        case TENSOR_NATIVE:    return "QNN_TENSOR_TYPE_NATIVE";
        case TENSOR_OUTPUT:    return "QNN_TENSOR_TYPE_APP_READ";
        default:               return "UNKNOWN_QNN_TENSOR_TYPE_T";
    }
}

std::string QNNTranslator::MapDataType(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_INT_8:              return "QNN_DATATYPE_INT_8";
        case QNN_DATATYPE_INT_16:             return "QNN_DATATYPE_INT_16";
        case QNN_DATATYPE_INT_32:             return "QNN_DATATYPE_INT_32";
        case QNN_DATATYPE_INT_64:             return "QNN_DATATYPE_INT_64";
        case QNN_DATATYPE_UINT_8:             return "QNN_DATATYPE_UINT_8";
        case QNN_DATATYPE_UINT_16:            return "QNN_DATATYPE_UINT_16";
        case QNN_DATATYPE_UINT_32:            return "QNN_DATATYPE_UINT_32";
        case QNN_DATATYPE_UINT_64:            return "QNN_DATATYPE_UINT_64";
        case QNN_DATATYPE_FLOAT_16:           return "QNN_DATATYPE_FLOAT_16";
        case QNN_DATATYPE_FLOAT_32:           return "QNN_DATATYPE_FLOAT_32";
        case QNN_DATATYPE_FLOAT_64:           return "QNN_DATATYPE_FLOAT_64";
        case QNN_DATATYPE_SFIXED_POINT_4:     return "QNN_DATATYPE_SFIXED_POINT_4";
        case QNN_DATATYPE_SFIXED_POINT_8:     return "QNN_DATATYPE_SFIXED_POINT_8";
        case QNN_DATATYPE_SFIXED_POINT_16:    return "QNN_DATATYPE_SFIXED_POINT_16";
        case QNN_DATATYPE_SFIXED_POINT_32:    return "QNN_DATATYPE_SFIXED_POINT_32";
        case QNN_DATATYPE_UFIXED_POINT_4:     return "QNN_DATATYPE_UFIXED_POINT_4";
        case QNN_DATATYPE_UFIXED_POINT_8:     return "QNN_DATATYPE_UFIXED_POINT_8";
        case QNN_DATATYPE_UFIXED_POINT_16:    return "QNN_DATATYPE_UFIXED_POINT_16";
        case QNN_DATATYPE_UFIXED_POINT_32:    return "QNN_DATATYPE_UFIXED_POINT_32";
        case QNN_DATATYPE_BOOL_8:             return "QNN_DATATYPE_BOOL_8";
        case QNN_DATATYPE_STRING:             return "QNN_DATATYPE_STRING";
        case QNN_DATATYPE_UNDEFINED:          return "QNN_DATATYPE_UNDEFINED";
        default:
            MNN_ERROR("MNN_QNN: Unknown data type.\n");
            return "";
    }
}

std::string QNNTranslator::TranslateDimensionsArray(const std::string & dimensionsNameSymbol, uint32_t rank, const uint32_t * dimensions) {
    std::string result = "  uint32_t ";
    result += dimensionsNameSymbol;
    result += "[] = {";
    for (uint32_t i = 0; i < rank; ++i) {
        result += std::to_string(dimensions[i]);
        if (i + 1 < rank) {
            result += ", ";
        }
    }
    result += "};";
    return result;
}

std::string QNNTranslator::TranslateParamDataArray(const std::string & dataNameSymbol, Qnn_DataType_t dataType, const Qnn_ClientBuffer_t & clientBuf) {
    std::string result = "  ";

    std::string dataTypeSymbol;
    switch (dataType) {
        case QNN_DATATYPE_UINT_32:
            dataTypeSymbol = "uint32_t";
            break;
        case QNN_DATATYPE_INT_32:
            dataTypeSymbol = "int";
            break;
        default:
            MNN_ERROR("MNN_QNN: Unknown data type for param tensor.\n");
            return "";
    }
    result += dataTypeSymbol;

    result += " ";
    result += dataNameSymbol;
    result += "[] = {";

    switch (dataType) {
        case QNN_DATATYPE_UINT_32: {
            const uint32_t * source = (const uint32_t *)clientBuf.data;
            uint32_t numEle = clientBuf.dataSize / sizeof(uint32_t);
            for (uint32_t i = 0; i < numEle; i++) {
                result += std::to_string(source[i]);
                if (i < numEle - 1) {
                    result += ", ";
                }
            }
            break;
        }
        case QNN_DATATYPE_INT_32: {
            const int * source = (const int *)clientBuf.data;
            uint32_t numEle = clientBuf.dataSize / sizeof(int);
            for (uint32_t i = 0; i < numEle; i++) {
                result += std::to_string(source[i]);
                if (i < numEle - 1) {
                    result += ", ";
                }
            }
            break;
        }
        default:
            MNN_ERROR("MNN_QNN: Unknown data type for param tensor.\n");
            return "";
    }

    result += "};";

    return result;
}

std::vector<std::string> QNNTranslator::TranslateQuantizeScaleOffsetDataArray(const std::string & tensorNameSymbol, const Qnn_QuantizeParams_t & quantizeParams, uint32_t rank, const uint32_t * dimensions){
    std::vector<std::string> result;
    if(quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET){
        result.push_back("  Qnn_ScaleOffset_t " + tensorNameSymbol + "_axis_scale_offset[] = {");
        int totalnum = (quantizeParams.axisScaleOffsetEncoding.numScaleOffsets + 3) / 4;
        for(int i = 0; i < totalnum; ++i){
            std::string line = "    ";
            for(int j = 0; j < 4; ++j){
                int index = i * 4 + j;
                if(index >= quantizeParams.axisScaleOffsetEncoding.numScaleOffsets)
                    break;
                line += "{.scale= " + std::to_string(quantizeParams.axisScaleOffsetEncoding.scaleOffset[index].scale) + ", .offset= " + std::to_string(quantizeParams.axisScaleOffsetEncoding.scaleOffset[index].offset) + "}, ";
            }
            result.push_back(line);
        }
        result.push_back("  };");
    }
    
    if(quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET){
        result.push_back("  float " + tensorNameSymbol + "_bwaxis_scale[] = {");
        int totalnum = (quantizeParams.bwAxisScaleOffsetEncoding.numElements + 3) / 4;
        for(int i = 0; i < totalnum; ++i){
            std::string line = "    ";
            for(int j = 0; j < 4; ++j){
                int index = i * 4 + j;
                if(index >= quantizeParams.bwAxisScaleOffsetEncoding.numElements)
                    break;
                line += std::to_string(quantizeParams.bwAxisScaleOffsetEncoding.scales[index]) + ", ";
            }
            result.push_back(line);
        }
        result.push_back("  };");
        if(quantizeParams.bwAxisScaleOffsetEncoding.offsets != nullptr){
            result.push_back("  int32_t " + tensorNameSymbol + "_bwaxis_offset[] = {");
            for(int i = 0; i < totalnum; ++i){
                std::string line = "    ";
                for(int j = 0; j < 4; ++j){
                    int index = i * 4 + j;
                    if(index >= quantizeParams.bwAxisScaleOffsetEncoding.numElements)
                        break;
                    line += std::to_string(quantizeParams.bwAxisScaleOffsetEncoding.offsets[index]) + ", ";
                }
                result.push_back(line);
            }
            result.push_back("  };");
        }
    }
    
    if(quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION){
        int axis = quantizeParams.blockwiseExpansion->axis;
        int oc = dimensions[axis];
        int blockSize = quantizeParams.blockwiseExpansion->numBlocksPerAxis;
        result.push_back("  Qnn_BlockwiseExpansion_t " + tensorNameSymbol + "_blockwiseExpansion = QNN_BLOCKWISE_EXPANSION_INIT;");
        
        result.push_back("  Qnn_ScaleOffset_t " + tensorNameSymbol + "_blockwiseExpansionScaleOffset[] = {");
        int totalnum = (oc + 3) / 4;
        for(int i = 0; i < totalnum; ++i){
            std::string line = "    ";
            for(int j = 0; j < 4; ++j){
                int index = i * 4 + j;
                if(index >= oc)
                    break;
                line += "{.scale= " + std::to_string(quantizeParams.blockwiseExpansion->scaleOffsets[index].scale) + ", .offset= " + std::to_string(quantizeParams.blockwiseExpansion->scaleOffsets[index].offset) + "}, ";
            }
            result.push_back(line);
        }
        result.push_back("  };");
        if(quantizeParams.blockwiseExpansion->blockScaleStorageType == QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8){
            result.push_back("  uint8_t " + tensorNameSymbol + "_blockwiseExpansionBlockScale[] = {");
            totalnum = (oc * blockSize + 3) / 4;
            for(int i = 0; i < totalnum; ++i){
                std::string line = "    ";
                for(int j = 0; j < 4; ++j){
                    int index = i * 4 + j;
                    if(index >= oc * blockSize)
                        break;
                    line += std::to_string(quantizeParams.blockwiseExpansion->blocksScale8[index]) + ", ";
                }
                result.push_back(line);
            }
            result.push_back("  };");
        }else{
            result.push_back("  uint16_t " + tensorNameSymbol + "_blockwiseExpansionBlockScale[] = {");
            totalnum = (oc * blockSize + 3) / 4;
            for(int i = 0; i < totalnum; ++i){
                std::string line = "    ";
                for(int j = 0; j < 4; ++j){
                    int index = i * 4 + j;
                    if(index >= oc * blockSize)
                        break;
                    line += std::to_string(quantizeParams.blockwiseExpansion->blocksScale16[index]) + ", ";
                }
                result.push_back(line);
            }
            result.push_back("  };");
        }
        result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.axis = " + std::to_string(quantizeParams.blockwiseExpansion->axis) + ";");
        result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.scaleOffsets = " + tensorNameSymbol + "_blockwiseExpansionScaleOffset;");
        result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.numBlocksPerAxis = " + std::to_string(quantizeParams.blockwiseExpansion->numBlocksPerAxis) + ";");
        result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.blockScaleBitwidth = " + std::to_string(quantizeParams.blockwiseExpansion->blockScaleBitwidth) + ";");
        if(quantizeParams.blockwiseExpansion->blockScaleStorageType == QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8){
            result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;");
            result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.blocksScale8 = " + tensorNameSymbol + "_blockwiseExpansionBlockScale;");
        }else{
            result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16;");
            result.push_back("  " + tensorNameSymbol + "_blockwiseExpansion.blocksScale16 = " + tensorNameSymbol + "_blockwiseExpansionBlockScale;");
        }
    }
    return result;
}

// Currently, only support QNN_QUANTIZATION_ENCODING_UNDEFINED, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET.
std::vector<std::string> QNNTranslator::TranslateTensorQuantizeParams(const std::string tensorNameSymbol, const Qnn_QuantizeParams_t & quantizeParams) {
    std::vector<std::string> result;

    if (quantizeParams.encodingDefinition == QNN_DEFINITION_UNDEFINED) {
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.scaleOffsetEncoding.scale = 0.0f;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.scaleOffsetEncoding.offset = 0;");
        return result;
    }

    if (quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.scaleOffsetEncoding.scale = " + std::to_string(quantizeParams.scaleOffsetEncoding.scale) + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.scaleOffsetEncoding.offset = " + std::to_string(quantizeParams.scaleOffsetEncoding.offset) + ";");
        return result;
    }
    
    if(quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET){
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.axisScaleOffsetEncoding.axis = " + std::to_string(quantizeParams.axisScaleOffsetEncoding.axis) + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets = " + std::to_string(quantizeParams.axisScaleOffsetEncoding.numScaleOffsets) + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset = " + tensorNameSymbol + "_axis_scale_offset;");
        return result;
    }
    
    if(quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET){
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.bwAxisScaleOffsetEncoding.axis = " + std::to_string(quantizeParams.bwAxisScaleOffsetEncoding.axis) + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.bwAxisScaleOffsetEncoding.bitwidth = " + std::to_string(quantizeParams.bwAxisScaleOffsetEncoding.bitwidth) + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.bwAxisScaleOffsetEncoding.numElements = " + std::to_string(quantizeParams.bwAxisScaleOffsetEncoding.numElements) + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.bwAxisScaleOffsetEncoding.scales = " + tensorNameSymbol + "_bwaxis_scale;");
        if(quantizeParams.bwAxisScaleOffsetEncoding.offsets != nullptr)
            result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.bwAxisScaleOffsetEncoding.offset = " + tensorNameSymbol + "_bwaxis_offset;");
        return result;
    }
    
    if(quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED && quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION){
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;");
        result.push_back("  " + tensorNameSymbol + ".v1.quantizeParams.blockwiseExpansion = &" + tensorNameSymbol + "_blockwiseExpansion;");
        return result;
    }


    MNN_ERROR("MNN_QNN: Unknown QuantizeParams.\n");

    return result;
}

std::vector<std::string> QNNTranslator::TranslateTensorClientBuf(const std::string & tensorNameSymbol, const std::string & dataNameSymbol, const std::string & sname, const Qnn_ClientBuffer_t & clientBuf, bool hasClientBuf, bool isParam) {
    std::vector<std::string> result;

    if (!hasClientBuf) {
        result.push_back("  " + tensorNameSymbol + ".v1.clientBuf.data = nullptr;");
        result.push_back("  " + tensorNameSymbol + ".v1.clientBuf.dataSize = 0;");
        return result;
    }

    if (isParam) {
        result.push_back("  " + tensorNameSymbol + ".v1.clientBuf.data = " + dataNameSymbol + ";");
        result.push_back("  " + tensorNameSymbol + ".v1.clientBuf.dataSize = " + std::to_string(clientBuf.dataSize) + ";");
        return result;
    }

    if (hasClientBuf && (!isParam)) {
        result.push_back("  " + tensorNameSymbol + ".v1.clientBuf.data = BINVARSTART(" + sname + ");");
        result.push_back("  " + tensorNameSymbol + ".v1.clientBuf.dataSize = BINLEN(" + sname + ");");
        return result;
    }

    MNN_ERROR("MNN_QNN: Illegal ClientBuf setting.\n");

    return result;
}

std::vector<std::string> QNNTranslator::TranslateNodeParamArray(const std::string & nodeName, const std::string & paramArraySymbol, uint32_t numOfParams, const Qnn_Param_t * params) {
    std::vector<std::string> result;

    for (uint32_t i = 0; i < numOfParams; i++) {
        Qnn_Param_t param = params[i];
        std::string paramNameSymbol = PARAM_NAME_SYMBOL(param.name);
        result.push_back("  Qnn_Param_t " + paramNameSymbol + " = QNN_PARAM_INIT;");
        result.push_back("  {");
        if (param.paramType == QNN_PARAMTYPE_SCALAR) {
            result.push_back("  " + paramNameSymbol + ".paramType = QNN_PARAMTYPE_SCALAR;");
            result.push_back("  " + paramNameSymbol + ".name = \"" + std::string(param.name) + "\";");
            result.push_back("  " + paramNameSymbol + ".scalarParam.dataType = " + MapDataType(param.scalarParam.dataType) + ";");
            switch (param.scalarParam.dataType) {
                case QNN_DATATYPE_BOOL_8:
                    result.push_back("  " + paramNameSymbol + ".scalarParam.bool8Value = " + std::to_string(param.scalarParam.bool8Value) + ";");
                    break;
                case QNN_DATATYPE_UINT_32:
                    result.push_back("  " + paramNameSymbol + ".scalarParam.uint32Value = " + std::to_string(param.scalarParam.uint32Value) + ";");
                    break;
                case QNN_DATATYPE_INT_32:
                    result.push_back("  " + paramNameSymbol + ".scalarParam.int32Value = " + std::to_string(param.scalarParam.int32Value) + ";");
                    break;
                case QNN_DATATYPE_FLOAT_32:
                    result.push_back("  " + paramNameSymbol + ".scalarParam.floatValue = " + std::to_string(param.scalarParam.floatValue) + ";");
                    break;
                default:
                    MNN_ERROR("MNN_QNN: Unkown dataType.\n");
                    return {};
            }
        } else {
            result.push_back("  " + paramNameSymbol + ".paramType = QNN_PARAMTYPE_TENSOR;");
            result.push_back("  " + paramNameSymbol + ".name = \"" + std::string(param.name) + "\";");
            result.push_back("  " + paramNameSymbol + ".tensorParam = " + TENSOR_NAME_SYMBOL(param.tensorParam.v1.name) + ";");
        }
        result.push_back("  }");
    }

    std::string str = "  Qnn_Param_t " + paramArraySymbol + "[] = {";
    for (uint32_t i = 0; i < numOfParams; i++) {
        str += PARAM_NAME_SYMBOL(params[i].name);
        if (i < numOfParams - 1) {
            str += ", ";
        }
    }
    str += "};";
    result.push_back(str);

    return result;
}

std::vector<std::string> QNNTranslator::TranslateNodeInputArray(const std::string & inputArraySymbol, uint32_t numOfInputs, const Qnn_Tensor_t * inputs) {
    std::vector<std::string> result;

    std::string str = "  const char * " + inputArraySymbol + "[] = {";
    for (uint32_t i = 0; i < numOfInputs; i++) {
        str += "\"";
        str += std::string(inputs[i].v1.name);
        str += "\"";
        if (i < numOfInputs - 1) {
            str += ", ";
        }
    }
    str += "};";

    result.push_back(str);

    return result;
}

std::vector<std::string> QNNTranslator::TranslateNodeOutputArray(const std::string & outputArraySymbol, uint32_t numOfOutputs, const Qnn_Tensor_t * outputs) {
    std::vector<std::string> result;

    std::string str = "  Qnn_Tensor_t " + outputArraySymbol + "[] = {";
    for (uint32_t i = 0; i < numOfOutputs; i++) {
        str += TENSOR_NAME_SYMBOL(outputs[i].v1.name);
        if (i < numOfOutputs - 1) {
            str +=", ";
        }
    }
    str += "};";

    result.push_back(str);

    return result;
}

} // end namespace MNN
} // end namespace QNN
