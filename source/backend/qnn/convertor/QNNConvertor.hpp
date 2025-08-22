#ifndef MNN_QNNCONVERTOR_HPP
#define MNN_QNNCONVERTOR_HPP

#include "QNNUtils.hpp"

namespace MNN {
namespace QNN {

enum QNNCommandType {
    QNNCommandTypeBegin,
    QNNCommandTypeTensor,
    QNNCommandTypeNode,
    QNNCommandTypeEnd
};

enum Qnn_Convertor_Tensor_t {
    TENSOR_INPUT,
    TENSOR_STATIC,
    TENSOR_PARAM,
    TENSOR_NATIVE,
    TENSOR_OUTPUT
};

struct QNNCommandTensor {
    const char* name;
    Qnn_Convertor_Tensor_t type; // We don't use Qnn_TensorType_t, because <Static Param Tensor> and <Static Input Tensor> should be handled differently.
    Qnn_DataType_t dataType;
    Qnn_QuantizeParams_t quantizeParams;
    uint32_t rank;
    uint32_t* dimensions;
    Qnn_ClientBuffer_t clientBuf;
};

struct QNNCommandNode {
    const char* name;
    const char* packageName;
    const char* typeName;
    uint32_t numOfParams;
    Qnn_Param_t* params;
    uint32_t numOfInputs;
    Qnn_Tensor_t* inputTensors;
    uint32_t numOfOutputs;
    Qnn_Tensor_t* outputTensors;
};

struct QNNCommand {
    QNNCommandType type;
    union {
        QNNCommandTensor commandTensor;
        QNNCommandNode commandNode;
    };
};

// Codes related to text generation.
class QNNTranslator {
// Interface.
public:
    static std::vector<std::string> TranslateCommand(const QNNCommand & cmd);
// Translator for different command types.
private:
    static std::vector<std::string> TranslateBegin();
    static std::vector<std::string> TranslateTensor(const QNNCommandTensor & cmdT);
    static std::vector<std::string> TranslateNode(const QNNCommandNode & cmdN);
    static std::vector<std::string> TranslateEnd();
// Utility functions.
private:
    // Utility functions used by TranslateTensor.
    static std::string MapTensorType(Qnn_Convertor_Tensor_t type);
    static std::string MapDataType(Qnn_DataType_t dataType);
    static std::string TranslateDimensionsArray(const std::string & dimensionsNameSymbol, uint32_t rank, const uint32_t * dimensions);
    static std::string TranslateParamDataArray(const std::string & dataNameSymbol, Qnn_DataType_t dataType, const Qnn_ClientBuffer_t & clientBuf);
    static std::vector<std::string> TranslateQuantizeScaleOffsetDataArray(const std::string & tensorNameSymbol, const Qnn_QuantizeParams_t & quantizeParams, uint32_t rank, const uint32_t * dimensions);
    static std::vector<std::string> TranslateTensorQuantizeParams(const std::string tensorNameSymbol, const Qnn_QuantizeParams_t & quantizeParams);
    static std::vector<std::string> TranslateTensorClientBuf(const std::string & tensorNameSymbol, const std::string & dataNameSymbol, const std::string & sname, const Qnn_ClientBuffer_t & clientBuf, bool hasClientBuf, bool isParam);
    // Utility functions used by TranslateNode.
    static std::vector<std::string> TranslateNodeParamArray(const std::string & nodeName,const std::string & paramArraySymbol, uint32_t numOfParams, const Qnn_Param_t * params);
    static std::vector<std::string> TranslateNodeInputArray(const std::string & inputArraySymbol, uint32_t numOfInputs, const Qnn_Tensor_t * inputs);
    static std::vector<std::string> TranslateNodeOutputArray(const std::string & outputArraySymbol, uint32_t numOfOutputs, const Qnn_Tensor_t * outputs);
// Graph name, set by QNNConvertor.
public:
    static std::string GraphNameSymbol;
};

class QNNConvertor {
public:
    static void RecordBegin(const char* graphName);
    static void RecordTensor(const Qnn_Tensor_t * tensor);
    static void RecordNode(const Qnn_OpConfig_t & opConfig);
    static void RecordEnd();

private:
    static void Translate(const QNNCommand & cmd);
    static void DumpBuffer(const char * name, const void * buffer, size_t size);

public:
    static std::string OutputDir; // Set by QnnRuntime::onSetCachePath.
private:
    static FILE * CppFilePointer;
    static std::string CppBuffer;
    static const uint32_t CppBufferSize;
};

} // end namespace MNN
} // end namespace QNN

#endif // end MNN_QNNCONVERTOT_HPP
