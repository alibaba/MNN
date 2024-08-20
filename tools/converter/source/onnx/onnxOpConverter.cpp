//
//  onnxOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <queue>
#include "onnxOpConverter.hpp"
#include "OpCount.hpp"
#include "OnnxTmpGraph.hpp"

using namespace MNN;
static int32_t _limit(int64_t i64) {
    if (i64 > (int64_t)(1 << 30)) {
        return 1 << 30;
    }
    if (i64 < (int64_t)(-(1 << 30))) {
        return (-(1 << 30));
    }
    return i64;
}
std::vector<int> OnnxScope::topoSort(const onnx::GraphProto& onnxGraph) {
    std::vector<int> idxMap;
    const int nodeCount   = onnxGraph.node_size();
    std::map<std::string, int> outputMap;
    std::map<int, std::vector<int>> graph; // key --[in]--> values
    std::vector<int> inDegree(nodeCount);
    // build Graph and inDegree
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        for (int k = 0; k < onnxNode.output_size(); k++) {
            outputMap.insert(std::make_pair(onnxNode.output(k), i));
        }
    }
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        for (int k = 0; k < onnxNode.input_size(); k++) {
            auto inputName = onnxNode.input(k);
            auto iter = outputMap.find(inputName);
            if (iter != outputMap.end()) {
                graph[iter->second].push_back(i);
            }
        }
        if (onnxNode.op_type() == "Loop") {
            auto& body = onnxNode.attribute(0).g();
            for (int j=0; j<body.node_size(); ++j) {
                for (int k=0; k<body.node(j).input_size(); ++k) {
                    auto inputName = body.node(j).input(k);
                    auto iter = outputMap.find(inputName);
                    if (iter != outputMap.end()) {
                        graph[iter->second].push_back(i);
                    }
                }
            }
        }
    }
    for (auto node : graph) {
        for (auto output : node.second) {
            inDegree[output]++;
        }
    }
    // topo sort
    std::queue<int> validNode;
    for (int i = 0; i < nodeCount; i++) {
        if (!inDegree[i]) {
            validNode.push(i);
        }
    }
    while (!validNode.empty()) {
        int node = validNode.front();
        validNode.pop();
        idxMap.push_back(node);
        for (auto succ : graph[node]) {
            if (--inDegree[succ] == 0) {
                validNode.push(succ);
            }
        }
    }
    MNN_ASSERT(idxMap.size() == nodeCount);
    return idxMap;
}

class DefaultonnxOpConverter : public onnxOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     OnnxScope* scope) override {
        auto extra        = new ExtraT;
        dstOp->main.type  = OpParameter_Extra;
        dstOp->main.value = extra;
        extra->engine     = "ONNX";
        extra->type       = onnxNode->op_type();
        for (auto srcAttr : onnxNode->attribute()) {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = srcAttr.name();
            switch (srcAttr.type()) {
                case onnx::AttributeProto_AttributeType_INTS:
                    attr->list.reset(new ListValueT);
                    attr->list->i.resize(srcAttr.ints_size());
                    for (int i = 0; i < srcAttr.ints_size(); ++i) {
                        attr->list->i[i] = _limit(srcAttr.ints(i));
                    }
                    break;
                case onnx::AttributeProto_AttributeType_FLOATS:
                    attr->list.reset(new ListValueT);
                    attr->list->f.resize(srcAttr.floats_size());
                    for (int i = 0; i < srcAttr.floats_size(); ++i) {
                        attr->list->f[i] = srcAttr.floats(i);
                    }
                    break;
                case onnx::AttributeProto_AttributeType_TENSOR:
                    attr->tensor.reset(convertTensorToBlob(&srcAttr.t(), scope->mModelDir, dstOp));
                    break;
                default:
                    break;
            }
            attr->i = _limit(srcAttr.i());
            attr->s = srcAttr.s();
            attr->f = srcAttr.f();
            extra->attr.emplace_back(std::move(attr));
        }
        // add onnx ir version for some differet impl
        std::unique_ptr<AttributeT> attr(new AttributeT);
        attr->key = "onnx_opset_version";
        attr->i = scope->mOpsetVersion;
        extra->attr.emplace_back(std::move(attr));
    }
    virtual MNN::OpParameter type() override {
        return OpParameter_Extra;
    }
    virtual MNN::OpType opType() override {
        return OpType_Extra;
    }
};

onnxOpConverterSuit::onnxOpConverterSuit() {
}

onnxOpConverterSuit::~onnxOpConverterSuit() {
    for (auto& iter : mConverterContainer) {
        delete iter.second;
    }
    mConverterContainer.clear();
}

onnxOpConverterSuit* onnxOpConverterSuit::global = nullptr;

onnxOpConverterSuit* onnxOpConverterSuit::get() {
    if (global == nullptr) {
        global = new onnxOpConverterSuit;
    }
    return global;
}

void onnxOpConverterSuit::insert(onnxOpConverter* t, const char* name) {
    MNN::OpCount::get()->insertOp("ONNX", std::string(name));
    mConverterContainer.insert(std::make_pair(name, t));
}

onnxOpConverter* onnxOpConverterSuit::search(const std::string& name) {
    auto iter = mConverterContainer.find(name);
    if (iter == mConverterContainer.end()) {
        static DefaultonnxOpConverter defaultConverter;
        return &defaultConverter;
    }
    return iter->second;
}
MNN::DataType onnxOpConverter::convertDataType(int32_t itype) {
    static std::map<::onnx::TensorProto_DataType, MNN::DataType> dataTypeMap{
        {onnx::TensorProto_DataType_FLOAT, MNN::DataType_DT_FLOAT},
        {onnx::TensorProto_DataType_FLOAT16, MNN::DataType_DT_HALF},
	{onnx::TensorProto_DataType_BFLOAT16, MNN::DataType_DT_BFLOAT16},     
   	{onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_INT32, MNN::DataType_DT_INT32},
        {onnx::TensorProto_DataType_INT64, MNN::DataType_DT_INT32},  // For compability, use int32 instead of int64
        {onnx::TensorProto_DataType_DOUBLE, MNN::DataType_DT_FLOAT}, // For compability, use float instead of double
        {onnx::TensorProto_DataType_UINT8, MNN::DataType_DT_UINT8},
        {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_BOOL, MNN::DataType_DT_INT32},   // For compability, use int32 instead of bool
        {onnx::TensorProto_DataType_INT16, MNN::DataType_DT_INT32},  // For compability, use int32 instead of int16
        {onnx::TensorProto_DataType_UINT16, MNN::DataType_DT_INT32}, // For compability, use int32 instead of uint16
        {onnx::TensorProto_DataType_UINT32, MNN::DataType_DT_INT32}, // For compability, use int32 instead of uint32
        {onnx::TensorProto_DataType_UINT64, MNN::DataType_DT_INT32}, // For compability, use int32 instead of uint64
    };
    auto type = static_cast<::onnx::TensorProto_DataType>(itype);
    if (dataTypeMap.find(type) != dataTypeMap.end()) {
        return dataTypeMap[type];
    }
    return MNN::DataType_DT_INVALID;
}
static bool _needConvert(int onnxDataType) {
    switch (onnxDataType) {
        case onnx::TensorProto_DataType_FLOAT:
        case onnx::TensorProto_DataType_FLOAT16:
        case onnx::TensorProto_DataType_BFLOAT16:
        case onnx::TensorProto_DataType_INT32:
        case onnx::TensorProto_DataType_UINT8:
        case onnx::TensorProto_DataType_INT8:
            return false;
            
        default:
            break;
    }
    return true;
}
MNN::BlobT* onnxOpConverter::convertTensorToBlob(const onnx::TensorProto* constantTp, const std::string& modelDir, MNN::OpT* op) {
    auto constantParam = new MNN::BlobT;
    auto dataType      = convertDataType(constantTp->data_type());
    // printf("origindataType = %d, dataType = %s\n", constantTp->data_type(), MNN::EnumNameDataType(dataType));

    constantParam->dataType   = dataType;
    constantParam->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;

    size_t dimSize = constantTp->dims().size();
    constantParam->dims.resize(dimSize);
    int64_t dataSize = 1;
    for (int i = 0; i < dimSize; ++i) {
        constantParam->dims[i] = constantTp->dims(i);
        dataSize               = dataSize * constantTp->dims(i);
    }
    std::vector<int64_t> alignContent;
    if (constantTp->data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
        std::string location;
        int64_t offset = 0;
        int64_t length = -1;
        for (const auto& k : constantTp->external_data()) {
            if (k.key() == "location") {
                location = k.value();
            } else if (k.key() == "offset") {
                offset = std::atoll(k.value().c_str());
            } else if (k.key() == "length") {
                length = std::atoll(k.value().c_str());
            }
        }
        if (!modelDir.empty()) {
            location = modelDir + location;
        }
        auto fp = fopen(location.c_str(), "rb");
        if (fp == nullptr) {
            DLOG(FATAL) << "Fail to open external data: " << location;
            return nullptr;
        }
        if (length < 0) {
            fseek(fp, 0, SEEK_END);
            length = ftell(fp) - offset;
        }
        fseek(fp, offset, SEEK_SET);
        if (_needConvert(constantTp->data_type())) {
            alignContent.resize((length + sizeof(int64_t) - 1) / sizeof(int64_t));
            fread(alignContent.data(), 1, length, fp);
        } else {
            op->externalPath = location;
            constantParam->external = {
                offset, length
            };
            dataSize = 0;
        }
        fclose(fp);
    } else {
        alignContent.resize((constantTp->raw_data().size() + sizeof(int64_t) - 1) / sizeof(int64_t));
        ::memcpy(alignContent.data(), constantTp->raw_data().data(), constantTp->raw_data().size());
    }

    const void* tensor_content = (const void*)alignContent.data();

    switch (constantTp->data_type()) {
#define CASE_DATA_TYPE(src, dst)                              \
    case src:                                                 \
        if (constantTp->dst##_data_size() != 0) {             \
            tensor_content = constantTp->dst##_data().data(); \
        }                                                     \
        break;
        CASE_DATA_TYPE(onnx::TensorProto_DataType_DOUBLE, double);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_INT64, int64);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_INT32, int32);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_UINT8, int32);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_INT8, int32);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_FLOAT, float);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_UINT64, uint64);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_BOOL, int32);
        default:
            break;
    }
    if (0 == dataSize) {
        // Empty blob
        return constantParam;
    }

    if (!tensor_content) {
        DLOG(FATAL) << "Convert no data, "
                       "Please make sure ";
        return nullptr;
    }

    switch (constantTp->data_type()) {
        case onnx::TensorProto_DataType_DOUBLE: {
            constantParam->float32s.resize(dataSize);
            auto source = (double*)tensor_content;

            for (int i = 0; i < dataSize; ++i) {
                constantParam->float32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT64: {
            constantParam->int32s.resize(dataSize);
            auto source = (int64_t*)tensor_content;

            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = _limit(source[i]);
            }
            break;
        }
        case onnx::TensorProto_DataType_INT32: {
            auto source = (int32_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT16: {
            auto source = (uint16_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT16: {
            auto source = (int16_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_BOOL: {
            auto source = (bool*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT8: {
            auto source = (int8_t*)tensor_content;
            constantParam->int8s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int8s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT8: {
            constantParam->uint8s.resize(dataSize);
            if (constantTp->int32_data_size() > 0) {
                auto source = (int32_t*)tensor_content;
                for (int i = 0; i < dataSize; ++i) {
                    constantParam->uint8s[i] = source[i];
                }
            } else {
                ::memcpy(constantParam->uint8s.data(), tensor_content, dataSize * sizeof(uint8_t));
            }
            break;
        }
        case onnx::TensorProto_DataType_FLOAT16: {
            constantParam->uint8s.resize(dataSize * sizeof(int16_t));
            ::memcpy(constantParam->uint8s.data(), tensor_content, dataSize * sizeof(int16_t));
            break;
        }
        case onnx::TensorProto_DataType_BFLOAT16: {
            constantParam->uint8s.resize(dataSize * sizeof(int16_t));
            ::memcpy(constantParam->uint8s.data(), tensor_content, dataSize * sizeof(int16_t));
            break;
        }
        case onnx::TensorProto_DataType_FLOAT: {
            float* tempFloatData = (float*)tensor_content;
            constantParam->float32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->float32s[i] = tempFloatData[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT32: {
            auto source = (uint32_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT64: {
            auto source = (uint64_t*)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = _limit(source[i]);
            }
            break;
        }
        default: {
            DLOG(FATAL) << "Don't support " << constantTp->data_type();
            break;
        }
    }
    return constantParam;
}

void OnnxScope::onnxInit() {
    const int initializerCount = mGraph->initializer_size();
    for (int i = 0; i < initializerCount; ++i) {
        const auto& initializer = mGraph->initializer(i);
        mInitializers.insert(std::make_pair(initializer.name(), &initializer));
    }
    const int inputCount = mGraph->input_size();
    for (int i = 0; i < inputCount; ++i) {
        const auto& input = mGraph->input(i);
        mInputs.insert(std::make_pair(input.name(), &input));
    }
    const int outputCount = mGraph->output_size();
    for (int i = 0; i < outputCount; ++i) {
        const auto& output = mGraph->output(i);
        mOutputs.insert(std::make_pair(output.name(), &output));
    }
}

int OnnxScope::lookupTensor(std::string name) {
    // onnx have optional input, which may be a placeholder when pytorch export onnx model,
    // so drop this input, but we should check it out sometimes.
    if(name == ""){
        return -1;
    }
    const auto iter = mTensorIdx.find(name);
    if (iter != mTensorIdx.end()) {
        return iter->second;
    }
    return -1;
}

std::pair<int, int> OnnxScope::buildTensorArrayOp(std::vector<int> element_shape, bool identical, const std::string& name, int init_size) {
    std::unique_ptr<MNN::OpT> tensorArrayOp(new MNN::OpT);
    tensorArrayOp->name      = name;
    tensorArrayOp->type      = MNN::OpType_TensorArray;
    tensorArrayOp->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    tensorArrayOp->main.type = MNN::OpParameter_TensorArray;
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = DataType_DT_FLOAT;
    tensorArray->dynamic_size = true;
    tensorArray->identical_element_shapes = identical;
    tensorArray->element_shape = element_shape;
    tensorArrayOp->main.value = tensorArray;
    tensorArrayOp->inputIndexes.push_back(buildIntConstOp({init_size}, name + "/init_size"));
    int idx_handle = declareTensor(name + "/handle");
    int idx = declareTensor(name);
    tensorArrayOp->outputIndexes.push_back(idx_handle);
    tensorArrayOp->outputIndexes.push_back(idx);
    oplists().emplace_back(std::move(tensorArrayOp));
    return std::make_pair(idx_handle, idx);
}

void OnnxScope::buildAccumulate(const std::string& name, const std::string& uName, const std::string& iName, const std::string& oName) {
    // for while_body: %user_defined_val = Add(%user_defined_val, %output)
    int idxAcc = declareTensor(name + "/accumulate_u");
    MNN::OpT* accumulateOp  = new MNN::OpT;
    accumulateOp->name      = name + "/accumulate";
    accumulateOp->type      = MNN::OpType_TensorArrayWrite;
    accumulateOp->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    accumulateOp->main.type = MNN::OpParameter_TensorArray;
    auto param  = new MNN::TensorArrayT;
    param->T = MNN::DataType_DT_FLOAT;
    accumulateOp->main.value = param;
    // handle, index, value, flow_in
    addInputForOp(accumulateOp, uName + "/handle");
    addInputForOp(accumulateOp, iName);
    addInputForOp(accumulateOp, oName);
    addInputForOp(accumulateOp, uName);
    accumulateOp->outputIndexes.push_back(idxAcc);
    oplists().emplace_back(accumulateOp);
    mSubNet->outputs.push_back(idxAcc);
}

std::vector<std::string> OnnxScope::buildSubGraph(const onnx::GraphProto* graph, std::string& name, bool forLoop) {
    std::unique_ptr<MNN::SubGraphProtoT> subgraph(new MNN::SubGraphProtoT);
    subgraph->name = name;
    std::unique_ptr<OnnxScope> scope(new OnnxScope(graph, subgraph.get(), mNet, this));
    const auto& initializers         = scope->mInitializers;
    const auto& inputs               = scope->mInputs;
    const auto& outputs              = scope->mOutputs;
    // set input node to MNN net
    for (int index=0; index < graph->input_size(); ++index) {
        auto inputName = graph->input(index).name();
        bool notHaveInitializer = initializers.find(inputName) == initializers.end();
        if (notHaveInitializer) {
            MNN::OpT* MNNOp  = new MNN::OpT;
            MNNOp->name      = inputName;
            MNNOp->type      = MNN::OpType_Input;
            MNNOp->main.type = MNN::OpParameter_Input;
            auto inputParam  = new MNN::InputT;
            const auto it    = inputs.find(inputName);
            const auto& tensorInfo = (it->second)->type().tensor_type();
            const int inputDimSize = tensorInfo.shape().dim_size();
            inputParam->dims.resize(inputDimSize);
            for (int i = 0; i < inputDimSize; ++i) {
                inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
            }
            inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
            MNNOp->outputIndexes.push_back(scope->declareTensor(inputName));
            MNNOp->main.value = inputParam;
            subgraph->inputs.emplace_back(MNNOp->outputIndexes[0]);
            subgraph->nodes.emplace_back(MNNOp);
        }
    }
    // Find Extra Input from outside graph
    std::map<std::string, int> outsideInputs;
    auto findConst = [&](const std::string& name) {
        if (scope->lookupTensor(name) >= 0) {
            return;
        }
        // onnx subgraph may use tensor from initializers in outter level graph, recurrsive find it
        for (auto curScope = scope.get(); curScope != nullptr; ) {
            const auto& curInits = curScope->mInitializers;
            const auto it = curInits.find(name);
            if (it != curInits.end()) {
                // Create const Op
                MNN::OpT* constOp   = new MNN::OpT;
                constOp->type       = MNN::OpType_Const;
                constOp->main.type  = MNN::OpParameter_Blob;
                constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second, mModelDir, constOp);
                constOp->name    = it->first;
                constOp->outputIndexes.push_back(scope->declareTensor(it->first));
                subgraph->nodes.emplace_back(constOp);
                break;
            }
            curScope = reinterpret_cast<decltype(curScope)>(curScope->mParent);
        }
    };
    for (int i=0; i<graph->output_size(); ++i) {
        findConst(graph->output(i).name());
    }
    auto indexes = OnnxScope::topoSort(*graph);
    // Firstly declare output names
    for (auto i : indexes) {
        const auto& onnxNode = graph->node(i);
        for (int k = 0; k < onnxNode.output_size(); k++) {
            scope->declareTensor(onnxNode.output(k));
        }
    }
    for (auto i : indexes) {
        const auto& onnxNode = graph->node(i);
        const auto& opType   = onnxNode.op_type();
        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);
        auto opConverter = onnxOpConverterSuit::get()->search(opType);
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            findConst(inputName);
        }
        // build input and output
        for (int k = 0; k < onnxNode.input_size(); k++) {
            auto inputName = onnxNode.input(k);
            int idx = scope->lookupTensor(inputName);
            if (idx < 0) {
                auto iter = outsideInputs.find(inputName);
                if (iter == outsideInputs.end()) {
                    idx = scope->declareTensor(inputName);
                    std::unique_ptr<MNN::OpT> inputOp(new MNN::OpT);
                    inputOp->name      = inputName;
                    inputOp->type      = MNN::OpType_Input;
                    inputOp->main.type = MNN::OpParameter_Input;
                    auto param  = new MNN::InputT;
                    param->dtype = MNN::DataType_DT_INT32;
                    param->dformat = MNN::MNN_DATA_FORMAT_NCHW;
                    inputOp->main.value = param;
                    inputOp->outputIndexes.push_back(idx);
                    subgraph->nodes.emplace_back(std::move(inputOp));
                    outsideInputs.insert(std::make_pair(inputName, idx));
                } else {
                    idx = iter->second;
                }
            }
            MNNOp->inputIndexes.push_back(idx);
        }
        for (int k = 0; k < onnxNode.output_size(); k++) {
            MNNOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
        }
        auto originIdx = subgraph->inputs.size();
        opConverter->run(MNNOp, &onnxNode, scope.get());
        // subgraph own by op may introduce extra input which is not exist on current graph, create it in op converter and detect it by subgraph->inputs
        for (int inputIdx = originIdx; inputIdx < subgraph->inputs.size(); ++inputIdx) {
            auto idx = subgraph->inputs[inputIdx];
            outsideInputs.insert(std::make_pair(scope->lookupTensorByIdx(idx), idx));
        }
        subgraph->inputs.erase(subgraph->inputs.begin() + originIdx, subgraph->inputs.end());
        subgraph->nodes.emplace_back(MNNOp);
    }
    if (!forLoop) {
        std::vector<std::string> resOutside;
        for (auto& iter : outsideInputs) {
            subgraph->inputs.emplace_back(iter.second);
            resOutside.emplace_back(iter.first);
        }
        for (int i = 0; i < graph->output_size(); ++i) {
            int idx = scope->lookupTensor(graph->output(i).name());
            MNN_ASSERT(idx >= 0);
            if (idx >= 0) {
                subgraph->outputs.push_back(idx);
            }
        }
        mNet->subgraphs.emplace_back(std::move(subgraph));
        return resOutside;
    }
    int N = graph->input_size() - 2, K = graph->output_size() - N - 1;
    for (int i = 0; i < N + 1; i++) {
        int idx = scope->lookupTensor(graph->output(i).name());
        if (idx >= 0) {
            subgraph->outputs.push_back(idx);
        } else {
            FUNC_PRINT_ALL(graph->output(i).name().c_str(), s);
        }
    }
    std::vector<std::string> resOutside;
    for (auto& iter : outsideInputs) {
        subgraph->inputs.emplace_back(iter.second);
        subgraph->outputs.emplace_back(iter.second);
        resOutside.emplace_back(iter.first);
    }
    for (int i = 0; i < K; ++i) {
        int idx = scope->lookupTensor(graph->output(i + N + 1).name());
        MNN_ASSERT(idx >= 0);
        if (idx >= 0) {
            subgraph->outputs.push_back(idx);
        }
    }
    mNet->subgraphs.emplace_back(std::move(subgraph));
    return resOutside;
}
