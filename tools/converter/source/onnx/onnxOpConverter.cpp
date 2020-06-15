//
//  onnxOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"
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
class DefaultonnxOpConverter : public onnxOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) override {
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
                    attr->tensor.reset(convertTensorToBlob(&srcAttr.t()));
                    break;
                default:
                    break;
            }
            attr->i = _limit(srcAttr.i());
            attr->s = srcAttr.s();
            attr->f = srcAttr.f();
            extra->attr.emplace_back(std::move(attr));
        }
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
MNN::DataType onnxOpConverter::convertDataType(::onnx::TensorProto_DataType type) {
    static std::map<::onnx::TensorProto_DataType, MNN::DataType> dataTypeMap{
        {onnx::TensorProto_DataType_FLOAT, MNN::DataType_DT_FLOAT},
        {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_INT32, MNN::DataType_DT_INT32},
        {onnx::TensorProto_DataType_INT64, MNN::DataType_DT_INT32},  // For compability, use int32 instead of int64
        {onnx::TensorProto_DataType_DOUBLE, MNN::DataType_DT_FLOAT}, // For compability, use float instead of double
        {onnx::TensorProto_DataType_UINT8, MNN::DataType_DT_UINT8},
        {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_BOOL, MNN::DataType_DT_INT32},   // For compability, use int32 instead of bool
        {onnx::TensorProto_DataType_INT16, MNN::DataType_DT_INT32},  // For compability, use int32 instead of int16
        {onnx::TensorProto_DataType_UINT16, MNN::DataType_DT_INT32}, // For compability, use int32 instead of uint16
    };
    if (dataTypeMap.find(type) != dataTypeMap.end()) {
        return dataTypeMap[type];
    }
    return MNN::DataType_DT_INVALID;
}
MNN::BlobT* onnxOpConverter::convertTensorToBlob(const onnx::TensorProto* constantTp) {
    auto constantParam = new MNN::BlobT;
    auto dataType      = convertDataType(constantTp->data_type());
    // printf("origindataType = %d, dataType = %s\n", constantTp->data_type(), MNN::EnumNameDataType(dataType));

    constantParam->dataType   = dataType;
    constantParam->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;

    size_t dimSize = constantTp->dims().size();
    constantParam->dims.resize(dimSize);
    size_t dataSize = 1;
    for (int i = 0; i < dimSize; ++i) {
        constantParam->dims[i] = constantTp->dims(i);
        dataSize               = dataSize * constantTp->dims(i);
    }

    const void* tensor_content = constantTp->raw_data().data();

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
        CASE_DATA_TYPE(onnx::TensorProto_DataType_FLOAT, float);
        default:
            break;
    }

    if (!tensor_content) {
        DLOG(FATAL) << "Convert no data, "
                       "Please make sure ";
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
            auto source = (uint8_t*)tensor_content;
            constantParam->uint8s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->uint8s[i] = source[i];
            }
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
        default: {
            DLOG(FATAL) << "Don't support " << constantTp->data_type();
            break;
        }
    }
    return constantParam;
}
