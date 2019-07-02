//
//  onnxOpConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ONNXOPCONVERTER_HPP
#define ONNXOPCONVERTER_HPP

#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "logkit.h"
#include "onnx.pb.h"

class onnxOpConverter {
public:
    onnxOpConverter() {
    }
    virtual ~onnxOpConverter() {
    }
    virtual void run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) = 0;
    virtual MNN::OpParameter type()                                      = 0;
    virtual MNN::OpType opType()                                         = 0;
    static MNN::DataType convertDataType(::onnx::TensorProto_DataType type) {
        static std::map<::onnx::TensorProto_DataType, MNN::DataType> dataTypeMap{
            {onnx::TensorProto_DataType_FLOAT, MNN::DataType_DT_FLOAT},
            {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
            {onnx::TensorProto_DataType_INT32, MNN::DataType_DT_INT32},
            {onnx::TensorProto_DataType_INT64, MNN::DataType_DT_INT32}, // For compability, use int32 instead of int64
            {onnx::TensorProto_DataType_UINT8, MNN::DataType_DT_UINT8},
        };
        if (dataTypeMap.find(type) != dataTypeMap.end()) {
            return dataTypeMap[type];
        }
        return MNN::DataType_DT_INVALID;
    }
    static MNN::BlobT* convertTensorToBlob(const onnx::TensorProto * tensor);
};

class onnxOpConverterSuit {
public:
    onnxOpConverterSuit();
    ~onnxOpConverterSuit();
    static onnxOpConverterSuit* get();
    void insert(onnxOpConverter* t, const char* name);

    onnxOpConverter* search(const std::string& name);

private:
    static onnxOpConverterSuit* global;
    std::map<std::string, onnxOpConverter*> mConverterContainer;
};

template <typename T>
class onnxOpConverterRegister {
public:
    onnxOpConverterRegister(const char* name) {
        T* opConverter                 = new T;
        onnxOpConverterSuit* container = onnxOpConverterSuit::get();
        container->insert(opConverter, name);
    }
    ~onnxOpConverterRegister() {
    }

private:
    onnxOpConverterRegister();
};

#define DECLARE_OP_CONVERTER(name)                                            \
    class name : public onnxOpConverter {                                     \
    public:                                                                   \
        name() {                                                              \
        }                                                                     \
        virtual ~name() {                                                     \
        }                                                                     \
        virtual void run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,    \
                         std::vector<const onnx::TensorProto*> initializers); \
        virtual MNN::OpType opType();                                         \
        virtual MNN::OpParameter type();                                      \
    }

#define REGISTER_CONVERTER(name, opType) static onnxOpConverterRegister<name> _Convert_##opType(#opType)

#endif // ONNXOPCONVERTER_HPP
