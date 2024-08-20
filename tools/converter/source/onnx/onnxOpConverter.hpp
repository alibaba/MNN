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
#include "ConverterScope.hpp"

class OnnxScope : public ConverterScope {
public:
    static std::vector<int> topoSort(const onnx::GraphProto& onnxGraph);
    OnnxScope(const onnx::GraphProto* graph, MNN::NetT* net, const std::string& modelDir) : mGraph(graph), ConverterScope(net) { onnxInit(); mModelDir = modelDir;}
    OnnxScope(const onnx::GraphProto* graph, MNN::SubGraphProtoT* subnet, MNN::NetT* net,
              OnnxScope* parent) : mGraph(graph), ConverterScope(subnet, net, parent) { onnxInit(); mModelDir = parent->mModelDir;}
    std::pair<int, int> buildTensorArrayOp(std::vector<int> element_shape, bool identical, const std::string& name, int init_size = 1);
    void buildAccumulate(const std::string& name, const std::string& uName, const std::string& iName, const std::string& oName);
    // Return extra input needed from subgraph
    // WhileModule implemention acquire
    std::vector<std::string> buildSubGraph(const onnx::GraphProto* graph, std::string& name, bool forLoop);
public:
    virtual int lookupTensor(std::string name);
public:
    std::map<std::string, const onnx::TensorProto*> mInitializers;
    std::map<std::string, const onnx::ValueInfoProto*> mInputs;
    std::map<std::string, const onnx::ValueInfoProto*> mOutputs;
    int mOpsetVersion;
    std::string mModelDir;
private:
    // onnx graph and infos
    const onnx::GraphProto* mGraph;
    void onnxInit();
};

class onnxOpConverter {
public:
    onnxOpConverter() {
    }
    virtual ~onnxOpConverter() {
    }
    virtual void run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) = 0;
    virtual MNN::OpParameter type()                                      = 0;
    virtual MNN::OpType opType()                                         = 0;
    static MNN::DataType convertDataType(int32_t type);
    static MNN::BlobT* convertTensorToBlob(const onnx::TensorProto* tensor, const std::string& modelDir, MNN::OpT* op);
    // static std::unique_ptr<MNN::SubGraphProtoT> buildSubGraph(const onnx::GraphProto* graph, std::string& name);
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
                         OnnxScope* scope);                                   \
        virtual MNN::OpType opType();                                         \
        virtual MNN::OpParameter type();                                      \
    }
#define REGISTER_CONVERTER(name, opType) static onnxOpConverterRegister<name> _Convert_##opType(#opType)

#endif // ONNXOPCONVERTER_HPP
