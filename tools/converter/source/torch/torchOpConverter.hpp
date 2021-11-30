//
//  torchOpConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2021/04/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TORCHOPCONVERTER_HPP
#define TORCHOPCONVERTER_HPP

#include <map>
#include <memory>
#include "MNN_generated.h"
#include <MNN/MNNDefine.h>
#include <torch/script.h>
#include "OpCount.hpp"
#include "ConverterScope.hpp"

template <typename T>
static inline T getValue(const torch::jit::Value* value) {
    auto optional_ivalue = toIValue(value);
    T res;
    if (!optional_ivalue) {
        MNN_ERROR("getValue: must Constant Node.\n");
        return res;
    }
    c10::IValue& val = optional_ivalue.value();
    auto optional_res = val.toOptional<T>();
    if (!optional_res) {
        // MNN_ERROR("getValue: value is None.");
        return res;
    }
    return optional_res.value();
}

template <typename T>
static std::vector<T> getValue(const torch::jit::Value* value, std::vector<int>& shape) {
    std::vector<T> data;
    const auto tensor = getValue<at::Tensor>(value);
    int size = tensor.numel();
    if (!size) {
        return data;
    }
    auto shapes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    if (shapes.empty()) {
        shapes.push_back(size);
        strides.push_back(1);
    }
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    data.resize(size);
    int idx = 0;
    std::function<void(int, int)> copyData = [&](int dim, int offset) {
        if (dim == shapes.size()-1) {
            for (int i = 0; i < shapes[dim]; i++) {
                data[idx++] = tensor.data_ptr<T>()[offset + i * strides[dim]];
            }
        } else {
            for (int i = 0; i < shapes[dim]; i++) {
                copyData(dim + 1, offset + i * strides[dim]);
            }
        }
    };
    copyData(0, 0);
    return data;
}

static std::vector<int> getShape(const torch::jit::Value* value) {
    const auto tensor = getValue<at::Tensor>(value);
    auto shape = tensor.sizes().vec();
    std::vector<int> res(shape.begin(), shape.end());
    return res;
}

static std::string getRealOpType(const torch::jit::Node *node) {
    const auto kind = node->kind();
    // custom op
    if (!(kind.is_attr() || kind.is_aten() || kind.is_cuda() ||
          kind.is_prim() || kind.is_onnx() || kind.is_user() ||
          kind.is_caffe2() || kind.is_dimname())) {
        return "__custom__";
    }
    std::string opType(kind.toUnqualString());
    // convert _xxx_ to xxx
    int last = opType.size() - 1;
    int last2 = last - 1;
    if (last > 0 && opType[last] == '_' && opType[last2] != '_') {
        opType = opType.substr(0, opType.size() - 1);
    }
    if (opType.size() > 2 && opType[0] == '_' && opType[1] != '_') {
        opType = opType.substr(1, opType.size() - 1);
    }
    // distinguish overload function
    auto symb = c10::Symbol::fromQualString("attr::mnn_tag");
    if (node->hasAttribute(symb)) {
        opType += ("_" + node->s(symb));
    }
    return opType;
}

static MNN::DataType ScalarType2Dtype(at::ScalarType scalarType) {
    switch (scalarType) {
        case at::ScalarType::Byte:
            return MNN::DataType_DT_UINT8;
        case at::ScalarType::Char:
            return MNN::DataType_DT_INT8;
        case at::ScalarType::Bool:
            return MNN::DataType_DT_BOOL;
        case at::ScalarType::Int:
        case at::ScalarType::Long:
            return MNN::DataType_DT_INT32;
        case at::ScalarType::Half:
        case at::ScalarType::Float:
        case at::ScalarType::Double:
            return MNN::DataType_DT_FLOAT;
        default:
            return MNN::DataType_DT_FLOAT;
    }
}

class TorchScope : public ConverterScope {
public:
    TorchScope(MNN::NetT* net) : ConverterScope(net) {}
    TorchScope(MNN::SubGraphProtoT* subnet, MNN::NetT* parentNet, TorchScope* parentScope)
             : ConverterScope(subnet, parentNet, parentScope) {}
    bool dealPrime(const torch::jit::Node* node);
    void buildMNNOp(const torch::jit::Node *node);
    virtual int lookupTensor(std::string name);
    void declareVar(std::string name, const torch::jit::Node* var);
    const torch::jit::Node* lookupVar(std::string name) const;
    void buildSubGraph(const torch::jit::Block* block, const std::string& name, bool increment = false);
private:
    std::map<std::string, const torch::jit::Node*> varTable;
};

class torchOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scop) = 0;
    virtual MNN::OpParameter type() = 0;
    virtual MNN::OpType opType() = 0;
    virtual std::vector<int> inputTensorIdx() { return { 0 }; }
    torchOpConverter() {}
    virtual ~torchOpConverter() {}
    friend class torchOpConverterSuit;
};

class torchOpConverterSuit {
public:
    torchOpConverterSuit() {}
    ~torchOpConverterSuit();
    static torchOpConverterSuit* get();
    void insert(torchOpConverter* t, const char* name);
    torchOpConverter* search(const std::string& name);

private:
    static torchOpConverterSuit* global;
    std::map<std::string, torchOpConverter*> mConverterContainer;
};

template <class T>
class torchOpConverterRegister {
public:
    torchOpConverterRegister(const char* name) {
        T* converter                  = new T;
        torchOpConverterSuit* container = torchOpConverterSuit::get();
        MNN::OpCount::get()->insertOp("TORCH", name);
        container->insert(converter, name);
    }

    ~torchOpConverterRegister() {
    }
};

#define DECLARE_OP_CONVERTER(name)                                                                    \
    class name : public torchOpConverter {                                                            \
    public:                                                                                           \
        name() {                                                                                      \
        }                                                                                             \
        virtual ~name() {                                                                             \
        }                                                                                             \
        virtual void run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope);           \
        virtual MNN::OpType opType();                                                                 \
        virtual MNN::OpParameter type();                                                              \
        virtual std::vector<int> inputTensorIdx();                                                    \
    }

#define REGISTER_CONVERTER(name, opType) static torchOpConverterRegister<name> _Convert_##opType(#opType)

#endif // TORCHOPCONVERTER_HPP
