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

template <typename T>
static inline T getValue(const torch::jit::Value* value) {
    auto optional_ivalue = toIValue(value);
    T res;
    if (!optional_ivalue) {
        MNN_ERROR("getValue: must Constant Node.");
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
    const auto shapes = tensor.sizes().vec();
    const auto strides = tensor.strides().vec();
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    auto scalarType = tensor.scalar_type();
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

static std::string getRealOpType(const char* type) {
    std::string opType(type);
    int last = opType.size() - 1;
    int last2 = last - 1;
    if (last > 0 && opType[last] == '_' && opType[last2] != '_') {
        opType = opType.substr(0, opType.size() - 1);
    }
    return opType;
}

class torchContext {
public:
    torchContext() : mNet(nullptr), mSubNet(nullptr) {}
    torchContext(MNN::NetT* net) : mNet(net), mSubNet(nullptr) {}
    torchContext(MNN::NetT* net, MNN::SubGraphProtoT* subnet) : mNet(net), mSubNet(subnet) {}
    void buildOp(const torch::jit::Node* node);
    bool dealPrime(const torch::jit::Node* node);
    int declareTensor(std::string name);
    int lookupTensor(std::string name);
    std::string lookupTensor(int idx) const;
    void declareVar(std::string name, const torch::jit::Node* var);
    const torch::jit::Node* lookupVar(std::string name) const;
    std::vector<int> addSubGraph(const torch::jit::Block* block, const std::string& name);
private:
    std::map<std::string, const torch::jit::Node*> varTable;
    std::map<std::string, int> tensorIdx;
    MNN::NetT* mNet;
    MNN::SubGraphProtoT* mSubNet;
};

class torchOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) = 0;
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
        virtual void run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context);       \
        virtual MNN::OpType opType();                                                                 \
        virtual MNN::OpParameter type();                                                              \
        virtual std::vector<int> inputTensorIdx();                                                    \
    }

#define REGISTER_CONVERTER(name, opType) static torchOpConverterRegister<name> _Convert_##opType(#opType)

#endif // TORCHOPCONVERTER_HPP
