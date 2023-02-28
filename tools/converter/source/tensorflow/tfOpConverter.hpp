//
//  tfOpConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TFtfOpCONVERTER_HPP
#define TFtfOpCONVERTER_HPP
#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "TmpGraph.hpp"
#include "graph.pb.h"

#include "logkit.h"

// The base class for tensorflow op converter
class tfOpConverter {
    friend class tfOpConverterSuit;

public:
    virtual void run(MNN::OpT *dstOp, TmpNode *srcNode) = 0;
    virtual MNN::OpParameter type()                                          = 0;
    virtual MNN::OpType opType()                                             = 0;
    tfOpConverter() {
    }
    virtual ~tfOpConverter() {
    }
    
    static void convertTensorToBlob(MNN::BlobT* dst, const ::tensorflow::TensorProto& tensor);

private:
};

// converter factory
class tfOpConverterSuit {
public:
    static tfOpConverterSuit *get();
    void insert(tfOpConverter *t, const char *name);

    tfOpConverter *search(const std::string &name);
    tfOpConverterSuit() {
    }
    ~tfOpConverterSuit();

private:
    static tfOpConverterSuit *global;
    std::map<std::string, tfOpConverter *> mTests;
};

template <class T>
class tfOpConverterRegister {
public:
    tfOpConverterRegister(const char *claim) {
        T *test               = new T;
        tfOpConverterSuit *ts = tfOpConverterSuit::get();
        ts->insert(test, claim);
    }
    ~tfOpConverterRegister() {
    }
};

#define DECLARE_OP_CONVERTER(name)                                                \
    class name : public tfOpConverter {                                           \
    public:                                                                       \
        virtual void run(MNN::OpT *dstOp, TmpNode *srcNode); \
        name() {                                                                  \
        }                                                                         \
        virtual ~name() {                                                         \
        }                                                                         \
        virtual MNN::OpType opType();                                             \
        virtual MNN::OpParameter type();                                          \
    }

#define REGISTER_CONVERTER(name, opType) static tfOpConverterRegister<name> _Convert_##opType(#opType)

#endif // TFtfOpCONVERTER_HPP
