//
//  liteOpConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LITEOPCONVERTER_HPP
#define LITEOPCONVERTER_HPP

#include <map>
// MNN fbs header
#include "MNN_generated.h"
// tflite fbs header
#include "schema_generated.h"

#include "logkit.h"

class liteOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                     const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                     const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                     const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) = 0;
    virtual MNN::OpParameter type(bool quantizedModel) = 0;
    virtual MNN::OpType opType(bool quantizedModel)    = 0;
    liteOpConverter() {
    }
    virtual ~liteOpConverter() {
    }

    friend class liteOpConverterSuit;
};

class liteOpConverterSuit {
public:
    static liteOpConverterSuit* get();
    void insert(liteOpConverter* t, const tflite::BuiltinOperator opIndex);
    liteOpConverter* search(const tflite::BuiltinOperator opIndex);

    liteOpConverterSuit() {
    }
    ~liteOpConverterSuit();

private:
    static liteOpConverterSuit* _uniqueSuit;
    std::map<tflite::BuiltinOperator, liteOpConverter*> _liteOpConverters;
};

template <class T>
class liteOpConverterRegister {
public:
    liteOpConverterRegister(const tflite::BuiltinOperator opIndex) {
        T* converter                  = new T;
        liteOpConverterSuit* liteSuit = liteOpConverterSuit::get();
        liteSuit->insert(converter, opIndex);
    }

    ~liteOpConverterRegister() {
    }
};

#define DECLARE_OP_COVERTER(name)                                                                                      \
    class name : public liteOpConverter {                                                                              \
    public:                                                                                                            \
        virtual void run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,                          \
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,                           \
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,                       \
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel); \
        name() {                                                                                                       \
        }                                                                                                              \
        virtual ~name() {                                                                                              \
        }                                                                                                              \
        virtual MNN::OpParameter type(bool quantizedModel);                                                            \
        virtual MNN::OpType opType(bool quantizedModel);                                                               \
    }

#define REGISTER_CONVERTER(name, opType) static liteOpConverterRegister<name> _Convert##opType(opType)

#endif // LITEOPCONVERTER_HPP
