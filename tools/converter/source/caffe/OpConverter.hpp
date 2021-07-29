//
//  OpConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpConverter_HPP
#define OpConverter_HPP

#include <map>
#include <string>
#include <vector>
#include "OpCount.hpp"
#include "MNN_generated.h"
#include "caffe.pb.h"

class OpConverter {
    friend class OpConverterSuit;

public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) = 0;
    virtual MNN::OpParameter type()                                                                                 = 0;
    virtual MNN::OpType opType()                                                                                    = 0;
    OpConverter() {
    }
    virtual ~OpConverter() {
    }

private:
};

class OpConverterSuit {
public:
    static OpConverterSuit* get();
    void insert(OpConverter* t, const char* name);

    OpConverter* search(const std::string& name);

    OpConverterSuit() {
    }
    ~OpConverterSuit();

private:
    static OpConverterSuit* global;
    std::map<std::string, OpConverter*> mTests;
};

template <class T>
class OpConverterRegister {
public:
    OpConverterRegister(const char* claim) {
        T* test             = new T;
        OpConverterSuit* ts = OpConverterSuit::get();
        MNN::OpCount::get()->insertOp("CAFFE", claim);
        ts->insert(test, claim);
    }
    ~OpConverterRegister() {
    }
};
#endif // OpConverter_HPP
