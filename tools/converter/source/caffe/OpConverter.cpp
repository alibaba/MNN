//
//  OpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include <stdlib.h>

OpConverterSuit* OpConverterSuit::global = nullptr;
class DefaultConverter : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) override {
        dstOp->main.value = new MNN::ExtraT;
        dstOp->main.AsExtra()->engine = "Caffe";
        dstOp->main.AsExtra()->type = parameters.type();
    }
    virtual MNN::OpParameter type() override {
        return MNN::OpParameter_Extra;
    }
    virtual MNN::OpType opType() override {
        return MNN::OpType_Extra;
    }
    
private:
};

OpConverter* OpConverterSuit::search(const std::string& name) {
    auto iter = mTests.find(name);
    if (iter == mTests.end()) {
        static DefaultConverter converter;
        return &converter;
    }
    return iter->second;
}

OpConverterSuit* OpConverterSuit::get() {
    if (global == nullptr)
        global = new OpConverterSuit;
    return global;
}

OpConverterSuit::~OpConverterSuit() {
    for (auto& iter : mTests) {
        delete iter.second;
    }
    mTests.clear();
}

void OpConverterSuit::insert(OpConverter* t, const char* name) {
    mTests.insert(std::make_pair(name, t));
}
