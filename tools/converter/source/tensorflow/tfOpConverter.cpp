//
//  tfOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "tfOpConverter.hpp"
#include <stdlib.h>

tfOpConverterSuit *tfOpConverterSuit::global = nullptr;
class DefaultConverter : public tfOpConverter {
public:
    virtual void run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) override {
        dstOp->main.value = new MNN::ExtraT;
        dstOp->main.AsExtra()->engine = "Tensorflow";
        dstOp->main.AsExtra()->type = srcNode->opType;
    }
    virtual MNN::OpParameter type() override {
        return MNN::OpParameter_Extra;
    }
    virtual MNN::OpType opType() override {
        return MNN::OpType_Extra;
    }
    
private:
};
tfOpConverter *tfOpConverterSuit::search(const std::string &name) {
    auto iter = mTests.find(name);
    if (iter == mTests.end()) {
        static DefaultConverter converter;
        return &converter;
    }
    return iter->second;
}

tfOpConverterSuit *tfOpConverterSuit::get() {
    if (global == nullptr)
        global = new tfOpConverterSuit;
    return global;
}

tfOpConverterSuit::~tfOpConverterSuit() {
    for (auto &iter : mTests) {
        delete iter.second;
    }
    mTests.clear();
}

void tfOpConverterSuit::insert(tfOpConverter *t, const char *name) {
    mTests.insert(std::make_pair(name, t));
}
