//
//  CPURange.cpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPURange.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/FileLoader.hpp"
namespace MNN {
class CPUExternalConst : public Execution {
public:
    CPUExternalConst(const Op* op, Backend* bn) : Execution(bn) {
        auto blob = op->main_as_Blob();
        mExternalFile = op->externalPath()->str();
        if (nullptr != blob->external()) {
            mOffset = blob->external()->data()[0];
            mSize = blob->external()->data()[1];
        }
    }
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        FileLoader l(mExternalFile.c_str());
        l.offset(mOffset);
        l.read(outputs[0]->host<char>(), mSize);
        return NO_ERROR;
    }
private:
    std::string mExternalFile;
    int64_t mOffset = 0;
    int64_t mSize = 0;
};

class CPUExternalConstCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (op->externalPath() == nullptr) {
            return nullptr;
        }
        return new CPUExternalConst(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUExternalConstCreator, OpType_Const);
REGISTER_CPU_OP_CREATOR(CPUExternalConstCreator, OpType_TrainableParam);
} // namespace MNN
