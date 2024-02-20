//
//  NPUEltwise.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUEltwise.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUEltwise::NPUEltwise(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    
}

ErrorCode NPUEltwise::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto param = mOp->main_as_Eltwise();
    auto coffs = param->coeff();
    if (param->type() == EltwiseType_SUM && coffs == nullptr) {
        auto inputIndex1 = mOp->inputIndexes()->data()[0];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        auto inputIndex2 = mOp->inputIndexes()->data()[1];
        auto iops2       = mNpuBackend->mGrapMap[inputIndex2]; // x
        auto xOp2        = iops2.back().first;
        shared_ptr<hiai::op::Add> sub(new hiai::op::Add(opName));
        if (mNpuBackend->mSclipMap.find(inputIndex1) == mNpuBackend->mSclipMap.end()) {
            (*sub).set_input_x1(*xOp1.get());
        } else {
            (*sub).set_input_x1(xOp1->GetOutput(mNpuBackend->mSclipMap[inputIndex1]));
        }
        if (mNpuBackend->mSclipMap.find(inputIndex2) == mNpuBackend->mSclipMap.end()) {
            (*sub).set_input_x2(*xOp2.get());
        } else {
            (*sub).set_input_x2(xOp2->GetOutput(mNpuBackend->mSclipMap[inputIndex2]));
        }
        mNpuBackend->setOutputOps(mOp, {sub}, outputs);
    } else if (param->type() == EltwiseType_SUB && coffs == nullptr) {
        auto inputIndex1 = mOp->inputIndexes()->data()[0];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        auto inputIndex2 = mOp->inputIndexes()->data()[1];
        auto iops2       = mNpuBackend->mGrapMap[inputIndex2]; // x
        auto xOp2        = iops2.back().first;
        shared_ptr<hiai::op::Sub> sub(new hiai::op::Sub(opName));
        if (mNpuBackend->mSclipMap.find(inputIndex1) == mNpuBackend->mSclipMap.end()) {
            (*sub).set_input_x1(*xOp1.get());
        } else {
            (*sub).set_input_x1(xOp1->GetOutput(mNpuBackend->mSclipMap[inputIndex1]));
        }
        if (mNpuBackend->mSclipMap.find(inputIndex2) == mNpuBackend->mSclipMap.end()) {
            (*sub).set_input_x2(*xOp2.get());
        } else {
            (*sub).set_input_x2(xOp2->GetOutput(mNpuBackend->mSclipMap[inputIndex2]));
        }
        mNpuBackend->setOutputOps(mOp, {sub}, outputs);
    } else {
        vector<float> coffAttr;
        if (coffs != nullptr) {
            for (int32_t j = 0; j < coffs->size(); j++) {
                coffAttr.push_back(coffs->Get(j));
            }
        }
        auto inputSize = mOp->inputIndexes()->size();
        shared_ptr<hiai::op::Eltwise> eltwise(new hiai::op::Eltwise(opName));
        (*eltwise)
            .create_dynamic_input_x(inputSize)
            .set_attr_N(inputSize)
            .set_attr_mode(param->type());
        for (int i = 0; i < inputSize; ++i) {
            auto inputIndex = mOp->inputIndexes()->data()[i];
            auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
            auto xOp        = iops.back().first;
            if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
                (*eltwise).set_dynamic_input_x(i + 1, *xOp.get());
            } else {
                (*eltwise).set_dynamic_input_x(i + 1, xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
            }
        }
        if (coffAttr.size() > 0) {
            (*eltwise).set_attr_coeff(coffAttr);
        }
        mNpuBackend->setOutputOps(mOp, {eltwise}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUEltwise>> __elewise_op(OpType_Eltwise);

} // namespace MNN