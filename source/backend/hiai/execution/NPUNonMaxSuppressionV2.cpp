//
//  NPUNonMaxSuppressionV2.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUNonMaxSuppressionV2.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUNonMaxSuppressionV2::NPUNonMaxSuppressionV2(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    
}

ErrorCode NPUNonMaxSuppressionV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::SplitD> slice(new hiai::op::SplitD(opName));

    auto param = mOp->main_as_Slice();
    auto axis = param->axis();
    if (axis < 0) {
        axis = axis + inputs[0]->dimensions();
    }

    // 
    auto inputIndex1 = mOp->inputIndexes()->data()[0];
    auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
    auto xOp1        = iops1.back().first;

    (*slice)
        .set_input_x(*xOp1.get())
        .set_attr_split_dim(axis);

    /*
     * add map
     * */
    vector<pair<shared_ptr<hiai::Operator>, string>> ops;
    ops.emplace_back(make_pair(slice, ""));

    for (size_t i = 0; i < mOp->outputIndexes()->size(); i++){
        auto index = mOp->outputIndexes()->data()[i];
        mNpuBackend->mGrapMap.insert(make_pair(index, ops));
    }
    
    return NO_ERROR;
}


NPUCreatorRegister<TypedCreator<NPUNonMaxSuppressionV2>> __nonmaxsuppressionV2_op(OpType_NonMaxSuppressionV2);

} // namespace MNN