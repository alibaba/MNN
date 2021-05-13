//
//  NPUReshape.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUReshape.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUReshape::NPUReshape(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

static bool isSameDims(Tensor * input,Tensor * output)
{
    if(input->buffer().dimensions == output->buffer().dimensions)
    {
        for(auto i =0; i < input->buffer().dimensions; i++) {
            if(input->buffer().dim[i].extent != output->buffer().dim[i].extent) {
                return false;
            }
        }
        return true;
    }
    return false;
}

ErrorCode NPUReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto input = inputs[0];
    auto opName = mOp->name()->str();
    shared_ptr<ge::op::Reshape> reshape(new ge::op::Reshape(opName));
 
    auto inputDims = tensorShapeFormat(inputs[0]);
    auto shapeDims = tensorShapeFormat(outputs[0]);
    
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    auto xOp        = iops.back().first;

    //MNN_PRINT("input dim:%d %d %d %d\n", inputDims[0], inputDims[1], inputDims[2], inputDims[3]);
    //MNN_PRINT("output dim:%d %d %d %d\n", shapeDims[0], shapeDims[1], shapeDims[2], shapeDims[3]);
    //MNN_PRINT("input->dimensionFormat:%d\n", TensorUtils::getDescribe(input)->dimensionFormat);
    //MNN_PRINT("output->dimensionFormat:%d\n", TensorUtils::getDescribe(outputs[0])->dimensionFormat);
    if ((TensorUtils::getDescribe(input)->dimensionFormat != MNN::MNN_DATA_FORMAT_NHWC) ||
        (isSameDims(input, outputs[0]) || (inputDims == shapeDims))) {
        (*reshape).set_input_tensor(*xOp).set_attr_shape(ge::AttrValue::LIST_INT(shapeDims));
        mNpuBackend->setOutputOps(mOp, {reshape}, outputs);
    } else {
        shared_ptr<ge::op::Permute> permute1(new ge::op::Permute(opName+"_perm1"));
        shared_ptr<ge::op::Permute> permute2(new ge::op::Permute(opName+"_perm2"));
        (*permute1)
            .set_input_x(*xOp.get())
            .set_attr_order(ge::AttrValue::LIST_INT({0,2,3,1}));
        vector<int64_t> nhwcShape = {shapeDims[0],shapeDims[2],shapeDims[3],shapeDims[1]};
        (*reshape)
            .set_input_tensor(*permute1.get())
            .set_attr_shape(ge::AttrValue::LIST_INT(nhwcShape));
        (*permute2)
            .set_input_x(*reshape.get())
            .set_attr_order(ge::AttrValue::LIST_INT({0,3,1,2}));
        mNpuBackend->setOutputOps(mOp, {permute1,reshape,permute2}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUReshape>> __reshape_op(OpType_Reshape);

} // namespace MNN