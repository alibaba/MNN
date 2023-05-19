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
    shared_ptr<hiai::op::Reshape> reshape(new hiai::op::Reshape(opName));
    
    auto shapeFormt = tensorShapeFormat(outputs[0]);
    std::vector<int32_t> shape(shapeFormt.begin(), shapeFormt.end());
    shapeConst = hiai::op::Const(opName + "_shape_const");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shape.size())}), 
            ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)shape.data(), shape.size() * sizeof(int32_t));

        shapeConst.set_attr_value(filter);
    }
 
    auto inputDims = tensorShapeFormat(inputs[0]);
    auto shapeDims = tensorShapeFormat(outputs[0]);
    
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    auto xOp        = iops.back().first;

    if ((TensorUtils::getDescribe(input)->dimensionFormat != MNN::MNN_DATA_FORMAT_NHWC) ||
        (isSameDims(input, outputs[0]) || (inputDims == shapeDims))) {
        (*reshape).set_input_x(*xOp).set_input_shape(shapeConst);
        mNpuBackend->setOutputOps(mOp, {reshape}, outputs);
    } else {
        shared_ptr<hiai::op::Permute> permute1(new hiai::op::Permute(opName+"_perm1"));
        shared_ptr<hiai::op::Permute> permute2(new hiai::op::Permute(opName+"_perm2"));
        (*permute1)
            .set_input_x(*xOp.get())
            .set_attr_order(ge::AttrValue::LIST_INT({0,2,3,1}));
        vector<int32_t> nhwcShape = {static_cast<int32_t>(shapeDims[0]), static_cast<int32_t>(shapeDims[2]),
            static_cast<int32_t>(shapeDims[3]), static_cast<int32_t>(shapeDims[1])}; 
        nhwshapeConst = hiai::op::Const(opName + "_nhwshape_const");
        {
            ge::TensorDesc fdesc(ge::Shape({4}), ge::FORMAT_NCHW,  ge::DT_INT32);
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)nhwcShape.data(), nhwcShape.size() * sizeof(int32_t));

            nhwshapeConst.set_attr_value(filter);
        }
        (*reshape)
            .set_input_x(*permute1.get())
            .set_input_shape(nhwshapeConst);
        (*permute2)
            .set_input_x(*reshape.get())
            .set_attr_order(ge::AttrValue::LIST_INT({0,3,1,2}));
        mNpuBackend->setOutputOps(mOp, {permute1,reshape,permute2}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUReshape>> __reshape_op(OpType_Reshape);

} // namespace MNN