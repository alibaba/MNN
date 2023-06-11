//
//  NPUConvertTensor.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUConvertTensor.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUConvertTensor::NPUConvertTensor(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUConvertTensor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto xOp = mNpuBackend->getInputOps(mOp);
     //om input weight const op
    auto shapeFormt = tensorShapeFormat(outputs[0]);
    std::vector<int32_t> shapeDims (shapeFormt.begin(), shapeFormt.end());
    shapeConst = hiai::op::Const(opName + "_shape_const");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shapeDims.size())}), 
            ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)shapeDims.data(), shapeDims.size() * sizeof(int32_t));

        shapeConst.set_attr_value(filter);
    }

    if (outputs[0]->buffer().dimensions==2) { //These conditions require special processing dimensions, not simple reshape, but equivalent transposes
        shared_ptr<hiai::op::Permute> permute1(new hiai::op::Permute(opName));
        (*permute1)
            .set_input_x(*xOp.get())
            .set_attr_order(ge::AttrValue::LIST_INT({2,1,0,3}));
        mNpuBackend->setOutputOps(mOp, {permute1}, outputs);
    } else {
        shared_ptr<hiai::op::Reshape> convertTensor(new hiai::op::Reshape(opName));

        int index = mOp->inputIndexes()->data()[0];
        auto iter = mNpuBackend->mSclipMap.find(index);
        if(iter != mNpuBackend->mSclipMap.end()){
            (*convertTensor).SetInput(0, *xOp, mNpuBackend->mSclipMap[index]);
            (*convertTensor).set_input_shape(shapeConst);
        }else{
            (*convertTensor).set_input_x(*xOp).set_input_shape(shapeConst);
        }
        mNpuBackend->setOutputOps(mOp, {convertTensor}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConvertTensor>> __convert_tensor_op(OpType_ConvertTensor);

} // namespace MNN