//
//  NPUBinary.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBinary.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

template<class T>
void NPUBinary::BinaryCastIR(string opName, hiai::Operator& input0, hiai::Operator& input1,
    const std::vector<Tensor *>& outputs, int activationType, shared_ptr<T> binary) {
    shared_ptr<hiai::op::CastT> castTOp(new hiai::op::CastT(opName + "castTOp"));
    shared_ptr<hiai::op::CastT> castTOp1(new hiai::op::CastT(opName + "castTOp1"));
    shared_ptr<hiai::op::CastT> castTOpAfter(new hiai::op::CastT(opName + "castTOpAfter"));
    auto binaryParam = mOp->main_as_BinaryOp();
    auto t = binaryParam->T();
    if (flag0) {
        (*castTOp)
            .set_input_x(input0.GetOutput(mNpuBackend->mSclipMap[inputIndex0]))
            .set_attr_dst_dtype(0);
        (*binary).set_input_x1(*castTOp.get());
    } else {
        (*castTOp)
            .set_input_x(input0)
            .set_attr_dst_dtype(0);
        (*binary).set_input_x1(*castTOp.get());
    }
    if (flag1) {
        (*castTOp1)
            .set_input_x(input1.GetOutput(mNpuBackend->mSclipMap[inputIndex1]))
            .set_attr_dst_dtype(0);
        (*binary).set_input_x2(*castTOp1.get());
    } else {
        (*castTOp1)
            .set_input_x(input1)
            .set_attr_dst_dtype(0);
        (*binary).set_input_x2(*castTOp1.get());
    }
    (*castTOpAfter)
            .set_input_x(*binary.get())
            .set_attr_dst_dtype(mapDataType(t));
    if(activationType == 1) {
        shared_ptr<hiai::op::Activation> binary_activation(new hiai::op::Activation(opName + "_Relu"));
        (*binary_activation)
            .set_input_x(*castTOpAfter.get())
            .set_attr_mode(1);
        mNpuBackend->setOutputOps(mOp, {castTOp, castTOp1, binary, castTOpAfter, binary_activation}, outputs);
    } else {
        mNpuBackend->setOutputOps(mOp, {castTOp, castTOp1, binary, castTOpAfter}, outputs);
    }
}
template<class T>
void NPUBinary::BinaryIR(string opName, hiai::Operator& input0, hiai::Operator& input1,
    const std::vector<Tensor *>& outputs, int activationType, shared_ptr<T> binary) {
    if (flag0) {
        (*binary).set_input_x1(input0.GetOutput(mNpuBackend->mSclipMap[inputIndex0]));
    } else {
        (*binary).set_input_x1(input0);
    }
    if (flag1) {
        (*binary).set_input_x2(input1.GetOutput(mNpuBackend->mSclipMap[inputIndex1]));
    } else {
        (*binary).set_input_x2(input1);
    }
    if(activationType == 1) {
        shared_ptr<hiai::op::Activation> binary_activation(new hiai::op::Activation(opName + "_Relu"));
        (*binary_activation)
            .set_input_x(*binary.get())
            .set_attr_mode(1);
        mNpuBackend->setOutputOps(mOp, {binary, binary_activation}, outputs);
    } else {
        mNpuBackend->setOutputOps(mOp, {binary}, outputs);
    }
}

void NPUBinary::OpInsert(int binary_type, string opName,
                         hiai::Operator& input0, hiai::Operator& input1,
                         const std::vector<Tensor *> &outputs, int activationType){

    if (binary_type == BinaryOpOperation_ADD) {
        shared_ptr<hiai::op::Add> binary(new hiai::op::Add(opName));
        BinaryCastIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_MUL) {
        shared_ptr<hiai::op::Mul> binary(new hiai::op::Mul(opName));
        BinaryCastIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_REALDIV) {
        shared_ptr<hiai::op::RealDiv> binary(new hiai::op::RealDiv(opName));
        BinaryCastIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_SUB) {
        shared_ptr<hiai::op::Sub> binary(new hiai::op::Sub(opName));
        BinaryCastIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_MINIMUM) {
        shared_ptr<hiai::op::Minimum> binary(new hiai::op::Minimum(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_MAXIMUM) {
        shared_ptr<hiai::op::Maximum> binary(new hiai::op::Maximum(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_EQUAL) {
        shared_ptr<hiai::op::Equal> binary(new hiai::op::Equal(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_LESS_EQUAL) {
        shared_ptr<hiai::op::LessEqual> binary(new hiai::op::LessEqual(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_POW) {
        shared_ptr<hiai::op::Pow> binary(new hiai::op::Pow(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_LESS) {
        shared_ptr<hiai::op::Less> binary(new hiai::op::Less(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_MOD) {
        shared_ptr<hiai::op::FloorMod> binary(new hiai::op::FloorMod(opName));
        BinaryCastIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_SquaredDifference) {
        shared_ptr<hiai::op::SquaredDifference> binary(new hiai::op::SquaredDifference(opName));
        BinaryCastIR(opName, input0, input1, outputs, activationType, binary);
    } else if (binary_type == BinaryOpOperation_GREATER) {
        shared_ptr<hiai::op::Greater> binary(new hiai::op::Greater(opName));
        BinaryIR(opName, input0, input1, outputs, activationType, binary);
    } else {
        MNN_ERROR("npu binary not support type : %d \n", binary_type);
        MNN_ASSERT(false);
    }
}

NPUBinary::NPUBinary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    auto opName = mOp->name()->str();
    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    auto binary_type = mOp->main_as_BinaryOp()->opType();
    auto len = mOp->inputIndexes()->size();
    Tensor* input = nullptr;
    if(isConst0 && !isConst1) {
        input = inputs[0];
    } else if (!isConst0 && isConst1) {
        input = inputs[1];
    }
    mConst = hiai::op::Const(opName + "_w_const");
    if(input != nullptr) {
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        vector<int64_t> dims;
        for (int32_t i = 0; i < input->buffer().dimensions; i++) {
            dims.push_back(input->buffer().dim[i].extent);
        }
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
        if (input->getType().code == halide_type_float) {
            filter->SetData((uint8_t *)input->host<float>(), input->elementSize() * sizeof(float));
        }
        if (input->getType().code == halide_type_int && input->getType().bits == 32) {
            fdesc.SetDataType(ge::DT_INT32);
            filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        }
        filter->SetTensorDesc(fdesc);
        mConst.set_attr_value(filter);
    }
}

ErrorCode NPUBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    auto binary_type = mOp->main_as_BinaryOp()->opType();
    int activationType = mOp->main_as_BinaryOp()->activationType();
    flag0 = false;
    flag1 = false;
    if (!isConst0 && isConst1) {
        inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0 = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0 = iops0.back().first;
        if (mNpuBackend->mSclipMap.find(inputIndex0) != mNpuBackend->mSclipMap.end()) {
            flag0 = true;
        }
        inputIndex1 = -1;
        OpInsert(binary_type, opName, *xOp0.get(), mConst, outputs, activationType);
    } else if(isConst0 && !isConst1) {
        inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1 = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1 = iops1.back().first;
        if (mNpuBackend->mSclipMap.find(inputIndex1) != mNpuBackend->mSclipMap.end()) {
            flag1 = true;
        }
        inputIndex0 = -1;
        OpInsert(binary_type, opName, mConst, *xOp1.get(), outputs, activationType);  
    } else {
        inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0 = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0 = iops0.back().first;
        inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1 = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1 = iops1.back().first;
        if (mNpuBackend->mSclipMap.find(inputIndex0) != mNpuBackend->mSclipMap.end()) {
            flag0 = true;
        }
        if (mNpuBackend->mSclipMap.find(inputIndex1) != mNpuBackend->mSclipMap.end()) {
            flag1 = true;
        }
        OpInsert(binary_type, opName, *xOp0.get(), *xOp1.get(), outputs, activationType);
    }
    return NO_ERROR;
}


NPUCreatorRegister<TypedCreator<NPUBinary>> __bianry_op(OpType_BinaryOp);

} // namespace MNN