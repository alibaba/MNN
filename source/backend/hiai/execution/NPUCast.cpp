//
//  NPUCast.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUCast.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

static ge::DataType mapDataType(DataType src) {
    ge::DataType retVal = ge::DataType::DT_UNDEFINED;
    switch (src) {
        case DataType_DT_FLOAT:
            retVal = ge::DataType::DT_FLOAT;
            break;
        case DataType_DT_DOUBLE:
            retVal = ge::DataType::DT_DOUBLE;
            break;
        case DataType_DT_INT32:
            retVal = ge::DataType::DT_INT32;
            break;
        case DataType_DT_UINT8:
            retVal = ge::DataType::DT_UINT8;
            break;
        case DataType_DT_INT16:
            retVal = ge::DataType::DT_INT16;
            break;
        case DataType_DT_INT8:
            retVal = ge::DataType::DT_INT8;
            break;
        case DataType_DT_INT64:
            retVal = ge::DataType::DT_INT64;
            break;
        default:
            MNN_ASSERT(false);
            printf("cast Datatype : %d \n", src);
            break;
    }
    return retVal;
}

NPUCast::NPUCast(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUCast::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    shared_ptr<ge::op::Cast> castOp(new ge::op::Cast(opName));

    auto xOp = mNpuBackend->getInputOps(mOp);
    auto castPara = mOp->main_as_CastParam();
    DataType srcT = castPara->srcT();
    DataType dstT = castPara->dstT();

    (*castOp)
        .set_input_x(*xOp.get())
        .set_attr_SrcT(mapDataType(srcT)) 
        .set_attr_DstT(mapDataType(dstT));
    mNpuBackend->setOutputOps(mOp, {castOp}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUCast>> __cast_op(OpType_Cast);

} // namespace MNN