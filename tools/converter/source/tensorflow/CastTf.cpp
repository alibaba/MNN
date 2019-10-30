//
//  CastTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(CastTf);

MNN::OpType CastTf::opType() {
    return MNN::OpType_Cast;
}
MNN::OpParameter CastTf::type() {
    return MNN::OpParameter_CastParam;
}

void CastTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::CastParamT;
    tensorflow::AttrValue value;
    parameter->dstT = MNN::DataType_DT_INVALID;
    parameter->srcT = MNN::DataType_DT_INVALID;
    if (find_attr_value(srcNode->tfNode, "DstT", value)) {
        parameter->dstT = (MNN::DataType)value.type();
    }
    if (find_attr_value(srcNode->tfNode, "SrcT", value)) {
        parameter->srcT = (MNN::DataType)value.type();
    }
    DCHECK(parameter->srcT != MNN::DataType_DT_INVALID && parameter->dstT != MNN::DataType_DT_INVALID)
        << "Cast Parameter ERROR!!! ===> " << srcNode->opName;

    dstOp->main.value = parameter;
}

REGISTER_CONVERTER(CastTf, Cast);
