//
//  StridedSliceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(StridedSliceTf);

MNN::OpType StridedSliceTf::opType() {
    return MNN::OpType_StridedSlice;
}
MNN::OpParameter StridedSliceTf::type() {
    return MNN::OpParameter_StridedSliceParam;
}

void StridedSliceTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto stridedslice = new MNN::StridedSliceParamT;

    tensorflow::AttrValue value;
    find_attr_value(srcNode->tfNode, "begin_mask", value);
    stridedslice->beginMask = value.i();

    find_attr_value(srcNode->tfNode, "end_mask", value);
    stridedslice->endMask = value.i();

    find_attr_value(srcNode->tfNode, "ellipsis_mask", value);
    stridedslice->ellipsisMask = value.i();

    find_attr_value(srcNode->tfNode, "new_axis_mask", value);
    stridedslice->newAxisMask = value.i();

    find_attr_value(srcNode->tfNode, "shrink_axis_mask", value);
    stridedslice->shrinkAxisMask = value.i();

    find_attr_value(srcNode->tfNode, "Index", value);
    stridedslice->Index = (MNN::DataType)value.type();

    find_attr_value(srcNode->tfNode, "T", value);
    stridedslice->T = (MNN::DataType)value.type();

    dstOp->main.value = stridedslice;
}

REGISTER_CONVERTER(StridedSliceTf, StridedSlice);
