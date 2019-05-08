//
//  SplitTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SplitTf);

MNN::OpType SplitTf::opType() {
    return MNN::OpType_Slice;
}
MNN::OpParameter SplitTf::type() {
    return MNN::OpParameter_Slice;
}

void SplitTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto splitParam = new MNN::SliceT;

    splitParam->sourceType = MNN::NetSource_TENSORFLOW;

    tensorflow::AttrValue value;
    DCHECK(2 == srcNode->inEdges.size()) << "INPUT ERROR: Split should have two inputs ==> " << srcNode->opName;
    TmpNode *axisNode = tempGraph->_getTmpNode(srcNode->inEdges[0]);
    DCHECK("Const" == axisNode->opType) << "INPUT ERROR: Split should have one Const node input";
    splitParam->axis = 0;
    if (find_attr_value(axisNode->tfNode, "value", value)) {
        splitParam->axis = value.tensor().int_val(0);
    }

    if (find_attr_value(srcNode->tfNode, "num_split", value)) {
        auto numSplitsTensor = value.tensor();
        size_t dimSize       = numSplitsTensor.tensor_shape().dim_size();
        size_t dataSize      = 1;
        for (int i = 0; i < dimSize; i++) {
            dataSize *= numSplitsTensor.tensor_shape().dim(i).size();
        }

        if (1 == dataSize) {
            // scalar
            splitParam->slicePoints.resize(1);
            splitParam->slicePoints[0] = value.i();
        } else {
            // one dimension tensor
            int *tempIntData = (int *)numSplitsTensor.tensor_content().data();
            splitParam->slicePoints.resize(dataSize);
            for (int i = 0; i < dataSize; i++) {
                splitParam->slicePoints[i] = tempIntData[i];
            }
        }
    }

    dstOp->main.value = splitParam;
}

REGISTER_CONVERTER(SplitTf, Split);

DECLARE_OP_CONVERTER(SplitVTf);

MNN::OpType SplitVTf::opType() {
    return MNN::OpType_Slice;
}
MNN::OpParameter SplitVTf::type() {
    return MNN::OpParameter_Slice;
}

void SplitVTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto splitvParam = new MNN::SliceT;

    splitvParam->sourceType = MNN::NetSource_TENSORFLOW;
    DCHECK(3 == srcNode->inEdges.size()) << "INPUT ERROR: SplitV should have three inputs ==> " << srcNode->opName;

    tensorflow::AttrValue value;

    int numSplits = 0;
    if (find_attr_value(srcNode->tfNode, "num_split", value)) {
        numSplits = value.i();
    }

    auto sizeSplitsNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    DCHECK("Const" == sizeSplitsNode->opType) << "sizeSplitsNode should be Const";
    if (find_attr_value(sizeSplitsNode->tfNode, "value", value)) {
        auto sizeSplitTensor = value.tensor();
        size_t dimSize       = sizeSplitTensor.tensor_shape().dim_size();
        DCHECK(dimSize == 1) << "one dimension tensor";
        const int dataSize = sizeSplitTensor.tensor_shape().dim(0).size();
        DCHECK(dataSize == numSplits);
        auto tempIntData = reinterpret_cast<const int *>(sizeSplitTensor.tensor_content().data());
        splitvParam->slicePoints.resize(dataSize);
        for (int i = 0; i < dataSize; ++i) {
            splitvParam->slicePoints[i] = tempIntData[i];
        }
    }

    // split_dim
    auto splitDimNode = tempGraph->_getTmpNode(srcNode->inEdges[2]);
    DCHECK("Const" == splitDimNode->opType) << "split dim node should be Const";
    splitvParam->axis = 0;
    if (find_attr_value(splitDimNode->tfNode, "value", value)) {
        auto si = value.tensor().int_val_size();
        DCHECK(1 == si) << "split_dim is scalar";
        splitvParam->axis = value.tensor().int_val(0);
    }

    dstOp->main.value = splitvParam;
}
REGISTER_CONVERTER(SplitVTf, SplitV);
