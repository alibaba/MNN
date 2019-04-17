//
//  BatchToSpaceNDTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(BatchToSpaceNDTf);

MNN::OpType BatchToSpaceNDTf::opType() {
    return MNN::OpType_BatchToSpaceND;
}
MNN::OpParameter BatchToSpaceNDTf::type() {
    return MNN::OpParameter_SpaceBatch;
}

void BatchToSpaceNDTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    DCHECK(srcNode->inEdges.size() == 3) << "BatchToSpaceND Input Node's Num ERROR";
    auto spacebatch = new MNN::SpaceBatchT;

    auto block_shape      = new MNN::BlobT;
    block_shape->dataType = MNN::DataType_DT_INT32;

    auto paddings      = new MNN::BlobT;
    paddings->dataType = MNN::DataType_DT_INT32;

    tensorflow::AttrValue weightsValue;
    if (find_attr_value(srcNode->tfNode, "Tblock_shape", weightsValue)) {
        block_shape->dataType = static_cast<MNN::DataType>(weightsValue.type());
    }
    if (find_attr_value(srcNode->tfNode, "Tpaddings", weightsValue)) {
        paddings->dataType = static_cast<MNN::DataType>(weightsValue.type());
    }

    DCHECK(block_shape->dataType == MNN::DataType_DT_INT32) << "BlockShape Data Type ERROR!";
    DCHECK(paddings->dataType == MNN::DataType_DT_INT32) << "BlockShape Data Type ERROR!";

    auto blockShapeTensor = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    find_attr_value(blockShapeTensor->tfNode, "value", weightsValue);
    const auto dimSize = weightsValue.tensor().tensor_shape().dim_size();
    block_shape->dims.resize(dimSize);
    int dataSize = 1;
    for (int i = 0; i < dimSize; ++i) {
        dataSize *= weightsValue.tensor().tensor_shape().dim(i).size();
        block_shape->dims[i] = weightsValue.tensor().tensor_shape().dim(i).size();
    }

    auto tensor_content = reinterpret_cast<const int *>(weightsValue.tensor().tensor_content().data());
    block_shape->int32s.resize(dataSize);
    ::memcpy(block_shape->int32s.data(), tensor_content, sizeof(int) * dataSize);

    auto paddingTensor = tempGraph->_getTmpNode(srcNode->inEdges[2]);
    find_attr_value(paddingTensor->tfNode, "value", weightsValue);
    const auto dim = weightsValue.tensor().tensor_shape().dim_size();
    paddings->dims.resize(dim);

    dataSize = 1;
    for (int i = 0; i < dim; ++i) {
        dataSize *= weightsValue.tensor().tensor_shape().dim(i).size();
        paddings->dims[i] = weightsValue.tensor().tensor_shape().dim(i).size();
    }

    auto paddingData = reinterpret_cast<const int *>(weightsValue.tensor().tensor_content().data());
    paddings->int32s.resize(dataSize);
    ::memcpy(paddings->int32s.data(), paddingData, sizeof(int) * dataSize);

    spacebatch->blockShape = std::unique_ptr<MNN::BlobT>(block_shape);
    spacebatch->padding    = std::unique_ptr<MNN::BlobT>(paddings);

    dstOp->main.value = spacebatch;
}

REGISTER_CONVERTER(BatchToSpaceNDTf, BatchToSpaceND);
