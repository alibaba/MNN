//
//  MomentsTf.cpp
//  MNNConvertor
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(MomentsTf);

MNN::OpType MomentsTf::opType() {
    return MNN::OpType_Moments;
}
MNN::OpParameter MomentsTf::type() {
    return MNN::OpParameter_MomentsParam;
}

void MomentsTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto momentsParam = new MNN::MomentsParamT;

    tensorflow::AttrValue value;
    momentsParam->dType = MNN::DataType_DT_FLOAT;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        momentsParam->dType = static_cast<MNN::DataType>(value.type());
    }

    momentsParam->keepDims = false;
    if (find_attr_value(srcNode->tfNode, "keep_dims", value)) {
        momentsParam->keepDims = value.b();
    }

    auto dimNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    DCHECK(dimNode->opType == "Const") << "Moments should have one Const dim node " << srcNode->opName;
    if (find_attr_value(dimNode->tfNode, "value", value)) {
        const tensorflow::TensorProto &momentsIndices    = value.tensor();
        const tensorflow::TensorShapeProto &momentsShape = momentsIndices.tensor_shape();

        int dimSize = 1;
        if (momentsShape.dim_size() > 0) {
            dimSize = momentsShape.dim(0).size();
        }
        momentsParam->dim.resize(dimSize);
        if (momentsIndices.int_val_size() > 0) {
            for (int i = 0; i < dimSize; ++i) {
                momentsParam->dim[i] = momentsIndices.int_val(i);
            }
        } else {
            DCHECK((MNN::DataType)momentsIndices.dtype() == MNN::DataType_DT_INT32);
            DCHECK(momentsIndices.tensor_content().size() > 0);
            auto dimData = reinterpret_cast<const int *>(momentsIndices.tensor_content().data());
            for (int i = 0; i < dimSize; i++) {
                momentsParam->dim[i] = dimData[i];
            }
        }
    }

    // default set the Moments'layout to NCHW, so change the dimension
    const int axisMap[4] = {0, 2, 3, 1};
    for (int i = 0; i < momentsParam->dim.size(); ++i) {
        momentsParam->dim[i] = axisMap[momentsParam->dim[i]];
    }

    dstOp->main.value = momentsParam;
}

REGISTER_CONVERTER(MomentsTf, Moments);
