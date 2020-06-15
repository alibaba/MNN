//
//  ResizeBilinearTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(InterpTf);

MNN::OpType InterpTf::opType() {
    return MNN::OpType_Interp;
}
MNN::OpParameter InterpTf::type() {
    return MNN::OpParameter_Interp;
}

void InterpTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto interpParam = new MNN::InterpT;

    tensorflow::AttrValue value;
#ifdef TF_CONVERT_ORIGIN
    TmpNode *constShapeNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    // ResizeBilinear's input shape could be computed at the runtime
    if (constShapeNode->opType == "Const") {
        if (find_attr_value(constShapeNode->tfNode, "value", value)) {
            const tensorflow::TensorProto &sizeTensor = value.tensor();
            const std::string tensor_content          = sizeTensor.tensor_content();
            if (!tensor_content.empty()) {
                assert(tensor_content.size() >= sizeof(int));
                int h = *(int *)tensor_content.data();
                int w = h;
                if (tensor_content.size() >= sizeof(int) * 2) {
                    w = *(int *)(tensor_content.data() + sizeof(int));
                }
                interpParam->outputHeight = h;
                interpParam->outputWidth  = w;
            } else {
                CHECK(sizeTensor.tensor_shape().dim_size() == 2)
                    << "Resize op Parameter ERROR!!! ===> " << srcNode->opName;
                const int *sizeData       = sizeTensor.int_val().data();
                interpParam->outputHeight = sizeData[0];
                interpParam->outputWidth  = sizeData[1];
            }
        }
    }
#endif
    interpParam->alignCorners = false; // defalut false
    if (find_attr_value(srcNode->tfNode, "align_corners", value)) {
        interpParam->alignCorners = value.b();
    }
    
    interpParam->halfPixelCenters = false; // defalut false
    if (find_attr_value(srcNode->tfNode, "half_pixel_centers", value)) {
        interpParam->halfPixelCenters = value.b();
    }

    // TODO defalut
    interpParam->widthScale  = 1.0;
    interpParam->heightScale = 1.0;
    // 1:near 2: bilinear 3: cubic
    if (srcNode->opType == "ResizeNearestNeighbor") {
        interpParam->resizeType = 1;
    } else {
        interpParam->resizeType = 2;
    }

    dstOp->main.value = interpParam;

#ifdef TF_CONVERT_ORIGIN
    // delete the const input edges!!! Must to do
    // Const node, others no delete
    if (constShapeNode->opType == "Const") {
        const std::vector<std::string>::iterator it2delete = srcNode->inEdges.begin() + 1;
        srcNode->inEdges.erase(it2delete);
        DCHECK(srcNode->inEdges.size() == 1) << "Resize op Input ERROR!!! ===> " << srcNode->opName;
    }
#endif
}

REGISTER_CONVERTER(InterpTf, ResizeBilinear);
REGISTER_CONVERTER(InterpTf, ResizeNearestNeighbor);
