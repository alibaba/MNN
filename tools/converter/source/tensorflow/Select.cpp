#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SelectTf);

MNN::OpType SelectTf::opType() {
    return MNN::OpType_Select;
}
MNN::OpParameter SelectTf::type() {
    return MNN::OpParameter_NONE;
}
void SelectTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    //Do nothing
}
REGISTER_CONVERTER(SelectTf, Select);
