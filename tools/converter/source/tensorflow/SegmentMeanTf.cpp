#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SegmentMeanTf);

MNN::OpType SegmentMeanTf::opType() {
    return MNN::OpType_Segment;
}
MNN::OpParameter SegmentMeanTf::type() {
    return MNN::OpParameter_ReductionParam;
}

void SegmentMeanTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = new MNN::ReductionParamT;
    dstOp->main.AsReductionParam()->operation = MNN::ReductionType_MEAN;
}

REGISTER_CONVERTER(SegmentMeanTf, SegmentMean);
