#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class SegmentMeanSizeComputer : public SizeComputer{
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) const override{
        // printf("Begin to Compute Shape of SegmentMean!\n");
        auto dataDim = inputs[0]->dimensions();
        auto segmentIds = inputs[1];
        auto output = outputs[0];
        output->buffer().dimensions = dataDim;
        int S = inputs[0]->length(0);
        output->setLength(0, inputs[1]->host<int>()[S-1]+1);
        for(int i=1; i<dataDim; i++) {
            output->setLength(i, inputs[0]->length(i));
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(SegmentMeanSizeComputer, OpType_Segment, std::vector<int>{1});
}//namespace MNN
