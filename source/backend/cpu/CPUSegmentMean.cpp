#include <cmath>

#include "core/Execution.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUBackend.hpp"
#include "MNN_generated.h"


namespace MNN {

class CPUSegmentMean : public Execution {
    int mDim;
public:
    explicit CPUSegmentMean(const MNN::Op* op, Backend * backend);
    ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
}; //class CPUSegmentMean;

CPUSegmentMean::CPUSegmentMean(const MNN::Op* op, Backend* backend): Execution(backend) {
    mDim = 1;
    return;
}//CPUSegmentMean

ErrorCode CPUSegmentMean::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto data = inputs[0];
    auto segmentIds = inputs[1];
    int seq_len = data->length(0);
    int k = 0;
    int c = 0;
    int dim = mDim;
    memset((void *)outputs[0]->host<float>(), 0, outputs[0]->size());
    for (int i=0; i<seq_len; i++){
        if (segmentIds->host<int>()[i] - k == 1){
            for (int j=0; j<dim; j++){ outputs[0]->host<float>()[k * dim + j] /= c;}
            k += 1;
            c = 0;
        }
        for (int j=0; j<dim; j++){ outputs[0]->host<float>()[k * dim + j] += data->host<float>()[i * dim + j]; };
        c += 1;
        if (i == seq_len - 1){ for (int j=0; j<dim; j++){ outputs[0]->host<float>()[k * dim + j] /= c; }}
    }
    return NO_ERROR;
}

ErrorCode CPUSegmentMean::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto data = inputs[0];
    mDim = 1;
    for (int i=1; i<data->buffer().dimensions; i++) {
        mDim *= data->length(i);
    }
    return NO_ERROR;
}


class CPUSegmentMeanCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs,
                        const MNN::Op* op, Backend* backend) const override {
        return new CPUSegmentMean(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSegmentMeanCreator, OpType_Segment);




}//namecpace MNN
