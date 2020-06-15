#include "OpCommonUtils.hpp"
#include "MNN_generated.h"
namespace MNN {
#define MAX_DIM 6
void OpCommonUtils::broastCastComputeDim(int* dims, int* stride, int* iStride0, int* iStride1, const Tensor* input0, const Tensor* input1, const Tensor* output) {
    for (int i = MAX_DIM - 1; i >= 0; --i) {
        dims[i]     = 1;
        stride[i]   = 0;
        iStride0[i] = 0;
        iStride1[i] = 0;
        int input0I = i - (output->dimensions() - input0->dimensions());
        int input1I = i - (output->dimensions() - input1->dimensions());
        if (i < output->dimensions()) {
            dims[i]   = output->length(i);
            stride[i] = output->stride(i);
        }
        if (input0I >= 0 && input0->length(input0I) != 1) {
            iStride0[i] = input0->stride(input0I);
        }
        if (input1I >= 0 && input1->length(input1I) != 1) {
            iStride1[i] = input1->stride(input1I);
        }
    }
}
std::vector<std::tuple<int, int, int>> OpCommonUtils::computeReduceDims(const std::vector<Tensor*>& inputs, const Op* op) {
    // Compute axises
    std::vector<int> axises;
    if (inputs.size() >= 2) {
        auto size = inputs[1]->elementSize();
        auto dims = inputs[1]->host<int32_t>();
        for (int i = 0; i < size; ++i) {
            axises.emplace_back(dims[i]);
        }
    } else {
        auto reduct = op->main_as_ReductionParam();
        if (nullptr != reduct->dim()) {
            for (int i = 0; i < reduct->dim()->size(); ++i) {
                axises.emplace_back(reduct->dim()->data()[i]);
            }
        }
    }
    auto totalSize = inputs[0]->elementSize();
    if (axises.empty()) {
        return {std::make_tuple(1, totalSize, 1)};
    }
    for (int i=0; i<axises.size(); ++i) {
        if (axises[i] < 0) {
            axises[i] = inputs[0]->dimensions() + axises[i];
        }
    }
    // Cache for input's dims
    std::vector<int> lengths(inputs[0]->dimensions());
    for (int i=0; i<lengths.size(); ++i) {
        lengths[i] = inputs[0]->length(i);
    }
    std::vector<std::pair<int, int>> groupAxises;
    {
        //Merge adj axis
        std::sort(axises.begin(), axises.end());
        int lastAxis = axises[0];
        int length = 1;
        int start = axises[0];
        for (int i=1; i<axises.size(); ++i) {
            //MNN_PRINT("%d - %d\n", axises[i], lastAxis);
            if (axises[i] - lastAxis == 1) {
                length++;
            } else {
                groupAxises.emplace_back(std::make_pair(start, length));
                length = 1;
                start = axises[i];
            }
            lastAxis = axises[i];
        }
        groupAxises.emplace_back(std::make_pair(start, length));
    }

    // Compute inside-outside-axis
    std::vector<std::tuple<int, int, int>> result;
    
    for (int i=0; i<groupAxises.size(); ++i) {
        int outsideSize = 1;
        int insideSize = 1;
        int axisSize = 1;
        auto start = groupAxises[i].first;
        auto length = groupAxises[i].second;
        for (int j=0; j<start; ++j) {
            outsideSize *= lengths[j];
        }
        for (int j=start; j<start+length; ++j) {
            axisSize *= lengths[j];
            lengths[j] = 1;
        }
        for (int j=start+length; j<lengths.size(); ++j) {
            insideSize *= lengths[j];
        }
        if (1 == axisSize) {
            continue;
        }
        result.emplace_back(std::make_tuple(outsideSize, axisSize, insideSize));
    }
    //FUNC_PRINT(result.size());
    if (result.empty()) {
        result.emplace_back(std::make_tuple(1, 1, totalSize));
    }
    return result;
}
}
