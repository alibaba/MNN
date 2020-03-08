#include "OpCommonUtils.hpp"
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
}
