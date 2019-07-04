//
//  CPUTranspose.cpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUTranspose.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUTranspose<T>::CPUTranspose(Backend* backend, const Op* op) : Execution(backend) {
    auto OpParam = op->main_as_Transpose();
    permDateType = OpParam->Tperm();
}

inline bool NonSingletonDimensionsAlign(const Tensor* input, const std::vector<int32_t>& permutation) {
    int lastNonsingletonPermDim = -1;
    for (int permDim : permutation) {
        if (input->buffer().dim[permDim].extent == 1) {
            continue;
        }
        if (permDim < lastNonsingletonPermDim) {
            return false;
        }
        lastNonsingletonPermDim = permDim;
    }
    return true;
}
template <typename T>
ErrorCode CPUTranspose<T>::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* input = inputs[0];
    const Tensor* perm  = inputs[1];
    auto output         = outputs[0];
    const int dims      = input->buffer().dimensions;
    MNN_ASSERT(dims == perm->buffer().dim[0].extent);

    std::vector<int32_t> permutation;
    for (int i = 0; i < perm->buffer().dim[0].extent; i++) {
        permutation.push_back(perm->host<int32_t>()[i]);
    }

    std::vector<int32_t> shape;
    shape.resize(dims);

    bool isIdentity = true;

    std::vector<bool> bits(dims);
    for (int i = 0; i < dims; ++i) {
        const int32_t d = permutation[i];
        MNN_ASSERT(0 <= d && d < dims);
        bits[d]            = true;
        const auto dimSize = input->buffer().dim[d].extent;
        shape.push_back(dimSize);
        if (d != i) {
            isIdentity = false;
        }
    }

    for (int i = 0; i < dims; ++i) {
        MNN_ASSERT(bits[i]);
    }

    const auto src = input->host<T>();
    T* dst         = output->host<T>();

    if ((dims <= 1 || isIdentity)) {
        memcpy(dst, src, input->size());
        return NO_ERROR;
    } else if (NonSingletonDimensionsAlign(input, permutation)) {
        memcpy(dst, src, input->size());
        return NO_ERROR;
    }

    if (2 == dims) {
        MNN_ASSERT(permutation.size() == 2);
        const int stride0 = input->stride(permutation[0]);
        const int stride1 = input->stride(permutation[1]);
        const int output0 = output->length(0);
        const int output1 = output->length(1);
        for (int i = 0; i < output0; i++) {
            for (int j = 0; j < output1; j++) {
                dst[i * output1 + j] = src[i * stride0 + j * stride1];
            }
        }
    } else if (3 == dims) {
        MNN_ASSERT(permutation.size() == 3);
        const int stride0 = input->stride(permutation[0]);
        const int stride1 = input->stride(permutation[1]);
        const int stride2 = input->stride(permutation[2]);

        const int output0    = output->length(0);
        const int output1    = output->length(1);
        const int output2    = output->length(2);
        const int outStride0 = output->stride(0);
        const int outStride1 = output->stride(1);

        for (int i = 0; i < output0; i++) {
            for (int j = 0; j < output1; j++) {
                for (int k = 0; k < output2; k++) {
                    dst[i * outStride0 + j * outStride1 + k] = src[i * stride0 + j * stride1 + k * stride2];
                }
            }
        }

    } else if (4 == dims) {
        MNN_ASSERT(permutation.size() == 4);

        const int stride0 = input->stride(permutation[0]);
        const int stride1 = input->stride(permutation[1]);
        const int stride2 = input->stride(permutation[2]);
        const int stride3 = input->stride(permutation[3]);

        const int output0    = output->length(0);
        const int output1    = output->length(1);
        const int output2    = output->length(2);
        const int output3    = output->length(3);
        const int outStride0 = output->stride(0);
        const int outStride1 = output->stride(1);
        const int outStride2 = output->stride(2);

        for (int i = 0; i < output0; i++) {
            for (int j = 0; j < output1; j++) {
                for (int k = 0; k < output2; k++) {
                    for (int m = 0; m < output3; m++) {
                        dst[i * outStride0 + j * outStride1 + k * outStride2 + m] =
                            src[i * stride0 + j * stride1 + k * stride2 + m * stride3];
                    }
                }
            }
        }
    } else if (5 == dims) {
        MNN_ASSERT(permutation.size() == 5);
        const int stride0 = input->stride(permutation[0]);
        const int stride1 = input->stride(permutation[1]);
        const int stride2 = input->stride(permutation[2]);
        const int stride3 = input->stride(permutation[3]);
        const int stride4 = input->stride(permutation[4]);

        const int output0    = output->length(0);
        const int output1    = output->length(1);
        const int output2    = output->length(2);
        const int output3    = output->length(3);
        const int output4    = output->length(4);
        const int outStride0 = output->stride(0);
        const int outStride1 = output->stride(1);
        const int outStride2 = output->stride(2);
        const int outStride3 = output->stride(3);

        for (int i = 0; i < output0; i++) {
            for (int j = 0; j < output1; j++) {
                for (int k = 0; k < output2; k++) {
                    for (int m = 0; m < output3; m++) {
                        for (int n = 0; n < output4; n++) {
                            dst[i * outStride0 + j * outStride1 + k * outStride2 + m * outStride3 + n] =
                                src[i * stride0 + j * stride1 + k * stride2 + m * stride3 + n * stride4];
                        }
                    }
                }
            }
        }
    } else {
        MNN_PRINT("Transpose Only Support dimension <= 5!\n");
        MNN_ASSERT(false);
    }

    return NO_ERROR;
}

class CPUTransposeeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUTranspose<float>(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUTransposeeCreator, OpType_Transpose);
} // namespace MNN
