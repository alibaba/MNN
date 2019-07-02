//
//  ShapeTranspose.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"
namespace MNN {

class TransposeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto OpParam        = op->main_as_Transpose();
        const Tensor* input = inputs[0];
        Tensor* perm        = inputs[1];
        std::shared_ptr<Tensor> perTemp;

        // copy data from device to host if needed
        if (!perm->host<int32_t>() && perm->deviceId()) {
            perTemp.reset(Tensor::createHostTensorFromDevice(perm, true));
            perm = perTemp.get();
        }

        const int dims = input->buffer().dimensions;
        MNN_ASSERT(dims == perm->buffer().dim[0].extent);

        std::vector<int32_t> permutation;
        if (OpParam->Tperm() == DataType_DT_INT32) {
            for (int i = 0; i < perm->buffer().dim[0].extent; i++) {
                permutation.push_back(perm->host<int32_t>()[i]);
            }
        } else if (OpParam->Tperm() == DataType_DT_INT64) {
            for (int i = 0; i < perm->buffer().dim[0].extent; i++) {
                permutation.push_back(static_cast<int32_t>(perm->host<int64_t>()[i]));
            }
        } else {
            MNN_ASSERT(false);
        }

        outputs[0]->buffer().dimensions = dims;

        for (int i = 0; i < dims; ++i) {
            const int32_t d                    = permutation[i];
            outputs[0]->buffer().dim[i].extent = input->buffer().dim[d].extent;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(TransposeComputer, OpType_Transpose);
} // namespace MNN
