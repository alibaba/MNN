//
//  SizeComputer.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include <stdlib.h>
#include <mutex>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "utils/InitNet.hpp"
// #define MNN_DEBUG_TENSOR_SIZE
namespace MNN {
void registerShapeOps();
SizeComputerSuite* SizeComputerSuite::gInstance = nullptr;

SizeComputerSuite::~SizeComputerSuite() {
    for (auto& iter : mRegistry) {
        delete iter;
    }
}

void SizeComputerSuite::init() {
    if (nullptr != gInstance) {
        return;
    }
    gInstance = new SizeComputerSuite;
    gInstance->mRegistry.resize(OpType_MAX + 1);
    ::memset(gInstance->mRegistry.data(), 0, gInstance->mRegistry.size() * sizeof(SizeComputer*));
    registerShapeOps();
}

SizeComputerSuite* SizeComputerSuite::get() {
    return gInstance;
}

void SizeComputerSuite::insert(SizeComputer* t, OpType type) {
    mRegistry[type] = t;
}

SizeComputer* SizeComputerSuite::search(OpType name) {
    auto iter = mRegistry[name];
    if (iter == nullptr) {
        return nullptr;
    }
    return iter;
}
float SizeComputer::onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) const {
    MNN_ASSERT(outputs.size() >= 1);
    return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
}

float SizeComputer::computeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    auto computer       = computeFactory->search(op->type());
    if (nullptr != computer) {
        return computer->onComputeFlops(op, inputs, outputs);
    }
    if (op->type() == OpType_While && op->main_type() == OpParameter_LoopParam) {
        auto sumFlops = 0.0f;
        auto loop = op->main_as_LoopParam();
        if (nullptr != loop->commands()) {
            auto cmdSize = loop->commands()->size();
            for (int i=0; i<cmdSize; ++i) {
                auto cmd = loop->commands()->GetAs<RegionCommand>(i);
                auto size = cmd->size()->data();
                sumFlops += (float)size[0] * (float)size[1] * (float)size[2];
            }
        }
        sumFlops *= (float)loop->loopNumber();
        return sumFlops / 1024.0f / 1024.0f;
    }
    auto sumFlops = 0.0f;
    for (auto output : outputs) {
        sumFlops += (float)output->elementSize() / 1024.0f / 1024.0f;
    }
    return sumFlops;
}
#ifdef MNN_DEBUG_TENSOR_SIZE
static void _printShape(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs) {
    if (op->name() != nullptr) {
        MNN_PRINT("===> compute shape: %s, [%s]\n", op->name()->c_str(), MNN::EnumNameOpType(op->type()));
    } else {
        MNN_PRINT("===> compute shape:[%s]\n", MNN::EnumNameOpType(op->type()));
    }
    if (inputs.size()) {
        MNN_PRINT("\tInputs:\n");
        for (auto o : inputs) {
            MNN_PRINT("\tptr=%p, format=%s, datatype=%d;\t", o, EnumNameMNN_DATA_FORMAT(TensorUtils::getDescribe(o)->dimensionFormat), o->getType().code);
            if (o->dimensions() == 0) {
                MNN_PRINT("\t*Scalar*");
            }
            for (int i = 0; i < o->dimensions(); ++i) {
                MNN_PRINT("%d, ", o->length(i));
            }
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("\tOutputs:\n");
    for (auto o : outputs) {
        MNN_PRINT("\tptr=:%p, format=%s, datatype=%d;\t",o, EnumNameMNN_DATA_FORMAT(TensorUtils::getDescribe(o)->dimensionFormat), o->getType().code);
        if (o->dimensions() == 0) {
            MNN_PRINT("\t*Scalar*");
        }
        for (int i = 0; i < o->dimensions(); ++i) {
            MNN_PRINT("%d, ", o->length(i));
        }
        MNN_PRINT("\n");
    }
}
#endif


bool SizeComputer::computeOutputSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        if (op->main_type() == OpParameter_Blob) {
            computeShapeForBlob(op->main_as_Blob(), outputs[0]);
            return true;
        }
        // For Loop Op
        if (op->type() == OpType_While && op->main_type() == OpParameter_LoopParam) {
            auto loop = op->main_as_LoopParam();
            if (loop->extraTensorInfos() == nullptr) {
                return false;
            }
            MNN_ASSERT(loop->extraTensorInfos()->size() == outputs.size());
            for (int i=0; i<outputs.size(); ++i) {
                auto des = loop->extraTensorInfos()->GetAs<TensorDescribe>(i);
                MNN_ASSERT(des->blob() != nullptr);
                auto blob = des->blob();
                TensorUtils::getDescribe(outputs[i])->dimensionFormat = blob->dataFormat();
                outputs[i]->setType(blob->dataType());
                if (blob->dims() != nullptr) {
                    auto dims = blob->dims()->data();
                    outputs[i]->buffer().dimensions = blob->dims()->size();
                    for (int j=0; j<blob->dims()->size(); ++j) {
                        outputs[i]->setLength(j, dims[j]);
                    }
                } else {
                    outputs[i]->buffer().dimensions = 0;
                }
            }
            return true;
        }

        // Don't support compute shape for control flow op
        if (op->type() == OpType_While || op->type() == OpType_If) {
            return false;
        }
        // Check -1 input
        for (auto& t : inputs) {
            for (int i=0; i < t->dimensions(); ++i) {
                if (t->length(i) < 0) {
                    return false;
                }
            }
        }
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            bool ret = computer->onComputeSize(op, inputs, outputs);
#ifdef MNN_DEBUG_TENSOR_SIZE
            _printShape(op, inputs, outputs);
#endif
            return ret;
        }
    }

    // Default Set to the same
    if (inputs.size() >= 1 && (outputs.size() == 1 || outputs.size() == inputs.size())) {
        if (inputs[0] == outputs[0]) {
            return true;
        }
        for (int i=0; i<outputs.size(); ++i) {
            const auto& ib = inputs[i]->buffer();
            auto& ob       = outputs[i]->buffer();
            memcpy(ob.dim, ib.dim, sizeof(halide_dimension_t) * ib.dimensions);
            ob.dimensions                                         = ib.dimensions;
            ob.type                                               = ib.type;
            TensorUtils::getDescribe(outputs[i])->dimensionFormat = TensorUtils::getDescribe(inputs[i])->dimensionFormat;
        }
#ifdef MNN_DEBUG_TENSOR_SIZE
        _printShape(op, inputs, outputs);
#endif
        return true;
    }
    // Not Support
    MNN_PRINT("Can't compute size for %d, name=%s\n", op->type(), op->name() ? op->name()->c_str() : "");

    return false;
}

std::vector<int> SizeComputer::needInputContent(const MNN::Op* op, int inputSize) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        // when hasOutputShape = true, deconv last is outputShape
        if (op->type() == OpType_Deconvolution && op->main_as_Convolution2D() && op->main_as_Convolution2D()->common()) {
            if (op->main_as_Convolution2D()->common()->hasOutputShape()) {
                return std::vector<int>{ inputSize - 1 };
            }
        }
        if (inputSize > 1 && (op->type() == OpType_Squeeze || op->type() == OpType_Unsqueeze)) {
            return std::vector<int>{1};
        }
        if (op->type() == OpType_CumSum) {
            return std::vector<int>{1};
        }
#ifdef MNN_SUPPORT_RENDER
        if (op->type() == OpType_RasterAndInterpolate) {
            int type = 4;
            if (op->main_type() == OpParameter_Extra) {
                auto extra = op->main_as_Extra();
                if (nullptr != extra->attr()) {
                    for (int i=0; i<extra->attr()->size(); ++i) {
                        auto attr = extra->attr()->GetAs<Attribute>(i);
                        if (attr->key()->str() == "primitiveType") {
                            type = attr->i();
                            break;
                        }
                    }
                }
            }
            if (type <= 4) {
                return std::vector<int>{0};
            }
            return std::vector<int>{};
        }
#endif
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            return computer->mNeedContentInputIndex;
        }
    }
    return std::vector<int>{};
}
bool SizeComputer::computeBroadCastDims(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    int maxDimensions = inputs[0]->dimensions();
    int maxIndex = 0;
    for (int index=1; index < inputs.size(); ++index) {
        if (inputs[index]->dimensions() > maxDimensions) {
            maxDimensions = inputs[index]->dimensions();
            maxIndex = index;
        }
    }
    int outputDims[MNN_MAX_TENSOR_DIM];
    for (int i = 0; i < maxDimensions; i++) {
        outputDims[i] = inputs[maxIndex]->length(i);
    }
    for (int index=0; index < inputs.size(); ++index) {
        if (index == maxIndex) {
            continue;
        }
        auto input1 = inputs[index];
        auto input0 = inputs[maxIndex];
        const int diffDimension = maxDimensions - input1->dimensions();
        for (int i = diffDimension; i < maxDimensions; i++) {
            const int input1Index = i - diffDimension;
            int dim1 = input1->buffer().dim[input1Index].extent;
            if (dim1 != outputDims[i] && (dim1 != 1 && outputDims[i] != 1)) {
                MNN_ERROR("Broad cast error, dim1 = %d, dim2 = %d\n", dim1, outputDims[i]);
                return false;
            }
            if (dim1 == outputDims[i]) {
                continue;
            }
            if (dim1 != outputDims[i] && (dim1 == 1 || outputDims[i] == 1)) {
                outputDims[i] = outputDims[i] * dim1;
            } else {
                return false;
            }
        }
    }
    auto& ob       = outputs[0]->buffer();
    ob.dimensions = maxDimensions;
    for (int i = 0; i < maxDimensions; i++) {
        ob.dim[i].extent = outputDims[i];
    }
    return true;
}
} // namespace MNN
