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
//#define MNN_DEBUG_TENSOR_SIZE
namespace MNN {
void registerShapeOps();
SizeComputerSuite* SizeComputerSuite::gInstance = nullptr;

SizeComputerSuite::~SizeComputerSuite() {
    for (auto& iter : mRegistry) {
        delete iter.second;
    }
}

void SizeComputerSuite::init() {
    if (nullptr != gInstance) {
        return;
    }
    gInstance = new SizeComputerSuite;
    registerShapeOps();
}

SizeComputerSuite* SizeComputerSuite::get() {
    return gInstance;
}

void SizeComputerSuite::insert(SizeComputer* t, OpType type) {
    mRegistry.insert(std::make_pair(type, t));
}

SizeComputer* SizeComputerSuite::search(OpType name) {
    auto iter = mRegistry.find(name);
    if (iter == mRegistry.end()) {
        return nullptr;
    }
    return iter->second;
}
float SizeComputer::onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) const {
    MNN_ASSERT(outputs.size() >= 1);
    return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
}
bool SizeComputer::opNeedContent(OpType type, int index) {
    switch (type) {
        case OpType_ZerosLike:
        case OpType_ZeroGrad:
        case OpType_Shape:
        case OpType_Rank:
        case OpType_Const:
        case OpType_Size:
        case OpType_PriorBox:
            return false;
        case OpType_Interp:
        case OpType_Crop:
        case OpType_Reshape:
        case OpType_Reduction:
        case OpType_Resize:
            if (1 == index) {
                return false;
            }
            break;
        default:
            break;
    }
    return true;
}
float SizeComputer::computeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    auto computer       = computeFactory->search(op->type());
    if (nullptr != computer) {
        return computer->onComputeFlops(op, inputs, outputs);
    }
    auto sumFlops = 0.0f;
    for (auto output : outputs) {
        sumFlops += (float)output->elementSize() / 1024.0f / 1024.0f;
    }
    return sumFlops;
}

bool SizeComputer::computeOutputSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        // Don't support compute shape for control flow op
        if (op->type() == OpType_While || op->type() == OpType_If) {
            return false;
        }
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            bool ret = computer->onComputeSize(op, inputs, outputs);
#ifdef MNN_DEBUG_TENSOR_SIZE
            if (op->name() != nullptr) {
                MNN_PRINT("\t===> compute shape: %s, [%s]\n", op->name()->c_str(), MNN::EnumNameOpType(op->type()));
            } else {
                MNN_PRINT("\t===> compute shape:[%s]\n", MNN::EnumNameOpType(op->type()));
            }
            if (inputs.size()) {
                MNN_PRINT("Inputs:\n");
                for (auto o : inputs) {
                    if (o->dimensions() == 0) {
                        MNN_PRINT("\t*Scalar*");
                    }
                    for (int i = 0; i < o->dimensions(); ++i) {
                        MNN_PRINT("%d, ", o->length(i));
                    }
                    MNN_PRINT("\n");
                }
            }
            MNN_PRINT("Outputs:\n");
            for (auto o : outputs) {
                if (o->dimensions() == 0) {
                    MNN_PRINT("\t*Scalar*");
                }
                for (int i = 0; i < o->dimensions(); ++i) {
                    MNN_PRINT("%d, ", o->length(i));
                }
                MNN_PRINT("\n");
            }
#endif
            return ret;
        }
    }

    // Default Set to the same
    if (inputs.size() >= 1 && outputs.size() == 1) {
        if (inputs[0] == outputs[0]) {
            return true;
        }
        const auto& ib = inputs[0]->buffer();
        auto& ob       = outputs[0]->buffer();
        memcpy(ob.dim, ib.dim, sizeof(halide_dimension_t) * ib.dimensions);
        ob.dimensions                                         = ib.dimensions;
        ob.type                                               = ib.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
    // Not Support
    MNN_PRINT("Can't compute size for %d, name=%s\n", op->type(), op->name() ? op->name()->c_str() : "");

    return false;
}

std::vector<int> SizeComputer::needInputContent(const MNN::Op* op) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            return computer->mNeedContentInputIndex;
        }
    }
    return std::vector<int>{};
}

} // namespace MNN
