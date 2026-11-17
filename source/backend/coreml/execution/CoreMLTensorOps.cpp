#include "CoreMLCommonExecution.hpp"
#include "core/TensorUtils.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"

namespace MNN {
namespace {
bool constantIntScalar(const Tensor* tensor) {
    return tensor && tensor->getType() == halide_type_of<int>() && TensorUtils::getRawSize(tensor) == 1 &&
           TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::CONSTANT && tensor->host<int>();
}
} // namespace

class CoreMLCumSum : public CoreMLCommonExecution {
public:
    CoreMLCumSum(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        mValid = inputs.size() == 2 && outputs.size() == 1 && inputs[0]->dimensions() > 0 && op->main_as_CumSum() &&
                 constantIntScalar(inputs[1]);
        if (mValid) {
            int axis = inputs[1]->host<int>()[0];
            mValid = axis >= -inputs[0]->dimensions() && axis < inputs[0]->dimensions();
        }
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        int axis = inputs[1]->host<int>()[0];
        if (axis < 0) {
            axis += inputs[0]->dimensions();
        }
        auto param = mOp->main_as_CumSum();
        operation("cumsum")
            .input("x", inputs[0])
            .scalar("axis", axis)
            .scalar("exclusive", param->exclusive())
            .scalar("reverse", param->reverse())
            .output(outputs);
        return NO_ERROR;
    }
};

class CoreMLGatherND : public CoreMLCommonExecution {
public:
    CoreMLGatherND(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        int batchDims = op->main_as_Axis() ? op->main_as_Axis()->axis() : 0;
        mValid = inputs.size() == 2 && outputs.size() == 1 && batchDims == 0 && inputs[0]->dimensions() > 0 &&
                 inputs[1]->dimensions() > 0 && inputs[1]->getType() == halide_type_of<int>() &&
                 inputs[1]->length(inputs[1]->dimensions() - 1) > 0 &&
                 inputs[1]->length(inputs[1]->dimensions() - 1) <= inputs[0]->dimensions();
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        operation("gather_nd")
            .input("x", inputs[0])
            .input("indices", inputs[1])
            .scalar<int32_t>("batch_dims", 0)
            .scalar("validate_indices", true)
            .output(outputs);
        return NO_ERROR;
    }
};

class CoreMLGatherElements : public CoreMLCommonExecution {
public:
    CoreMLGatherElements(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        mValid = (inputs.size() == 2 || inputs.size() == 3) && outputs.size() == 1 && inputs[0]->dimensions() > 0 &&
                 inputs[0]->dimensions() == inputs[1]->dimensions() && inputs[1]->getType() == halide_type_of<int>() &&
                 (inputs.size() == 2 || constantIntScalar(inputs[2]));
        if (!mValid) {
            return;
        }
        int axis = inputs.size() == 3 ? inputs[2]->host<int>()[0] : 0;
        if (axis < 0) {
            axis += inputs[0]->dimensions();
        }
        mValid = axis >= 0 && axis < inputs[0]->dimensions();
        for (int i = 0; mValid && i < inputs[0]->dimensions(); ++i) {
            mValid = i == axis || inputs[0]->length(i) == inputs[1]->length(i);
        }
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        int axis = inputs.size() == 3 ? inputs[2]->host<int>()[0] : 0;
        if (axis < 0) {
            axis += inputs[0]->dimensions();
        }
        operation("gather_along_axis")
            .input("x", inputs[0])
            .input("indices", inputs[1])
            .scalar("axis", axis)
            .scalar("validate_indices", true)
            .output(outputs);
        return NO_ERROR;
    }
};

class CoreMLReverse : public CoreMLCommonExecution {
public:
    CoreMLReverse(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        mValid = inputs.size() == 2 && outputs.size() == 1 && constantIntScalar(inputs[1]);
        if (mValid) {
            int axis = inputs[1]->host<int>()[0];
            mValid = axis >= 0 && axis < inputs[0]->dimensions();
        }
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        operation("reverse").input("x", inputs[0]).array<int>("axes", {inputs[1]->host<int>()[0]}).output(outputs);
        return NO_ERROR;
    }
};

class CoreMLOneHot : public CoreMLCommonExecution {
public:
    CoreMLOneHot(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        mValid = inputs.size() == 4 && outputs.size() == 1 && op->main_as_OneHotParam() &&
                 inputs[0]->getType() == halide_type_of<int>() && constantIntScalar(inputs[1]);
        if (!mValid) {
            return;
        }
        for (int i : {2, 3}) {
            mValid = mValid && TensorUtils::getRawSize(inputs[i]) == 1 && inputs[i]->host<void>() &&
                     TensorUtils::getDescribe(inputs[i])->usage == Tensor::InsideDescribe::CONSTANT;
        }
        int axis = op->main_as_OneHotParam()->axis();
        int rank = inputs[0]->dimensions() + 1;
        mValid = mValid && inputs[1]->host<int>()[0] > 0 && axis >= -rank && axis < rank &&
                 inputs[2]->getType() == inputs[3]->getType() && inputs[2]->getType() == outputs[0]->getType();
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        int axis = mOp->main_as_OneHotParam()->axis();
        if (axis < 0) {
            axis += inputs[0]->dimensions() + 1;
        }
        auto scalar = [&](int index, const char* name) {
            if (inputs[index]->getType() == halide_type_of<int>()) {
                return buildScalar(inputs[index]->host<int>()[0], name);
            }
            float value = 0.0f;
            auto owner = TensorUtils::getDescribeOrigin(inputs[index])->getBackend();
            if (owner && owner->type() == MNN_FORWARD_NN) {
                MNNDequantizeFP16(inputs[index]->host<int16_t>(), &value, 1);
            } else {
                value = inputs[index]->host<float>()[0];
            }
            return buildScalar(value, name);
        };
        // iOS BNNS rejects out-of-range one_hot indices; MNN requires off_value for them.
        auto indexShape = inputs[0]->shape();
        indexShape.insert(indexShape.begin() + axis, 1);
        auto indices = operation("reshape")
                           .input("x", inputs[0])
                           .shape("shape", indexShape)
                           .output(indexShape, "indices", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32);
        int depth = inputs[1]->host<int>()[0];
        std::vector<int> classes(depth);
        for (int i = 0; i < depth; ++i) {
            classes[i] = i;
        }
        std::vector<int> classShape(indexShape.size(), 1);
        classShape[axis] = depth;
        auto values = operation("reshape")
                          .array("x", classes, "classes")
                          .shape("shape", classShape, "class_shape")
                          .output(classShape, "class_values", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32);
        auto matches = operation("equal")
                           .input("x", indices)
                           .input("y", values)
                           .output(outputs[0]->shape(), "matches", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
        operation("select")
            .input("cond", matches)
            .input("a", scalar(2, "on"))
            .input("b", scalar(3, "off"))
            .output(outputs);
        return NO_ERROR;
    }
};

class CoreMLMatrixBandPart : public CoreMLCommonExecution {
public:
    CoreMLMatrixBandPart(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        mValid = inputs.size() == 3 && outputs.size() == 1 && inputs[0]->dimensions() >= 2 &&
                 inputs[0]->getType() == halide_type_of<float>() && constantIntScalar(inputs[1]) &&
                 constantIntScalar(inputs[2]);
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        operation("band_part")
            .input("x", inputs[0])
            .scalar("lower", inputs[1]->host<int>()[0])
            .scalar("upper", inputs[2]->host<int>()[0])
            .output(outputs);
        return NO_ERROR;
    }
};

class CoreMLGridSample : public CoreMLCommonExecution {
public:
    CoreMLGridSample(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        auto param = op->main_as_GridSample();
        mValid = inputs.size() == 2 && outputs.size() == 1 && param && !param->backward() &&
                 inputs[0]->dimensions() == 4 && inputs[1]->dimensions() == 4 && inputs[1]->length(3) == 2 &&
                 inputs[0]->length(0) == inputs[1]->length(0) && inputs[0]->getType() == halide_type_of<float>() &&
                 inputs[1]->getType() == halide_type_of<float>() && param->mode() == SampleMode_BILINEAR &&
                 TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NHWC &&
                 (param->paddingMode() == BorderMode_ZEROS || param->paddingMode() == BorderMode_CLAMP);
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto param = mOp->main_as_GridSample();
        std::string padding = param->paddingMode() == BorderMode_ZEROS ? "constant" : "border";
        operation("resample")
            .input("x", inputs[0])
            .input("coordinates", inputs[1])
            .scalar("sampling_mode", "bilinear", "sampling")
            .scalar("padding_mode", padding, "padding")
            .scalar("padding_value", 0.0f)
            .scalar("coordinates_mode", "normalized_minus_one_to_one")
            .scalar("align_corners", param->alignCorners())
            .output(outputs);
        return NO_ERROR;
    }
};

class CoreMLLRN : public CoreMLCommonExecution {
public:
    CoreMLLRN(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
        : CoreMLCommonExecution(backend, op) {
        auto param = op->main_as_LRN();
        mValid = inputs.size() == 1 && outputs.size() == 1 && param && param->regionType() == 0 &&
                 param->localSize() > 0 && param->localSize() % 2 == 1 && inputs[0]->dimensions() == 4 &&
                 inputs[0]->getType() == halide_type_of<float>() &&
                 TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NHWC;
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto param = mOp->main_as_LRN();
        operation("local_response_norm")
            .input("x", inputs[0])
            .scalar("size", param->localSize())
            .scalar("alpha", param->alpha())
            .scalar("beta", param->beta())
            .scalar("k", param->bias(), "bias")
            .output(outputs);
        return NO_ERROR;
    }
};

REGISTER_COREML_OP_CREATOR(CoreMLLRN, OpType_LRN);
REGISTER_COREML_OP_CREATOR(CoreMLGridSample, OpType_GridSample);
REGISTER_COREML_OP_CREATOR(CoreMLMatrixBandPart, OpType_MatrixBandPart);
REGISTER_COREML_OP_CREATOR(CoreMLReverse, OpType_Reverse);
REGISTER_COREML_OP_CREATOR(CoreMLOneHot, OpType_OneHot);
REGISTER_COREML_OP_CREATOR(CoreMLCumSum, OpType_CumSum);
REGISTER_COREML_OP_CREATOR(CoreMLGatherND, OpType_GatherND);
REGISTER_COREML_OP_CREATOR(CoreMLGatherElements, OpType_GatherElements);
} // namespace MNN
