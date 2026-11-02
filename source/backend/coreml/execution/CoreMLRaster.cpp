//
//  CoreMLRaster.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLRaster.hpp"
#include <cmath>
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"

namespace MNN {

CoreMLRaster::CoreMLRaster(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    if (op->type() == OpType_Padding && inputs.size() > 2) {
        mValid = TensorUtils::getDescribe(inputs[2])->usage == Tensor::InsideDescribe::CONSTANT &&
                 TensorUtils::getRawSize(inputs[2]) == 1 && inputs[2]->host<void>();
    }
    if (op->type() == OpType_StridedSlice && op->main_as_StridedSliceParam()->fromType() == 0 && inputs.size() == 5) {
        mValid = false;
    }
}

inline std::vector<int> tovec(const flatbuffers::Vector<int>* dims) {
    return std::vector<int>(dims->data(), dims->data() + dims->size());
}

ErrorCode CoreMLRaster::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (mOp->type() == OpType_Raster) {
        // Raster addresses dense storage linearly. Keeping its public MIL bindings
        // flat also avoids ANE compiler reshape expansion for rank-five tensors
        // (for example [2,2,2,2,2] -> [32] -> [2,2,2,2,2]).
        mCoreMLBackend->useFlatIO();
        OpCommonUtils::rasterInputReset(inputs, outputs[0]);
        auto regions = TensorUtils::getDescribe(outputs[0])->regions;
        const int total = TensorUtils::getRawSize(outputs[0]);
        const auto dtype = mCoreMLBackend->typeConvert(outputs[0]->getType());
        int serial = 0;
        auto reshape = [&](const std::string& x, const std::vector<int>& shape) {
            auto tag = std::to_string(serial++);
            return operation("reshape").input("x", x).shape("shape", shape, "shape_" + tag).output(shape, tag, dtype);
        };
        auto gather = [&](const std::string& x, const std::vector<int>& indices) {
            auto tag = std::to_string(serial++);
            return operation("gather")
                .input("x", x)
                .array("indices", indices, "indices_" + tag)
                .scalar<int32_t>("axis", 0, "axis_" + tag)
                .scalar("validate_indices", true, "validate_" + tag)
                .output({int(indices.size())}, tag, dtype);
        };
        auto contiguous = [](const Tensor::InsideDescribe::Region& region, bool source) {
            const auto& view = source ? region.src : region.dst;
            int stride = 1;
            for (int i = 2; i >= 0; --i) {
                if (region.size[i] > 1 && view.stride[i] != stride) {
                    return false;
                }
                stride *= region.size[i];
            }
            return true;
        };
        bool concatenation = true;
        int covered = 0;
        for (const auto& region : regions) {
            if (!contiguous(region, false) || region.dst.offset != covered) {
                concatenation = false;
            }
            covered += region.size[0] * region.size[1] * region.size[2];
        }
        concatenation = concatenation && covered == total;
        std::vector<std::string> pieces;
        std::string output;
        if (!concatenation) {
            auto fill = operation("fill");
            fill.shape("shape", {total}, "zero_shape");
            fill.input("value", outputs[0]->getType() == halide_type_of<int>() ? buildScalar(0, "zero")
                                                                               : buildScalar(0.0f, "zero"));
            output = fill.output({total}, "zero", dtype);
        }
        for (const auto& region : regions) {
            if (!region.origin) {
                return NOT_SUPPORT;
            }
            int count = region.size[0] * region.size[1] * region.size[2];
            int sourceSize = TensorUtils::getRawSize(region.origin);
            auto value = mCoreMLBackend->getTensorName(region.origin);
            auto flat = value;
            std::vector<int> order = {0, 1, 2};
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return region.src.stride[a] > region.src.stride[b]; });
            int stride = 1;
            bool permutation = region.src.offset == 0 && count == sourceSize;
            for (int i = 2; i >= 0; --i) {
                int axis = order[i];
                if (region.size[axis] > 1 && region.src.stride[axis] != stride) {
                    permutation = false;
                }
                stride *= region.size[axis];
            }
            if (contiguous(region, true)) {
                if (region.src.offset < 0 || count < 0 ||
                    static_cast<int64_t>(region.src.offset) + count > sourceSize) {
                    return NOT_SUPPORT;
                }
                if (region.src.offset == 0 && count == sourceSize) {
                    value = flat;
                } else {
                    auto tag = std::to_string(serial++);
                    value = operation("slice_by_size")
                                .input("x", flat)
                                .array<int>("begin", {region.src.offset}, "begin_" + tag)
                                .array<int>("size", {count}, "size_" + tag)
                                .output({count}, tag, dtype);
                }
            } else if (permutation) {
                std::vector<int> shape, perm(3);
                for (int i = 0; i < 3; ++i) {
                    shape.push_back(region.size[order[i]]);
                    perm[order[i]] = i;
                }
                value = reshape(flat, shape);
                auto tag = std::to_string(serial++);
                auto transpose = operation("transpose");
                transpose.input("x", value).array("perm", perm, "perm_" + tag);
                value = transpose.output({region.size[0], region.size[1], region.size[2]}, tag, dtype);
                value = reshape(value, {count});
            } else {
                std::vector<int> indices;
                indices.reserve(count);
                for (int z = 0; z < region.size[0]; ++z) {
                    for (int y = 0; y < region.size[1]; ++y) {
                        for (int x = 0; x < region.size[2]; ++x) {
                            int index = region.src.offset + z * region.src.stride[0] + y * region.src.stride[1] +
                                        x * region.src.stride[2];
                            if (index < 0 || index >= sourceSize) {
                                return NOT_SUPPORT;
                            }
                            indices.push_back(index);
                        }
                    }
                }
                value = gather(flat, indices);
            }
            if (concatenation) {
                pieces.push_back(value);
            } else {
                std::vector<int> indices;
                indices.reserve(count);
                for (int z = 0; z < region.size[0]; ++z) {
                    for (int y = 0; y < region.size[1]; ++y) {
                        for (int x = 0; x < region.size[2]; ++x) {
                            int index = region.dst.offset + z * region.dst.stride[0] + y * region.dst.stride[1] +
                                        x * region.dst.stride[2];
                            if (index < 0 || index >= total) {
                                return NOT_SUPPORT;
                            }
                            indices.push_back(index);
                        }
                    }
                }
                auto tag = std::to_string(serial++);
                output = operation("scatter")
                             .input("data", output)
                             .input("updates", value)
                             .array("indices", indices, "dst_" + tag)
                             .scalar("axis", 0, "scatter_axis_" + tag)
                             .scalar("mode", "update", "scatter_mode_" + tag)
                             .scalar("validate_indices", true, "scatter_validate_" + tag)
                             .output({total}, tag, dtype);
            }
        }
        if (concatenation) {
            if (pieces.size() == 1) {
                output = pieces[0];
            } else {
                output = operation("concat")
                             .inputs("values", pieces)
                             .scalar("axis", 0, "concat_axis")
                             .scalar("interleave", false)
                             .output({total}, "joined", dtype);
            }
        }
        operation("reshape").input("x", output).shape("shape", {total}, "output_shape").output(outputs);
        return NO_ERROR;
    }
    auto input = inputs[0];
    auto inputName = mCoreMLBackend->getTensorName(input);
    switch (mOp->type()) {
        case OpType_Select: {
            auto cond = operation("cast")
                            .input("x", inputName)
                            .scalar("dtype", "bool", "condition_dtype")
                            .output(inputs[0]->shape(), "condition", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
            operation("select").input("cond", cond).input("a", inputs[1]).input("b", inputs[2]).output(outputs);
            break;
        }
        case OpType_ConvertTensor: {
            bool fromNHWC = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC;
            bool toNHWC = TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC;
            int rank = input->dimensions();
            if (fromNHWC == toNHWC || rank <= 2) {
                operation("identity").input("x", inputName).output(outputs);
                return NO_ERROR;
            }
            std::vector<int> perm(rank);
            perm[0] = 0;
            if (toNHWC) {
                for (int i = 1; i < rank - 1; ++i) {
                    perm[i] = i + 1;
                }
                perm.back() = 1;
            } else {
                perm[1] = rank - 1;
                for (int i = 2; i < rank; ++i) {
                    perm[i] = i - 1;
                }
            }
            operation("transpose").input("x", inputName).array<int>("perm", perm).output(outputs);
            break;
        }
        case OpType_Reshape: {
            // auto reshape = mOp->main_as_Reshape();
            // auto dims = toVec(reshape->dims());
            operation("reshape").input("x", inputName).shape("shape", outputs[0]->shape()).output(outputs);
            break;
        }
        case OpType_Unsqueeze:
        case OpType_Squeeze: {
            auto builder = operation(mOp->type() == OpType_Unsqueeze ? "expand_dims" : "squeeze");
            builder.input("x", inputName);
            if (inputs.size() == 2) {
                builder.input("axes", inputs[1]);
            } else {
                auto unsqueeze = mOp->main_as_SqueezeParam();
                if (unsqueeze->squeezeDims() != nullptr) {
                    auto axes = tovec(unsqueeze->squeezeDims());
                    builder.array<int>("axes", axes);
                }
            }
            builder.output(outputs);
            break;
        }
        case OpType_Permute: {
            auto permute = mOp->main_as_Permute();
            auto dims = tovec(permute->dims());
            operation("transpose").input("x", inputName).array<int>("perm", dims).output(outputs);
            break;
        }
        case OpType_Concat: {
            auto axis = mOp->main_as_Axis()->axis();
            operation("concat")
                .inputs("values", inputs)
                .scalar("axis", axis)
                .scalar<bool>("interleave", false)
                .output(outputs);
            break;
        }
        case OpType_GatherV2: {
            auto builder = operation("gather");
            builder.input("x", inputName).input("indices", inputs[1]);
            int axis = 0;
            if (inputs.size() == 3) {
                if (inputs[2]->getType() != halide_type_of<int>() || !inputs[2]->host<int>() ||
                    TensorUtils::getRawSize(inputs[2]) != 1) {
                    return NOT_SUPPORT;
                }
                axis = inputs[2]->host<int32_t>()[0];
            } else if (mOp->main_type() == OpParameter_Axis) {
                axis = mOp->main_as_Axis()->axis();
            }
            builder.scalar("axis", axis).scalar<bool>("validate_indices", true).output(outputs);
            break;
        }
        case OpType_Slice: {
            auto slice = mOp->main_as_Slice();
            auto axis = slice->axis();
            if (axis < 0) {
                axis = axis + input->dimensions();
            }
            std::vector<int> splitSizes(outputs.size());
            for (int i = 0; i < outputs.size(); ++i) {
                auto output = outputs[i];
                splitSizes[i] = output->length(axis);
            }
            bool isNumSplits = true;
            for (int i = 1; i < splitSizes.size(); i++) {
                isNumSplits &= (splitSizes[i] == splitSizes[i - 1]);
            }
            auto builder = operation("split");
            builder.input("x", inputName).scalar("axis", axis);
            if (isNumSplits) {
                builder.scalar<int>("num_splits", static_cast<int>(splitSizes.size()));
            } else {
                builder.array<int>("split_sizes", splitSizes);
            }
            builder.output(outputs);
            break;
        }
        case OpType_StridedSlice: {
            const auto param = mOp->main_as_StridedSliceParam();
            const int rank = input->dimensions();
            if (inputs.size() < 3) {
                return NOT_SUPPORT;
            }
            const int count = TensorUtils::getRawSize(inputs[1]);
            for (size_t i = 1; i < inputs.size(); ++i) {
                if (inputs[i]->getType() != halide_type_of<int>() || !inputs[i]->host<int>() ||
                    TensorUtils::getRawSize(inputs[i]) != count) {
                    return NOT_SUPPORT;
                }
            }
            std::vector<int> begin(rank, 0), end = input->shape(), stride(rank, 1), slicedShape(rank);
            std::vector<bool> beginMask(rank, false), endMask(rank, false);
            int axis = 0;
            for (int i = 0; i < count; ++i) {
                if (param->fromType() == 0 && (param->newAxisMask() & (1 << i))) {
                    continue;
                }
                if (param->fromType() == 0 && (param->ellipsisMask() & (1 << i))) {
                    int remaining = 0;
                    for (int j = i + 1; j < count; ++j) {
                        if (!(param->newAxisMask() & (1 << j))) {
                            ++remaining;
                        }
                    }
                    axis = rank - remaining;
                    continue;
                }
                int dim = param->fromType() == 1 && inputs.size() >= 4 ? inputs[3]->host<int>()[i] : axis;
                if (dim < 0) {
                    dim += rank;
                }
                if (dim < 0 || dim >= rank) {
                    return NOT_SUPPORT;
                }
                begin[dim] = inputs[1]->host<int>()[i];
                end[dim] = inputs[2]->host<int>()[i];
                int stepInput = param->fromType() == 1 ? 4 : 3;
                if (inputs.size() > stepInput) {
                    stride[dim] = inputs[stepInput]->host<int>()[i];
                }
                if (!stride[dim]) {
                    return NOT_SUPPORT;
                }
                if (param->fromType() == 0) {
                    beginMask[dim] = param->beginMask() & (1 << i);
                    endMask[dim] = param->endMask() & (1 << i);
                    if (param->shrinkAxisMask() & (1 << i)) {
                        int start = begin[dim] < 0 ? begin[dim] + input->length(dim) : begin[dim];
                        begin[dim] = start;
                        end[dim] = start + 1;
                        stride[dim] = 1;
                        beginMask[dim] = endMask[dim] = false;
                    }
                }
                ++axis;
            }
            for (int i = 0; i < rank; ++i) {
                int64_t length = input->length(i), start = begin[i], stop = end[i], step = stride[i];
                if (beginMask[i]) {
                    start = step > 0 ? 0 : length - 1;
                } else if (start < 0) {
                    start += length;
                }
                if (endMask[i]) {
                    stop = step > 0 ? length : -1;
                } else if (stop < 0) {
                    stop += length;
                }
                if (step > 0) {
                    start = std::max<int64_t>(0, std::min(length, start));
                    stop = std::max<int64_t>(0, std::min(length, stop));
                    slicedShape[i] = std::max<int64_t>(0, (stop - start + step - 1) / step);
                } else {
                    start = std::max<int64_t>(-1, std::min(length - 1, start));
                    stop = std::max<int64_t>(-1, std::min(length - 1, stop));
                    slicedShape[i] = std::max<int64_t>(0, (start - stop - step - 1) / -step);
                }
                if (slicedShape[i] <= 0) {
                    return NOT_SUPPORT;
                }
                begin[i] = start;
                end[i] = stop;
                beginMask[i] = false;
                endMask[i] = step < 0 && stop == -1;
            }
            auto builder = operation("slice_by_index");
            builder.input("x", inputName)
                .array("begin", begin)
                .array("end", end)
                .array("stride", stride)
                .array("begin_mask", beginMask)
                .array("end_mask", endMask);
            if (slicedShape != outputs[0]->shape()) {
                auto sliced = builder.output(slicedShape, "sliced", mCoreMLBackend->typeConvert(input->getType()));
                operation("reshape")
                    .input("x", sliced)
                    .shape("shape", outputs[0]->shape(), "slice_shape")
                    .output(outputs);
                return NO_ERROR;
            }
            builder.output(outputs);
            break;
        }
        case OpType_Padding: {
            auto pad = mOp->main_as_PadParam();
            const int rank = input->dimensions();
            if (inputs[1]->getType() != halide_type_of<int>() || !inputs[1]->host<int>() ||
                TensorUtils::getRawSize(inputs[1]) != 2 * rank) {
                return NOT_SUPPORT;
            }
            std::vector<int> pads(inputs[1]->host<int>(), inputs[1]->host<int>() + 2 * rank);
            if (pad->mode() == PadValueMode_SYMMETRIC) {
                auto value = inputName;
                auto shape = input->shape();
                for (int axis = 0; axis < rank; ++axis) {
                    if (pads[2 * axis] == 0 && pads[2 * axis + 1] == 0) {
                        continue;
                    }
                    int length = shape[axis];
                    int size = outputs[0]->length(axis);
                    if (length <= 0 || size <= 0) {
                        return NOT_SUPPORT;
                    }
                    std::vector<int> indices(size);
                    for (int i = 0; i < size; ++i) {
                        int64_t position = (static_cast<int64_t>(i) - pads[2 * axis]) % (2LL * length);
                        if (position < 0) {
                            position += 2LL * length;
                        }
                        indices[i] = position < length ? position : 2LL * length - 1 - position;
                    }
                    shape[axis] = size;
                    auto tag = std::to_string(axis);
                    value = operation("gather")
                                .input("x", value)
                                .array("indices", indices, "indices_" + tag)
                                .scalar("axis", axis, "axis_" + tag)
                                .scalar("validate_indices", true, "valid_" + tag)
                                .output(shape, "symmetric_" + tag, mCoreMLBackend->typeConvert(input->getType()));
                }
                operation("identity").input("x", value).output(outputs);
                break;
            }
            std::string padMode = "constant";
            switch (pad->mode()) {
                case PadValueMode_CONSTANT:
                    break;
                case PadValueMode_REFLECT:
                    padMode = "reflect";
                    break;
                case PadValueMode_EDGE:
                    padMode = "replicate";
                    break;
                default:
                    return NOT_SUPPORT;
            }
            auto builder = operation("pad");
            builder.input("x", inputName);
            float fill = 0.0f;
            if (inputs.size() > 2) {
                auto tensor = inputs[2];
                auto owner = TensorUtils::getDescribeOrigin(tensor)->getBackend();
                if (tensor->getType() == halide_type_of<int>()) {
                    fill = tensor->host<int>()[0];
                } else if (owner && owner->type() == MNN_FORWARD_NN) {
                    MNNDequantizeFP16(tensor->host<int16_t>(), &fill, 1);
                } else {
                    fill = tensor->host<float>()[0];
                }
            }
            builder.array("pad", pads, "pads").scalar("mode", padMode).scalar("constant_val", fill).output(outputs);
            break;
        }
        case OpType_Tile: {
            operation("tile").input("x", inputName).input("reps", inputs[1]).output(outputs);
            break;
        }
        case OpType_Fill: {
            operation("fill_like").input("ref_tensor", inputs[0]).input("value", inputs[1]).output(outputs);
            break;
        }
        case OpType_BroadcastTo: {
            auto oShape = outputs[0]->shape();
            auto iShape = inputs[0]->shape();
            if (iShape.size() > oShape.size()) {
                return NOT_SUPPORT;
            }
            if (iShape.size() < oShape.size()) {
                const bool forward = mOp->main() && mOp->main_as_Axis()->axis();
                iShape.insert(forward ? iShape.end() : iShape.begin(), oShape.size() - iShape.size(), 1);
                inputName = operation("reshape")
                                .input("x", inputName)
                                .shape("shape", iShape)
                                .output(iShape, "broadcast_input", mCoreMLBackend->typeConvert(input->getType()));
            }
            auto builder = operation("tile");
            builder.input("x", inputName);
            std::vector<int> reps(oShape.size(), 1);
            for (auto i = 0; i < oShape.size(); ++i) {
                if (iShape[i] <= 0 || (iShape[i] != 1 && iShape[i] != oShape[i])) {
                    return NOT_SUPPORT;
                }
                reps[i] = oShape[i] / iShape[i];
            }
            builder.array<int>("reps", reps).output(outputs);
            break;
        }
        case OpType_Cast: {
            auto dstType = mOp->main_as_CastParam()->dstT();
            if (dstType == DataType_DT_BOOL) {
                auto compare = operation("not_equal");
                compare.input("x", inputName);
                compare.input("y", input->getType() == halide_type_of<int>() ? buildScalar(0, "zero")
                                                                             : buildScalar(0.0f, "zero"));
                inputName = compare.output(input->shape(), "boolean", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
            }
            operation("cast")
                .input("x", inputName)
                .scalar("dtype", outputs[0]->getType() == halide_type_of<int>() ? "int32" : "fp16")
                .output(outputs);
            break;
        }
        default:
            return NOT_SUPPORT;
    }
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Select);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Raster);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_ConvertTensor);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Reshape);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Unsqueeze);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Squeeze);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Permute);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Concat);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_GatherV2);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Slice);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_StridedSlice);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Padding);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Cast);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Fill);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Tile);
REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_BroadcastTo);

} // namespace MNN
