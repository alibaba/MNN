//
//  OpCommonUtils.cpp
//  MNN
//
//  Created by MNN on 2020/03/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpCommonUtils.hpp"
#include "MNN_generated.h"
#include "Macro.h"
#include <random>

namespace MNN {
void* OpCommonUtils::blobData(const Op* op) {
    if (OpParameter_Blob != op->main_type()) {
        return nullptr;
    }
    auto b       = op->main_as_Blob();
    void* result = nullptr;
    switch (b->dataType()) {
        case DataType_DT_FLOAT:
            result = (void*)b->float32s()->Data();
            break;
        case DataType_DT_INT32:
            result = (void*)b->int32s()->Data();
            break;
        case DataType_DT_QUINT8:
        case DataType_DT_UINT8:
            return (void*)b->uint8s()->Data();
            break;
        case DataType_DT_INT8:
            return (void*)b->int8s()->Data();
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return result;
}

static std::tuple<int, int, int> _split(int offset, int axisL, int area) {
    int inside  = offset % area;
    int temp    = offset / area;
    int axis    = temp % axisL;
    int outside = temp / axisL;
    return std::make_tuple(inside, axis, outside);
}
static std::tuple<bool, bool, bool> _computeAxisFused(const std::tuple<int, int, int>& dstTup) {
    bool ncFused = std::get<1>(dstTup) > 0 && std::get<2>(dstTup) > 0;
    bool nwFused = std::get<0>(dstTup) > 0 && std::get<2>(dstTup) > 0;
    bool cwFused = std::get<1>(dstTup) > 0 && std::get<0>(dstTup) > 0;
    return std::make_tuple(ncFused, cwFused, nwFused);
}
static std::tuple<int, int, int> _computeStride(const std::tuple<int, int, int>& srcTup, const std::tuple<int, int, int>& srcSplit, int step, bool swapnc) {
    int inside  = std::get<0>(srcTup) / step;
    int axis    = std::get<1>(srcTup) / step;
    int outside = std::get<2>(srcTup) / step;
    auto fuse = _computeAxisFused(srcTup);
    if (std::get<0>(fuse)) {
        // nc fused
        if (swapnc) {
            axis = 0;
            if (step >= std::get<1>(srcSplit)) {
                // Cover C, N may be part
                auto outsideStride = (step + 1) / (std::get<1>(srcSplit));
                outside = std::get<2>(srcTup) / (outsideStride - 1);
            } else {
                outside = 1;
            }
        } else {
            outside = 0;
            if (step >= std::get<2>(srcSplit)) {
                // Cover N, C may be a part
                auto axisStride = (step + 1) / (std::get<2>(srcSplit));
                axis = std::get<1>(srcTup) / (axisStride - 1);
            } else {
                axis = 1;
            }
        }
    } else if (std::get<2>(fuse)) {
        // nw fused
        outside = 0;
        if (step >= std::get<2>(srcSplit)) {
            // Cover N, W may be a part
            auto insideStride = (step + 1) / (std::get<2>(srcSplit));
            inside = std::get<0>(srcTup) / (insideStride - 1);
        } else {
            // Cover W
            inside = 1;
        }
    } else if (std::get<1>(fuse)) {
        // cw fused
        axis = 0;
        if (step >= std::get<1>(srcSplit)) {
            auto insideStride = (step + 1) / (std::get<1>(srcSplit));
            inside = std::get<0>(srcTup) / (insideStride - 1);
        } else {
            inside = 1;
        }
    }
    return std::make_tuple(inside, axis, outside);
}
bool OpCommonUtils::canBlitFast(const Tensor::InsideDescribe::Region& region, const SPLITS& srcSplits,
                                const SPLITS& dstSplits, int pack, bool swapnc, bool swapcw) {
    int srcCOffset = (region.src.offset / std::get<0>(srcSplits)) % std::get<1>(srcSplits);
    if (srcCOffset % pack != 0) {
        return false;
    }
    int dstCOffset = (region.dst.offset / std::get<0>(dstSplits)) % std::get<1>(dstSplits);
    if (dstCOffset % pack != 0) {
        return false;
    }
    bool srcAllLengthValid = std::get<0>(srcSplits) > 1 && std::get<1>(srcSplits) > 1 && std::get<2>(srcSplits) > 1;
    bool dstAllLengthValid = std::get<0>(dstSplits) > 1 && std::get<1>(dstSplits) > 1 && std::get<2>(dstSplits) > 1;
    // Check Dst stride
    for (int i = 0; i < 3; ++i) {
        int dstStride  = (region.size[i] - 1) * region.dst.stride[i];
        auto srcStride = region.src.stride[i] * (region.size[i] - 1);
        auto dstTup = _split(dstStride, std::get<1>(dstSplits), std::get<0>(dstSplits));
        auto srcTup = _split(srcStride, std::get<1>(srcSplits), std::get<0>(srcSplits));
        if (std::get<1>(dstTup) != std::get<1>(srcTup)) {
            return false;
        }
        if (srcAllLengthValid) {
            auto srcFused = _computeAxisFused(srcTup);
            if (swapnc) {
                // cw can't be fused, because layout is c, n, w
                if (std::get<1>(srcFused)) {
                    return false;
                }
            } else if (swapcw) {
                // nc can't be fused
                if (std::get<0>(srcFused)) {
                    return false;
                }
            } else {
                // nw can't be fused
                if (std::get<2>(srcFused)) {
                    return false;
                }
            }
        }
        if (dstAllLengthValid) {
            auto dstFused = _computeAxisFused(dstTup);
            if (swapnc) {
                // cw can't be fused, because layout is c, n, w
                if (std::get<1>(dstFused)) {
                    return false;
                }
            } else if (swapcw) {
                // nc can't be fused
                if (std::get<0>(dstFused)) {
                    return false;
                }
            } else {
                // nw can't be fused
                if (std::get<2>(dstFused)) {
                    return false;
                }
            }
        }
    }
    return true;
}
void OpCommonUtils::turnToPackRegion(const Tensor::InsideDescribe::Region& region,
                                     Tensor::InsideDescribe::Region& c4Region, const SPLITS& srcSplits,
                                     const SPLITS& dstSplits, int pack, bool swapnc) {
    int srcAxisC4  = UP_DIV(std::get<1>(srcSplits), pack);
    auto dstAxisC4 = UP_DIV(std::get<1>(dstSplits), pack);
    auto fuseSrc = std::get<0>(srcSplits) * std::get<2>(srcSplits);
    auto fuseDst = std::get<0>(dstSplits) * std::get<2>(dstSplits);

    for (int i = 0; i < 3; ++i) {
        int dstStride = (region.size[i] - 1) * region.dst.stride[i];

        // Get Last Point's inside, axis, outside postion
        auto tup = _split(dstStride, std::get<1>(dstSplits), std::get<0>(dstSplits));
        if (std::get<1>(tup) > 0) {
            // The size has axis offset, divide the axis and mul axisC4 instead
            auto midC4       = UP_DIV(std::get<1>(tup) + 1, pack);
            c4Region.size[i] = region.size[i] / (std::get<1>(tup) + 1) * midC4;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (region.size[i] <= 1) {
            // No need compute stride
            c4Region.src.stride[i] = 0;
            c4Region.dst.stride[i] = 0;
            continue;
        }
        int step = region.size[i] - 1;
        int dstStride  = region.dst.stride[i] * step;
        auto srcStride = region.src.stride[i] * step;
        auto dstTup = _split(dstStride, std::get<1>(dstSplits), std::get<0>(dstSplits));
        auto srcTup = _split(srcStride, std::get<1>(srcSplits), std::get<0>(srcSplits));
        {
            auto tup = _computeStride(srcTup, srcSplits, step, swapnc);
            int inside  = std::get<0>(tup);
            int axis    = std::get<1>(tup);
            int outside = std::get<2>(tup);
            if (swapnc) {
                c4Region.src.stride[i] =
                    outside * std::get<0>(srcSplits) + axis * std::get<0>(srcSplits) * std::get<2>(srcSplits) + inside;
            } else {
                c4Region.src.stride[i] =
                    outside * srcAxisC4 * std::get<0>(srcSplits) + axis * std::get<0>(srcSplits) + inside;
            }
        }
        {
            auto tup = _computeStride(dstTup, dstSplits, step, swapnc);
            int inside  = std::get<0>(tup);
            int axis    = std::get<1>(tup);
            int outside = std::get<2>(tup);
            if (swapnc) {
                c4Region.dst.stride[i] =
                    outside * std::get<0>(dstSplits) + axis * std::get<0>(dstSplits) * std::get<2>(dstSplits) + inside;
            } else {
                c4Region.dst.stride[i] =
                    outside * dstAxisC4 * std::get<0>(dstSplits) + axis * std::get<0>(dstSplits) + inside;
            }
        }
    }
    {
        // Origin offset is compute as NCHW
        auto offsetTup      = _split(region.src.offset, std::get<1>(srcSplits), std::get<0>(srcSplits));
        if (swapnc) {
            // New offset is compute as C4NHW
            c4Region.src.offset = std::get<2>(offsetTup) * pack * std::get<0>(srcSplits)
                + std::get<1>(offsetTup) * std::get<0>(srcSplits) * std::get<2>(srcSplits)
                + std::get<0>(offsetTup) * pack;
        } else {
            c4Region.src.offset = std::get<2>(offsetTup) * srcAxisC4 * pack * std::get<0>(srcSplits) +
                                  std::get<1>(offsetTup) * std::get<0>(srcSplits) + std::get<0>(offsetTup) * pack;
        }
    }
    {
        // Origin offset is compute as NCHW
        auto offsetTup      = _split(region.dst.offset, std::get<1>(dstSplits), std::get<0>(dstSplits));
        if (swapnc) {
            // New offset is compute as C4NHW
            c4Region.dst.offset = std::get<2>(offsetTup) * pack * std::get<0>(dstSplits)
                + std::get<1>(offsetTup) * std::get<0>(dstSplits) * std::get<2>(dstSplits)
                + std::get<0>(offsetTup) * pack;
        } else {
            c4Region.dst.offset = std::get<2>(offsetTup) * dstAxisC4 * pack * std::get<0>(dstSplits) +
                                  std::get<1>(offsetTup) * std::get<0>(dstSplits) + std::get<0>(offsetTup) * pack;
        }
    }
//    MNN_PRINT("Origin:%d, %d, %d, %d, src: %d - %d, %d, %d, dst: %d - %d, %d, %d\n", pack,
//    region.size[0],region.size[1], region.size[2], region.src
//              .offset, region.src.stride[0], region.src.stride[1], region.src.stride[2], region.dst.offset,
//              region.dst.stride[0], region.dst .stride[1], region.dst.stride[2]);
//
//
//    MNN_PRINT("Pack:%d, %d, %d, %d, src: %d - %d, %d, %d, dst: %d - %d, %d, %d\n", pack,
//     c4Region.size[0],c4Region.size[1], c4Region.size[2], c4Region.src
//               .offset, c4Region.src.stride[0], c4Region.src.stride[1], c4Region.src.stride[2], c4Region.dst.offset,
//               c4Region.dst.stride[0], c4Region.dst .stride[1], c4Region.dst.stride[2]);
}

bool OpCommonUtils::canBlitFast(const Tensor::InsideDescribe::Region& region, const Tensor* dest, int pack, bool swapnc, bool swapcw) {
    auto src    = region.origin;
    int srcArea = 1;
    // FIXME: Support dimensions = 1
    if (src->dimensions() == 1 || dest->dimensions() == 1) {
        return false;
    }
    for (int i = 2; i < src->dimensions(); ++i) {
        srcArea *= src->length(i);
    }
    int dstArea = 1;
    for (int i = 2; i < dest->dimensions(); ++i) {
        dstArea *= dest->length(i);
    }
    int inputBatch   = 1;
    int inputChannel = 1;
    if (src->dimensions() > 0) {
        inputBatch = src->length(0);
    }
    if (src->dimensions() > 1) {
        inputChannel = src->length(1);
    }
    int dstBatch   = 1;
    int dstChannel = 1;
    if (dest->dimensions() > 0) {
        dstBatch = dest->length(0);
    }
    if (dest->dimensions() > 1) {
        dstChannel = dest->length(1);
    }
    return canBlitFast(region, std::make_tuple(srcArea, inputChannel, inputBatch),
                       std::make_tuple(dstArea, dstChannel, dstBatch), pack, swapnc, swapcw);
}

void OpCommonUtils::turnToPackRegion(const Tensor::InsideDescribe::Region& region,
                                     Tensor::InsideDescribe::Region& c4Region, const Tensor* dest, int pack, bool swapnc) {
    c4Region    = region;
    auto src    = region.origin;
    int srcArea = 1;
    for (int i = 2; i < src->dimensions(); ++i) {
        srcArea *= src->length(i);
    }
    int dstArea = 1;
    for (int i = 2; i < dest->dimensions(); ++i) {
        dstArea *= dest->length(i);
    }
    int inputBatch   = 1;
    int inputChannel = 1;
    if (src->dimensions() > 0) {
        inputBatch = src->length(0);
    }
    if (src->dimensions() > 1) {
        inputChannel = src->length(1);
    }
    int dstBatch   = 1;
    int dstChannel = 1;
    if (dest->dimensions() > 0) {
        dstBatch = dest->length(0);
    }
    if (dest->dimensions() > 1) {
        dstChannel = dest->length(1);
    }
    turnToPackRegion(region, c4Region, std::make_tuple(srcArea, inputChannel, inputBatch),
                     std::make_tuple(dstArea, dstChannel, dstBatch), pack, swapnc);
}
void OpCommonUtils::broastCastComputeDim(int* dims, int* stride, int* iStride0, int* iStride1, const Tensor* input0,
                                         const Tensor* input1, const Tensor* output) {
    for (int i = MNN_MAX_TENSOR_DIM - 1; i >= 0; --i) {
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
std::vector<std::tuple<int, int, int>> OpCommonUtils::computeReduceDims(const std::vector<Tensor*>& inputs,
                                                                        const Op* op) {
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
    for (int i = 0; i < axises.size(); ++i) {
        if (axises[i] < 0) {
            axises[i] = inputs[0]->dimensions() + axises[i];
            if (axises[i] < 0) {
                return {std::make_tuple(1, totalSize, 1)};
            }
        }
    }
    // Cache for input's dims
    std::vector<int> lengths(inputs[0]->dimensions());
    for (int i = 0; i < lengths.size(); ++i) {
        lengths[i] = inputs[0]->length(i);
    }
    std::vector<std::pair<int, int>> groupAxises;
    {
        // Merge adj axis
        std::sort(axises.begin(), axises.end());
        int lastAxis = axises[0];
        int length   = 1;
        int start    = axises[0];
        for (int i = 1; i < axises.size(); ++i) {
            // MNN_PRINT("%d - %d\n", axises[i], lastAxis);
            if (axises[i] - lastAxis == 1) {
                length++;
            } else {
                groupAxises.emplace_back(std::make_pair(start, length));
                length = 1;
                start  = axises[i];
            }
            lastAxis = axises[i];
        }
        groupAxises.emplace_back(std::make_pair(start, length));
    }

    // Compute inside-outside-axis
    std::vector<std::tuple<int, int, int>> result;

    for (int i = 0; i < groupAxises.size(); ++i) {
        int outsideSize = 1;
        int insideSize  = 1;
        int axisSize    = 1;
        auto start      = groupAxises[i].first;
        auto length     = groupAxises[i].second;
        if (start >= (int)lengths.size()) {
            break;
        }
        for (int j = 0; j < start; ++j) {
            outsideSize *= lengths[j];
        }
        for (int j = start; j < start + length; ++j) {
            if (j >= (int)lengths.size()) {
                break;
            }
            axisSize *= lengths[j];
            lengths[j] = 1;
        }
        for (int j = start + length; j < lengths.size(); ++j) {
            insideSize *= lengths[j];
        }
        if (1 == axisSize) {
            continue;
        }
        result.emplace_back(std::make_tuple(outsideSize, axisSize, insideSize));
    }
    // FUNC_PRINT(result.size());
    if (result.empty()) {
        result.emplace_back(std::make_tuple(1, 1, totalSize));
    }
    return result;
}
void OpCommonUtils::unravelIndexHelper(int32_t* coordinate, const int32_t* mod, int size,
                                       int indice) {
    int value = indice;
    for (int i = 0; i < size; ++i) {
        coordinate[i] = value / mod[i];
        value         = value % mod[i];
    }
}
int OpCommonUtils::computeStride(int32_t* strides, const int* shape, int length) {
    if (length <= 0) {
        return 1;
    }
    int stride = 1;
    for (int i = length - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return stride;
}

bool OpCommonUtils::opNeedContent(int type, int index) {
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
bool OpCommonUtils::opCompabilityForLowp(const Op* op) {
    switch (op->type()) {
        case OpType_Scale:
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
        case OpType_Deconvolution:
        case OpType_DeconvolutionDepthwise:
        case OpType_While:
        case OpType_MatMul:
        case OpType_BatchMatMul:
        case OpType_BinaryOp:
        case OpType_Eltwise:
        case OpType_UnaryOp:
        case OpType_Pooling:
        case OpType_Raster:
        case OpType_ReLU:
        case OpType_ReLU6:
        case OpType_PReLU:
        case OpType_GridSample:
        case OpType_ROIPooling:
        case OpType_ROIAlign:
            return true;
        default:
            break;
    }
    return false;
}

} // namespace MNN
