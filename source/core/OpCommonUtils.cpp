//
//  OpCommonUtils.cpp
//  MNN
//
//  Created by MNN on 2020/03/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpCommonUtils.hpp"
#include "FileLoader.hpp"
#include "MNN_generated.h"
#include "Macro.h"
#include <random>
#include <fstream>

namespace MNN {
Tensor::DimensionType OpCommonUtils::convertDimType(MNN_DATA_FORMAT dimensionFormat) {
    auto dimType = Tensor::CAFFE;
    switch (dimensionFormat) {
        case MNN_DATA_FORMAT_NCHW:
            break;
        case MNN_DATA_FORMAT_NC4HW4:
            dimType = Tensor::CAFFE_C4;
            break;
        case MNN_DATA_FORMAT_NHWC:
            dimType = Tensor::TENSORFLOW;
            break;
        default:
            break;
    }
    return dimType;
}

void OpCommonUtils::loadBlobData(Backend* backend, const Op* op, char* ptr, int size) {
    if (OpParameter_Blob != op->main_type()) {
        return;
    }
    auto b       = op->main_as_Blob();
    if (USE_EXTERNAL_DATA(b)) {
        loadExternalDatas(backend, { ptr }, b->external()->data());
        return;
    }
    void* result = nullptr;
    switch (b->dataType()) {
        case DataType_DT_FLOAT:
            result = (void*)b->float32s()->Data();
            break;
        case DataType_DT_BFLOAT16:
            result = (void*)b->uint8s()->Data();
            break;
        case DataType_DT_INT32:
            result = (void*)b->int32s()->Data();
            break;
        case DataType_DT_QUINT8:
        case DataType_DT_UINT8:
            result = (void*)b->uint8s()->Data();
            break;
        case DataType_DT_INT8:
            result = (void*)b->int8s()->Data();
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    ::memcpy(ptr, result, size);
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
static std::tuple<int, int, int> _computeStride(const std::tuple<int, int, int>& srcTup, const std::tuple<int, int, int>& srcSplit, int step, bool swapnc, int stride) {
    int inside  = std::get<0>(srcTup) / step;
    int axis    = std::get<1>(srcTup) / step;
    int outside = std::get<2>(srcTup) / step;
    auto fuse = _computeAxisFused(srcTup);
    if (std::get<0>(fuse)) {
        // nc fused
        if (swapnc) {
            axis = 0;
            outside = stride / std::get<0>(srcSplit);
        } else {
            outside = 0;
            axis = stride / std::get<0>(srcSplit);
        }
    } else if (std::get<2>(fuse)) {
        // nw fused
        outside = 0;
        inside = stride;
    } else if (std::get<1>(fuse)) {
        // cw fused
        axis = 0;
        inside = stride;
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
            auto tup = _computeStride(srcTup, srcSplits, step, swapnc, region.src.stride[i]);
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
            auto tup = _computeStride(dstTup, dstSplits, step, swapnc, region.dst.stride[i]);
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
#ifdef MNN_DEBUG_BLIT
    MNN_PRINT("Src WCN: %d-%d-%d, Dst WCN:%d-%d-%d\n", std::get<0>(srcSplits), std::get<1>(srcSplits), std::get<2>(srcSplits), std::get<0>(dstSplits), std::get<1>(dstSplits), std::get<2>(dstSplits));
    MNN_PRINT("Origin:%d, %d, %d, %d, src: %d - %d, %d, %d, dst: %d - %d, %d, %d\n", pack,
    region.size[0],region.size[1], region.size[2], region.src
              .offset, region.src.stride[0], region.src.stride[1], region.src.stride[2], region.dst.offset,
              region.dst.stride[0], region.dst .stride[1], region.dst.stride[2]);


    MNN_PRINT("Pack:%d, %d, %d, %d, src: %d - %d, %d, %d, dst: %d - %d, %d, %d\n", pack,
     c4Region.size[0],c4Region.size[1], c4Region.size[2], c4Region.src
               .offset, c4Region.src.stride[0], c4Region.src.stride[1], c4Region.src.stride[2], c4Region.dst.offset,
               c4Region.dst.stride[0], c4Region.dst .stride[1], c4Region.dst.stride[2]);
#endif
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
    auto totalSize = TensorUtils::getRawSize(inputs[0]);
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

bool OpCommonUtils::opNeedContent(const MNN::Op* op, int index) {
    int type = op->type();
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
        case OpType_GridSample:
            if (2 == index) {
                return false;
            }
            break;
#ifdef MNN_SUPPORT_RENDER
        case OpType_RasterAndInterpolate:
        {
            if (0 == index) {
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
                    return false;
                }
            }
            break;
        }
#endif
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
void OpCommonUtils::rasterInputReset(const std::vector<Tensor *> &inputs, Tensor *output) {
    auto outputDes = TensorUtils::getDescribe(output);
    // For output with copy, the region's size may not be the same as input's size
    outputDes->regions.resize(inputs.size());
    for (int i=0; i<outputDes->regions.size(); ++i) {
        outputDes->regions[i].origin = inputs[i];
    }
}

void OpCommonUtils::loadExternalData(Backend* backend, char* addr, int64_t offset, int64_t size) {
    FileLoader fileloader(backend->externalFile().c_str());
    fileloader.offset(offset);
    fileloader.read(addr, size);
}

void OpCommonUtils::loadExternalDatas(Backend* backend, std::vector<char*> addrs,  const int64_t* external) {
    FileLoader fileloader(backend->externalFile().c_str());
    fileloader.offset(external[0]);
    for (int i = 0; i < addrs.size(); i++) {
        fileloader.read(addrs[i], external[i+1]);
    }
}

bool OpCommonUtils::loadConvData(Backend* backend, const Op* op, std::unique_ptr<Tensor>& weight, std::unique_ptr<Tensor>& bias, int& weightSize, int& biasSize) {
    auto conv2d = op->main_as_Convolution2D();
    auto offset = conv2d->external()->Get(0);
    auto weightBytes = conv2d->external()->Get(1);
    auto biasBytes = conv2d->external()->Get(2);
    weightSize = static_cast<int>(weightBytes / sizeof(float));
    biasSize = static_cast<int>(biasBytes / sizeof(float));
    weight.reset(Tensor::createDevice<float>({weightSize}));
    bias.reset(Tensor::createDevice<float>({biasSize}));
    bool res = backend->onAcquire(weight.get(), Backend::STATIC);
    if (!res) {
        return res;
    }
    res = backend->onAcquire(bias.get(), Backend::STATIC);
    if (!res) {
        return res;
    }
    loadExternalDatas(backend, {weight->host<char>(), bias->host<char>()}, conv2d->external()->data());
    return true;
}

static void getBatchChannelArea(const Tensor* t, int& batch, int& channel, int& area) {
    if (t->dimensions() == 0) {
        batch = 1;
        channel = 1;
        area = 1;
        return;
    }
    if (t->dimensions() == 1) {
        batch = t->length(0);
        channel = 1;
        area = 1;
        return;
    }
    batch = t->length(0);
    channel = t->length(1);
    area = 1;
    for (int i=2; i<t->dimensions(); ++i) {
        area *= t->length(i);
    }
}

bool OpCommonUtils::isTranspose(const Tensor::InsideDescribe::Region& region, int& srcOne, int& dstOne) {
    srcOne = -1;
    dstOne = -1;
    for (int i = 0; i < 3; i++) {
        if (region.size[i] == 1) {
            continue;
        }
        if (region.src.stride[i] == 1) {
            if (srcOne >= 0) {
                return false;
            }
            srcOne = i;
        }
        if (region.dst.stride[i] == 1) {
            if (dstOne >= 0) {
                return false;
            }
            dstOne = i;
        }
    }
    return srcOne >= 0 && dstOne >= 0 && srcOne != dstOne;
}

void OpCommonUtils::turnRegion2Convert(const Tensor::InsideDescribe::Region& region, const Tensor* dest, OpCommonUtils::TensorConvertParameter& info) {
    auto origin = region.origin;
    auto srcFormat = TensorUtils::getDescribe(origin)->dimensionFormat;
    auto dstFormat = TensorUtils::getDescribe(dest)->dimensionFormat;
    info.type = 0;
    if (srcFormat == dstFormat) {
        return;
    }
    if (srcFormat != MNN_DATA_FORMAT_NC4HW4 && dstFormat != MNN_DATA_FORMAT_NC4HW4) {
        return;
    }
    const Tensor* nc4hw4Tensor = origin;
    const Tensor* originTensor = dest;
    if (dstFormat == MNN_DATA_FORMAT_NC4HW4) {
        nc4hw4Tensor = dest;
        originTensor = origin;
    }
    getBatchChannelArea(nc4hw4Tensor, info.batch, info.channel, info.area);
    if (0 != region.src.offset || 0 != region.dst.offset) {
        return;
    }
    if (TensorUtils::isCopyRegion(region)) {
        if (info.batch * info.channel * info.area == region.size[0] * region.size[1] * region.size[2]) {
            info.type = 1;
            return;
        }
        return;
    }
    int srcOne, dstOne;
    if (isTranspose(region, srcOne, dstOne)) {
        int keepDim = -1;
        for (int i = 0; i < 3; i++) {
            if (i != srcOne && i != dstOne) {
                keepDim = i;
                break;
            }
        }
        if (info.batch == region.size[keepDim]) {
            if ((info.channel == region.size[srcOne] && info.area == region.size[dstOne]) // NCHW
               || (info.area == region.size[srcOne] && info.channel == region.size[dstOne])) {// NHWC
                auto srcSize = TensorUtils::getRawSize(originTensor);
                auto dstSize = TensorUtils::getRawSize(nc4hw4Tensor);
                auto regionSize = region.size[0] * region.size[1] * region.size[2];
                if (srcSize != dstSize || regionSize != srcSize) {
                    return;
                }
                info.type = 2;
                return;
            }
            return;
        }
    }
    return;
}
bool OpCommonUtils::computeMatMulSize(bool transposeA, bool transposeB, const Tensor* A, const Tensor* B, int& e, int& l, int& h) {
    auto i0Dim = A->dimensions();
    auto i1Dim = B->dimensions();
    if (i0Dim < 1 || i1Dim < 1) {
        return false;
    }
    int w0, h0;
    int w1, h1;
    if (i0Dim == 1) {
        w0 = A->length(0);
        h0 = 1;
        transposeA = false;
    } else {
        w0 = A->length(i0Dim - 1);
        h0 = A->length(i0Dim - 2);
    }
    if (i1Dim == 1) {
        w1 = 1;
        h1 = B->length(0);
        transposeB = false;
    } else {
        w1 = B->length(i1Dim - 1);
        h1 = B->length(i1Dim - 2);
    }
    if (transposeA) {
        auto t = w0;
        w0     = h0;
        h0     = t;
    }
    if (transposeB) {
        auto t = w1;
        w1     = h1;
        h1     = t;
    }

    if (w0 != h1) {
        return false;
    }
    e = h0;
    l = w0;
    h = w1;
    return true;
}


} // namespace MNN
