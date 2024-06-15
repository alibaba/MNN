//
//  TensorUtils.cpp
//  MNN
//
//  Created by MNN on 2018/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/TensorUtils.hpp"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "core/Backend.hpp"
#include "core/Macro.h"
namespace MNN {
Tensor::InsideDescribe::NativeInsideDescribe* TensorUtils::getDescribe(const Tensor* tensor) {
    return tensor->mDescribe->mContent.get();
}
bool TensorUtils::regionIsFull(Tensor* input) {
    auto des = TensorUtils::getDescribe(input);
    int size = 1;
    for (int i = 0; i < input->dimensions(); ++i) {
        size *= input->length(i);
    }
    int regionSize = 0;
    for (auto& region : des->regions) {
        regionSize += region.size[1] * region.size[0] * region.size[2];
    }
    return regionSize == size;
}

Tensor::InsideDescribe::Region TensorUtils::makeFullSlice(Tensor* input) {
    Tensor::InsideDescribe::Region totalSlice;
    totalSlice.src.offset = 0;
    totalSlice.dst.offset = 0;
    totalSlice.origin     = input;
    for (int i = 0; i < input->dimensions(); ++i) {
        totalSlice.size[2] *= input->length(i);
    }
    totalSlice.dst.stride[1] = totalSlice.size[2];
    totalSlice.dst.stride[0] = totalSlice.size[2];
    totalSlice.src.stride[1] = totalSlice.size[2];
    totalSlice.src.stride[0] = totalSlice.size[2];
    return totalSlice;
}
bool TensorUtils::reshapeSlice(Tensor::InsideDescribe::Region& slice, int outside, int inside, int axis) {
    if (slice.size[1] == 1 && slice.size[0] == 1 && slice.size[2] == outside * inside * axis) {
        slice.size[0]       = outside;
        slice.size[2]       = inside;
        slice.size[1]       = axis;
        slice.dst.stride[0] = inside * axis;
        slice.dst.stride[1] = inside;

        auto originStride   = slice.src.stride[2];
        slice.src.stride[0] = originStride * inside * axis;
        slice.src.stride[1] = originStride * inside;
        return true;
    }
    if (slice.size[0] == outside && slice.size[1] == axis && slice.size[2] == inside) {
        return true;
    }
    return false;
}

void TensorUtils::setupTensorInfo(const Tensor* tensor, Tensor* wrapTensor, MNN_DATA_FORMAT mMidFormat) {
    TensorUtils::getDescribe(wrapTensor)->dimensionFormat = mMidFormat;
    auto tensorFormat                                     = TensorUtils::getDescribe(tensor)->dimensionFormat;
    bool originCaffeFormat = (tensorFormat == MNN_DATA_FORMAT_NCHW || tensorFormat == MNN_DATA_FORMAT_NC4HW4);
    bool wrapCaffeFormat   = (mMidFormat == MNN_DATA_FORMAT_NCHW || mMidFormat == MNN_DATA_FORMAT_NC4HW4);
    bool originTfFormat    = (tensorFormat == MNN_DATA_FORMAT_NHWC || tensorFormat == MNN_DATA_FORMAT_NHWC4);
    bool wrapTfFormat      = (mMidFormat == MNN_DATA_FORMAT_NHWC || mMidFormat == MNN_DATA_FORMAT_NHWC4);
    if ((originCaffeFormat && wrapCaffeFormat) || (originTfFormat && wrapTfFormat)) {
        TensorUtils::copyShape(tensor, wrapTensor);
    } else if (originCaffeFormat && wrapTfFormat) {
        for (int i = 1; i < wrapTensor->dimensions() - 1; ++i) {
            wrapTensor->setLength(i, tensor->length(i + 1));
        }
        wrapTensor->setLength(0, tensor->length(0));
        wrapTensor->setLength(wrapTensor->dimensions() - 1, tensor->length(1));
    } else if (originTfFormat && wrapCaffeFormat) {
        for (int i = 2; i < wrapTensor->dimensions(); ++i) {
            wrapTensor->setLength(i, tensor->length(i - 1));
        }
        wrapTensor->setLength(0, tensor->length(0));
        wrapTensor->setLength(1, tensor->length(tensor->dimensions() - 1));
    } else {
        // will not reach here
        MNN_ASSERT(false);
    }
    TensorUtils::setLinearLayout(wrapTensor);
    wrapTensor->buffer().type = tensor->getType();
}

void TensorUtils::copyShape(const Tensor* source, Tensor* dest, bool copyFormat, bool copyRef) {
    auto& ob      = dest->buffer();
    auto& ib      = source->buffer();
    ob.dimensions = ib.dimensions;
    ::memcpy(ob.dim, ib.dim, ib.dimensions * sizeof(halide_dimension_t));
    if (copyFormat) {
        getDescribe(dest)->dimensionFormat = getDescribe(source)->dimensionFormat;
    }
    if (copyRef) {
        auto dstDes = getDescribe(dest);
        auto srcDes = getDescribe(source);
        dstDes->regions = srcDes->regions;
        dstDes->quantAttr = srcDes->quantAttr;
        dstDes->type = srcDes->type;
        dest->buffer().type = source->getType();
    }
    adjustTensorForCompability(dest);
}

void TensorUtils::setShape(Tensor* dest, const std::vector<int>& alldims) {
    auto& ob      = dest->buffer();
    ob.dimensions = alldims.size();
    int stride = 1;
    for (int i = alldims.size() - 1; i >= 0; --i) {
        ob.dim[i].stride = stride;
        ob.dim[i].extent = alldims[i];
        stride *= alldims[i];
    }
    return;
}

void TensorUtils::setLinearLayout(Tensor* tensor) {
    auto& buffer = tensor->buffer();
    int size     = 1;
    for (int i = 0; i < buffer.dimensions; ++i) {
        auto index  = buffer.dimensions - i - 1;
        auto extent = buffer.dim[index].extent;
        if (1 == index && tensor->mDescribe->mContent->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            extent = ROUND_UP(extent, 4);
        }
        buffer.dim[index].stride = size;
        size *= extent;
    }
}

static const Tensor* createHostPlanar(const Tensor* source) {
    // check
    auto bnType        = MNN_FORWARD_CPU;
    auto tensorBackend = TensorUtils::getDescribeOrigin(source)->getBackend();
    if (tensorBackend) {
        bnType = tensorBackend->type();
    }
    bool device = bnType != MNN_FORWARD_CPU;
    bool chunky = TensorUtils::getDescribe(source)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;

    // no convert needed
    if (!device && !chunky) {
        return source;
    }

    // convert
    if (chunky) {
        Tensor* result = source->createHostTensorFromDevice(source, false);
        if (result->getDimensionType() == MNN::Tensor::TENSORFLOW) {
            TensorUtils::getDescribe(result)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        } else {
            TensorUtils::getDescribe(result)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        }
        TensorUtils::setLinearLayout(result);

        if (device) {
            void *host = ((Tensor *)source)->map(MNN::Tensor::MAP_TENSOR_READ, result->getDimensionType());
            if(host != nullptr) {
                ::memcpy(result->buffer().host, host, result->size());
            }
            ((Tensor *)source)->unmap(MNN::Tensor::MAP_TENSOR_READ,  result->getDimensionType(), host);
        } else {
            Backend::Info info;
            info.type = MNN_FORWARD_CPU;
            std::shared_ptr<Runtime> runtime(MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU)->onCreate(info));
            auto backend = runtime->onCreate();
            backend->onCopyBuffer(source, result);
            delete backend;
        }
        return result;
    } else {
        return source->createHostTensorFromDevice(source, true);
    }
}

template <typename T>
static void copyTensorToFloat(const Tensor* source, double* dest) {
    auto srcData = source->host<T>();
    auto size    = source->elementSize();
    for (int i = 0; i < size; ++i) {
        dest[i] = srcData[i];
    }
}

static bool equals(const double* pa, const double* pb, size_t size, double tolerance, double epsilon, bool overall,
                   bool prints) {
    // get max if using overall torelance
    double max = fabs(pb[0]);
    if (overall) {
        for (int i = 1; i < size; i++) {
            max = std::max(max, fabs(pb[i]));
        }
    }

    // compare
    for (int i = 0; i < size; i++) {
        float va = pa[i], vb = pb[i];
        if (std::isinf(va) && std::isinf(vb)) {
            continue;
        }
        if (fabs(va) < epsilon && fabs(vb) < epsilon) {
            continue;
        }
        float div = overall ? max : fabsf(vb);
        if (fabsf(va - vb) / div > tolerance) {
            if (prints) {
                MNN_PRINT("%d: %f != %f\n", i, va, vb);
            }
            return false;
        }
    }
    return true;
}

bool TensorUtils::compareTensors(const Tensor* compare, const Tensor* expect, float tolerance, bool overall,
                                 bool printsErrors, bool printsTensors) {
    // type
    if (compare->getType().code != expect->getType().code || compare->getType().bits != expect->getType().bits) {
        if (printsErrors) {
            MNN_PRINT("NOT equal in type: %d/%d - %d/%d.\n", compare->getType().code, compare->getType().bits,
                      expect->getType().code, expect->getType().bits);
        }
        return false;
    }

    // dimensions
    if (compare->dimensions() != expect->dimensions()) {
        if (printsErrors) {
            MNN_PRINT("NOT equal in dimensions: %d - %d.\n", compare->dimensions(), expect->dimensions());
        }
        return false;
    }
    for (int i = 0; i < compare->dimensions(); i++) {
        if (compare->length(i) == expect->length(i)) {
            continue;
        }
        if (printsErrors) {
            MNN_PRINT("NOT equal in dimensions[%d]: %d - %d.\n", i, compare->length(i), expect->length(i));
        }
        return false;
    }

    // convert to host if needed
    auto a = createHostPlanar(compare), b = createHostPlanar(expect);

    // get value as double
    auto size = expect->elementSize();
    std::vector<double> expectValue(expect->elementSize(), 0.0f);
    std::vector<double> compareValue(compare->elementSize(), 0.0f);

    auto result = false;
    if (b->buffer().type.code == halide_type_uint) {
        switch (b->buffer().type.bits) {
            case 8:
                copyTensorToFloat<uint8_t>(a, compareValue.data());
                copyTensorToFloat<uint8_t>(b, expectValue.data());
                break;
            case 16:
                copyTensorToFloat<uint16_t>(a, compareValue.data());
                copyTensorToFloat<uint16_t>(b, expectValue.data());
                break;
            case 32:
                copyTensorToFloat<uint32_t>(a, compareValue.data());
                copyTensorToFloat<uint32_t>(b, expectValue.data());
                break;
            case 64:
                copyTensorToFloat<uint64_t>(a, compareValue.data());
                copyTensorToFloat<uint64_t>(b, expectValue.data());
                break;
            default:
                break;
        }
    } else if (b->buffer().type.code == halide_type_int) {
        switch (b->buffer().type.bits) {
            case 8:
                copyTensorToFloat<int8_t>(a, compareValue.data());
                copyTensorToFloat<int8_t>(b, expectValue.data());
                break;
            case 16:
                copyTensorToFloat<int16_t>(a, compareValue.data());
                copyTensorToFloat<int16_t>(b, expectValue.data());
                break;
            case 32:
                copyTensorToFloat<int32_t>(a, compareValue.data());
                copyTensorToFloat<int32_t>(b, expectValue.data());
                break;
            case 64:
                copyTensorToFloat<int64_t>(a, compareValue.data());
                copyTensorToFloat<int64_t>(b, expectValue.data());
                break;
            default:
                break;
        }
    } else if (b->buffer().type.code == halide_type_float) {
        switch (b->buffer().type.bits) {
            case 32:
                copyTensorToFloat<float>(a, compareValue.data());
                copyTensorToFloat<float>(b, expectValue.data());
                break;
            default:
                break;
        }
    } else {
        if (printsErrors) {
            MNN_PRINT("unsupported data type.");
        }
    }
    auto epsilon = FLT_EPSILON;
    if ((NULL != compareValue.data()) && (NULL != expectValue.data())) {
        result = equals(compareValue.data(), expectValue.data(), size, tolerance, epsilon, overall, printsErrors);
    }

    // clean up
    if (a != compare) {
        delete a;
    }
    if (b != expect) {
        delete b;
    }
    return result;
}

// is copy only region
bool TensorUtils::isCopyRegion(const Tensor::InsideDescribe::Region& region) {
    bool eq = true;
    for (int i = 0; i < 3; i++) {
        eq &= ((region.src.stride[i] == region.dst.stride[i]) || (region.size[i] <= 1));
    }
    return eq;
}

bool TensorUtils::isTransposeRegion(const Tensor::InsideDescribe::Region& region) {
    int srcOne = -1, dstOne = -1;
    for (int i = 0; i < 3; i++) {
        if (region.src.stride[i] == 1 && region.size[i] != 1) {
            if (srcOne >= 0/* || region.size[i] < 4*/) {
                return false;
            }
            srcOne = i;
        }
        if (region.dst.stride[i] == 1 && region.size[i] != 1) {
            if (dstOne >= 0/* || region.size[i] < 4*/) {
                return false;
            }
            dstOne = i;
        }
    }
    return srcOne >= 0 && dstOne >= 0 && srcOne != dstOne;
}

bool TensorUtils::isTileRegion(const Tensor::InsideDescribe::Region& region) {
    bool res = true;
    for (int i = 0; i < 3; i++) {
        if (region.src.stride[i] != 0 && region.size[i] > 1) {
            res &= (region.src.stride[i] == region.dst.stride[i]);
        }
    }
    return res;
}

bool TensorUtils::isDepthToSpaceRegions(const Tensor* output) {
    const auto& regions = TensorUtils::getDescribe(output)->regions;
    if (regions.empty()) {
        return false;
    }
    auto input = regions[0].origin;
    for (const auto region : regions) {
        if (region.origin != input) {
            return false;
        }
    }
    auto ic = input->channel();
    auto ih = input->height();
    auto iw = input->width();
    auto oc = output->channel();
    auto oh = output->height();
    auto ow = output->width();
    if (ic * ih * iw != oc * oh * ow) {
        return false;
    }
    int hblock = oh / ih;
    int wblock = ow / iw;
    if (hblock != wblock) {
        return false;
    }
    if (hblock * wblock * oc != ic) {
        return false;
    }
    if (regions.size() != hblock * wblock) {
        return false;
    }
    return true;
}

// compute offset through region
static inline int offsetCompute(const Tensor::InsideDescribe::Region& reg, int srcOffset, int dstOffset, bool backward) {
    const Tensor::InsideDescribe::View* src;
    const Tensor::InsideDescribe::View* dst;
    if (backward) {
        src = &reg.dst;
        dst = &reg.src;
    } else {
        src = &reg.src;
        dst = &reg.dst;
    }
    int res = 0;
    for (int i = 0; i < 3; i++) {
        if (reg.size[i] > 1) {
            res += (srcOffset / src->stride[i] - dstOffset / src->stride[i]) * dst->stride[i];
            srcOffset %= src->stride[i];
            dstOffset %= src->stride[i];
        }
    }
    return res;
}

// expand src stride with expand value
static inline bool expandSrc(std::vector<int>& src, std::vector<int>& dst, std::vector<int>& size, int expandValue) {
    if (expandValue <= 0) {
        return false;
    }
    for (int i = size.size()-1; i >= 0; i--) {
        int splitSize = expandValue / src[i];
        if (!(expandValue % src[i] || size[i] % splitSize)) {
            src.insert(src.begin()+i, expandValue);
            dst.insert(dst.begin()+i, splitSize * dst[i]);
            size[i] /= splitSize;
            size.insert(size.begin()+i+1, splitSize);
            return true;
        }
    }
    return false;
}
// expand stride and size with expand value
static inline bool expandStrideSize(int* src, int* dst, int* size, int& num, int expandValue) {
#define MNN_3_INT_INSERT(x, i, y) if (i == 2) { x[2] = y; } else if (i == 1) { x[2] = x[1]; x[1] = y; } else if (i == 0) { x[2] = x[1]; x[1] = x[0]; x[0] = y; } else { return false; }
    for (int i = num-1; i >= 0; i--) {
        int splitSize = expandValue / src[i];
        if (!(expandValue % src[i] || size[i] % splitSize)) {
            MNN_3_INT_INSERT(src, i, expandValue)
            MNN_3_INT_INSERT(dst, i, (splitSize * dst[i]))
            size[i] /= splitSize;
            MNN_3_INT_INSERT(size, (i+1), splitSize)
            if (++num > 3) return false;
            return true;
        }
    }
    return false;
#undef MNN_3_INT_INSERT
}

bool TensorUtils::refTensorContent(Tensor* dst, const Tensor* src) {
    auto des = TensorUtils::getDescribe(dst);
    auto srcDes = TensorUtils::getDescribe(src);
    auto desO = TensorUtils::getDescribeOrigin(dst);
    auto srcDesO = TensorUtils::getDescribeOrigin(src);
    bool needMalloc = dst->buffer().host != src->buffer().host || dst->buffer().device != src->buffer().device || des->extra.offset != srcDes->extra.offset;
    desO->setBackend(srcDesO->getBackend());
    dst->buffer().host = src->buffer().host;
    dst->buffer().device = src->buffer().device;
    dst->buffer().flags = src->buffer().flags;
    des->extra.offset = srcDes->extra.offset;
    des->group = -1;
    return needMalloc;
}

static bool _ClipDst(int* stride, int srcOffset, int dstOffset, const int* srcSize, const int* dstSize, const int sizeNum, int* dstMax, int* dstMin) {
    /* Compute The range of dx, dy, dz:
     s0 * (dx-sx) + s1 * (dy-sy) + s2 * (dz-sz) + (doff-soff) = 0
     Assume the region won't be overlapped, then extract doff -> s0*xd+ s1*yd+s2*zd, soff -> s0*xs+s1*ys+s2*zs
     xd-xs=xo, yd-ys=yo, zd-zs=zo
     then:
     dx-sx+xo = 0
     dy-sy+yo = 0
     dz-sz+zo = 0
     dx=sx-xo -> [max(0, -xo), max(0, min(sxr-xo, dxr))]
     dy,dz compute the same
     **/
    
    int offsetBias = dstOffset - srcOffset;
    if (sizeNum == 0) {
        // All stride is zero, then size will be all one
        return offsetBias == 0;
    }
    int o[3] = {0, 0, 0};
    int validIndex[3] = {0, 1, 2};
    if (sizeNum == 2) {
        if (stride[0] < stride[1]) {
            validIndex[0] = 1;
            validIndex[1] = 0;
        }
    } else if (sizeNum > 2) {
        int maxs = stride[0];
        int mins = stride[0];
        int maxi = 0;
        int mini = 0;
        // Sort index by stride
        for (int i=1; i<sizeNum; ++i) {
            int s = stride[i];
            if (s > maxs) {
                maxs = s;
                maxi = i;
            }
            if (s < mins) {
                mins = s;
                mini = i;
            }
        }
        for (int i=0; i<sizeNum; ++i) {
            if (i != maxi && i != mini) {
                validIndex[1] = i;
                break;
            }
        }
        validIndex[0] = maxi;
        validIndex[2] = mini;
    }
    // Compute offset
    for (int i=0; i<sizeNum; ++i) {
        int s = stride[validIndex[i]];
        int xs = srcOffset / s;
        int xd = dstOffset / s;
        o[validIndex[i]] = xd-xs;
        srcOffset = srcOffset % s;
        dstOffset = dstOffset % s;
    }
    if (0 != srcOffset || 0 != dstOffset) {
        return false;
    }
    for (int i=0; i<sizeNum; ++i) {
        dstMin[i] = ALIMAX(0, -o[i]);
        dstMax[i] = ALIMIN(srcSize[i]-o[i], dstSize[i]);
    }
    return true;
}
static bool _RegionValid(int* stride, int offset, int* size, int sizeNum, size_t limitSize) {
    int maxOffset = offset;
    int minOffset = offset;
    // Check start and end
    for (int i=0; i<sizeNum; ++i) {
        if (stride[i] > 0) {
            maxOffset += (stride[i] * (size[i] - 1));
        } else {
            minOffset += (stride[i] * (size[i] - 1));
        }
    }
    if (minOffset < 0 || maxOffset >= limitSize) {
        return false;
    }
    return true;
}
class TensorUtils::FuseRegionStatus {
public:
    enum Status {
        FUSE_SRC_COPY,
        FUSE_DST_COPY,
        FUSE_REGION_COMPUTE
    };
    void apply(const Tensor::InsideDescribe::Region& srcReg, Tensor::InsideDescribe::Region& dstReg) {
        switch (mStatus) {
            case FUSE_SRC_COPY:
                dstReg.origin = srcReg.origin;
                dstReg.src.offset += srcReg.src.offset - srcReg.dst.offset;
                break;
            case FUSE_DST_COPY:
                dstReg.origin = srcReg.origin;
                dstReg.dst = srcReg.dst;
                dstReg.src = srcReg.src;
                dstReg.src.offset = mSrcOff;
                dstReg.dst.offset = mDstOff;
                dstReg.size[0] = srcReg.size[0];
                dstReg.size[1] = srcReg.size[1];
                dstReg.size[2] = srcReg.size[2];
                break;
            case FUSE_REGION_COMPUTE:
            {
                if (dstSize[0] == 0) {
                    dstReg.size[0] = 0;
                    dstReg.origin = nullptr;
                    break;
                }
                for (int i=0; i<3; ++i) {
                    dstReg.size[i] = 1;
                    dstReg.src.stride[i] = 0;
                    dstReg.dst.stride[i] = 0;
                }
                int valid[3] = {0, 0, 0};
                int offset = 3 - dstNum;
                if (dstNum > sizeNum) {
                    for (int i = 2; i >= 0; i--) {
                        if (i < dstNum) {
                            if (dstSize[i] == 1) {
                                expandIdx = i;
                            }
                            dstReg.size[i+offset] = dstMax[i] - dstMin[i];
                            valid[i] = dstSize[i] > 1;
                        } else {
                            dstReg.size[i+offset] = 1;
                            valid[i] = 0;
                        }
                    }
                } else {
                    for (int i=0; i<dstNum; ++i) {
                        dstReg.size[i+offset] = dstMax[i] - dstMin[i];
                        valid[i] = dstSize[i] > 1 ? 1 : 0;
                    }
                }
                int idx = 0;
                for (int i = 0; i < 3; i++) {
                    if (valid[i] > 0 || i == expandIdx) {
                        dstReg.src.stride[i+offset] = newSrc[idx];
                        dstReg.dst.stride[i+offset] = dstDst[idx++];
                    }
                }
                dstReg.origin = srcReg.origin;
                dstReg.src.offset = newSrcOffset;
                dstReg.dst.offset = newDstOffset;
            }
                break;
            default:
                break;
        }
    }
    bool match(const Tensor::InsideDescribe::Region& srcReg, const Tensor::InsideDescribe::Region& dstReg) {
        // dont deal size > 1 && stride <= 0
        for (int i = 0; i < 3; i++) {
            if (srcReg.size[i] > 1 && (srcReg.src.stride[i] <= 0 || srcReg.dst.stride[i] <= 0)) {
                return false;
            }
            if (dstReg.size[i] > 1 && (dstReg.src.stride[i] <= 0 || dstReg.dst.stride[i] <= 0)) {
                return false;
            }
        }
        bool copyValid = true;
        // src data isnot full data of dst
        if (srcReg.dst.offset > dstReg.src.offset ||
            srcReg.dst.stride[1] > srcReg.size[2] ||
            srcReg.dst.stride[2] > srcReg.size[1] * srcReg.size[2]) {
            copyValid = false;
        }
        int dstTotalSize = 1, srcTotalSize = 1;
        int dstSrcMin = dstReg.src.offset;
        int dstSrcMax = dstSrcMin;
        int srcDstMin = srcReg.dst.offset;
        int srcDstMax = srcDstMin;
        for (int i = 0; i < 3; i++) {
            srcDstMax += srcReg.dst.stride[i] * (srcReg.size[i] - 1);
            dstSrcMax += dstReg.src.stride[i] * (dstReg.size[i] - 1);
            if (dstReg.size[i] > 1) {
                dstTotalSize *= dstReg.size[i];
            }
            if (srcReg.size[i] > 1) {
                srcTotalSize *= srcReg.size[i];
            }
        }
        // src data is not full data of dst
        if (dstTotalSize > srcTotalSize) {
            copyValid = false;
        }
        // Valid range is from srcReg: srcDstMin - srcDstMax, if dst's srcReg exceed, not valid for copy
        if (srcDstMin > dstSrcMin || srcDstMax < dstSrcMax) {
            copyValid = false;
        }
        // src copy fuse
        if (isCopyRegion(srcReg) && copyValid) {
            mStatus = FUSE_SRC_COPY;
            return true;
        }
        // dst copy fuse
        if (isCopyRegion(dstReg) && dstTotalSize == srcTotalSize && copyValid) {
            mSrcOff = dstReg.src.offset - srcReg.dst.offset;
            mDstOff = dstReg.dst.offset;
            mSrcOff = offsetCompute(srcReg, dstReg.src.offset, srcReg.dst.offset, true) + srcReg.src.offset;
            if (!(srcReg.src.stride[2] > 0 && mSrcOff % srcReg.src.stride[2] != 0)) {
                // when transpose + slice, offset is not align can't fuse
                mStatus = FUSE_DST_COPY;
                return true;
            }
        }
    #define MNN_3_INT_INIT(x, y) { x[0] = y; x[1] = y; x[2] = y; }
        MNN_3_INT_INIT(dstStride, -1)
        MNN_3_INT_INIT(srcStride, -1)
        expandIdx = -1;
    #undef MNN_3_INT_INIT
        srcNum = 0, dstNum = 0, sizeNum = 0;
        for (int i = 0; i < 3; i++) {
            if (srcReg.size[i] > 1) {
                srcStride[srcNum] = srcReg.dst.stride[i];
                srcDst[srcNum]    = srcReg.dst.stride[i];
                srcSrc[srcNum]    = srcReg.src.stride[i];
                srcSize[srcNum]   = srcReg.size[i];
                srcNum++;
            }
            if (dstReg.size[i] > 1) {
                dstStride[dstNum] = dstReg.src.stride[i];
                dstDst[dstNum]    = dstReg.dst.stride[i];
                dstSrc[dstNum]    = dstReg.src.stride[i];
                dstSize[dstNum]   = dstReg.size[i];
                dstNum++;
            }
        }
        sizeNum = dstNum;
    #define MNN_3_INT_DIFF(r, x, y, i) if ((x[i] != y[0]) && (x[i] != y[1]) && (x[i] != y[2])) { if (r > 0) { return false; } else { r = x[i]; } }
        int srcExtra = -1, dstExtra = -1;
        MNN_3_INT_DIFF(srcExtra, srcStride, dstStride, 0)
        MNN_3_INT_DIFF(srcExtra, srcStride, dstStride, 1)
        MNN_3_INT_DIFF(srcExtra, srcStride, dstStride, 2)
        MNN_3_INT_DIFF(dstExtra, dstStride, srcStride, 0)
        MNN_3_INT_DIFF(dstExtra, dstStride, srcStride, 1)
        MNN_3_INT_DIFF(dstExtra, dstStride, srcStride, 2)
    #undef MNN_3_INT_DIFF
        if (dstExtra > 0) {
            if (!expandStrideSize(srcDst, srcSrc, srcSize, srcNum, dstExtra)) {
                return false;
            }
        }
        if (srcExtra > 0) {
            if (!expandStrideSize(dstSrc, dstDst, dstSize, dstNum, srcExtra)) {
                return false;
            }
        }
        // reorder srcSrc to newSrc by align srcDst and dstSrc
        for (int i = 0; i < srcNum; i++) {
            int index = -1;
            for (int j = 0; j < dstNum; j++) {
                if (dstSrc[j] == srcDst[i]) {
                    index = j;
                    break;
                }
            }
            if (-1 == index) {
                return false;
            }
            newSrc[index] = srcSrc[i];
            newSrcSize[index] = srcSize[i];
        }
        // set final size and set expandIdx if expand val is 1
        newSrcOffset = offsetCompute(srcReg, dstReg.src.offset, srcReg.dst.offset, true) + srcReg.src.offset;
        bool valid = _ClipDst(dstSrc, srcReg.dst.offset, dstReg.src.offset, newSrcSize, dstSize, dstNum, dstMax, dstMin);
        if (!valid) {
            return false;
        }
        newDstOffset = dstReg.dst.offset;
        for (int i=0; i<dstNum; ++i) {
            if (dstMax[i] <= dstMin[i]) {
                // Set region as empty
                dstSize[0] = 0;
                dstSize[1] = 0;
                dstSize[2] = 0;
                break;
            }
            if (dstMin[i] > 0) {
                newDstOffset += dstMin[i] * dstDst[i];
                newSrcOffset += dstMin[i] * newSrc[i];
            }
        }
        mStatus = FUSE_REGION_COMPUTE;
        return true;
    }
private:
    int mStatus;
    int mSrcOff;
    int mDstOff;
    // general fuse
    int srcDst[3], srcSrc[3], dstSrc[3], dstDst[3], srcSize[3], dstSize[3], newSrc[3], dstStride[3], srcStride[3];
    int dstMin[3],dstMax[3];
    int newSrcSize[3];
    int srcNum, dstNum, sizeNum;
    int newSrcOffset;
    int newDstOffset;
    int expandIdx;
};

TensorUtils::FuseWrap::FuseWrap() {
    mStatus = new FuseRegionStatus;
}
TensorUtils::FuseWrap::~ FuseWrap() {
    delete mStatus;
}
bool TensorUtils::FuseWrap::match(const Tensor::InsideDescribe::Region& srcReg, const Tensor::InsideDescribe::Region& dstReg) {
    return mStatus->match(srcReg, dstReg);
}
void TensorUtils::FuseWrap::apply(const Tensor::InsideDescribe::Region& srcReg, Tensor::InsideDescribe::Region& dstReg) {
    mStatus->apply(srcReg, dstReg);
}

void TensorUtils::adjustTensorForCompability(Tensor* newTensor) {
    if (newTensor->dimensions() < 4) {
        for (int n = newTensor->dimensions(); n < 4; ++n) {
            newTensor->setLength(n, 1);
        }
    }
}

Tensor::DimensionType TensorUtils::getDimType(const Tensor* t) {
    auto format = TensorUtils::getDescribe(t)->dimensionFormat;
    switch (format) {
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        default:
            break;
    }
    return Tensor::TENSORFLOW;
}

std::vector<float> TensorUtils::getQuantInfo(const Tensor* t) {
    float scale = getDescribe(t)->quantAttr ? getDescribe(t)->quantAttr->scale : 0.0f;
    float zero = getDescribe(t)->quantAttr ? getDescribe(t)->quantAttr->zero : 0.0f;
    float min = getDescribe(t)->quantAttr ? getDescribe(t)->quantAttr->min : -127.0f;
    float max = getDescribe(t)->quantAttr ? getDescribe(t)->quantAttr->max : 127.0f;
    return {scale, zero, min, max};
}

Tensor::InsideDescribe* TensorUtils::getDescribeOrigin(const Tensor* tensor) {
    return tensor->mDescribe;
}
size_t TensorUtils::getRawSize(const Tensor* t) {
    size_t len = 1;
    int dim = t->dimensions();
    for (int i=0; i<dim; ++i) {
        len *= (size_t)t->length(i);
    }
    return len;
}
void TensorUtils::setRasterInputs(Command* cmd) {
    auto& regions = TensorUtils::getDescribe(cmd->outputs[0])->regions;
    cmd->inputs.resize(regions.size());
    for (int i=0; i<regions.size(); ++i) {
#ifdef DEBUG
        for (int j=0; j<3; ++j) {
            MNN_ASSERT(regions[i].size[j] > 0);
        }
#endif
        cmd->inputs[i] = regions[i].origin;
        auto des = getDescribe(regions[i].origin);
    }
}

int TensorUtils::getTensorChannelPack(const Tensor* tensor) {
    auto srcDes = TensorUtils::getDescribe(tensor);
    return srcDes->support_pack16 ? srcDes->channel_pack_num : 4;
}

void TensorUtils::setTensorChannelPack(const Tensor* tensor, int pack) {
    auto srcDes = TensorUtils::getDescribe(tensor);
    srcDes->channel_pack_num = srcDes->support_pack16 ? pack : 4;
}

void TensorUtils::setTensorSupportPack(const Tensor* tensor, bool flag) {
    auto srcDes = TensorUtils::getDescribe(tensor);
    srcDes->support_pack16 = flag;
}

void TensorUtils::setTensorPad(const Tensor* tensor, int left, int right, int bottom, int top) {
    auto srcDes = TensorUtils::getDescribe(tensor);
    srcDes->mPads.left = std::max(srcDes->mPads.left,left);
    srcDes->mPads.right = std::max(srcDes->mPads.right, right);
    srcDes->mPads.bottom = std::max(srcDes->mPads.bottom, bottom);
    srcDes->mPads.top = std::max(srcDes->mPads.top, top);
}

} // namespace MNN
