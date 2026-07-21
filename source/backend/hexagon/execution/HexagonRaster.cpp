#include "HexagonRaster.hpp"

#include "backend/hexagon/backend/HexagonBackend.hpp"
#include "backend/hexagon/backend/HexagonRuntime.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/hexagon/htp-ops-lib/include/htp_command.h"
#include <algorithm>
namespace MNN {
//#define HEXAGON_DEBUG

namespace {
static void appendZeroCmd(HexagonBackend* backend, Tensor* output, std::vector<HexagonCommand>& dst) {
    auto dstDev = HexagonBackend::getDevicePtr(output);
    int size = (int)backend->getSize(output);
    int params[] = {size};
    std::vector<std::pair<int, int>> inputFds;
    std::vector<std::pair<int, int>> outputFds = {dstDev};
    std::vector<Tensor*> cmdOutputs = {output};
    dst.emplace_back();
    dst.back().build(backend, DSP_OP_ZERO, params, sizeof(params), inputFds,  outputFds,  {}, cmdOutputs);
}
}

HexagonRaster::HexagonRaster(Backend* backend) : HexagonExecution(backend) {
}

HexagonRaster::~HexagonRaster() = default;

void HexagonRaster::releaseDynamicTemps() {
    for (auto& iter : mTempInput) {
        if (iter.second && TensorUtils::getDescribeOrigin(iter.second.get())->mem.get() != nullptr) {
            backend()->onReleaseBuffer(iter.second.get(), Backend::DYNAMIC);
        }
    }
    if (mTempOutput && TensorUtils::getDescribeOrigin(mTempOutput.get())->mem.get() != nullptr) {
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
}

ErrorCode HexagonRaster::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    std::vector<HexagonCommand>& dst) {
    auto output = outputs[0];
    OpCommonUtils::rasterInputReset(inputs, output);
    mBytes = HexagonBackend::getBytes(outputs[0]);
    auto des = TensorUtils::getDescribe(output);
    auto outputDes = TensorUtils::getDescribe(output);
    auto hexagonBackend = static_cast<HexagonBackend*>(backend());
    const int pack = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info().vectorSize;
    dst.reserve(dst.size() + des->regions.size() + inputs.size() + outputs.size() + 8);

    mNeedZero = !TensorUtils::regionIsFull(output);

    releaseDynamicTemps();
    mTempInput.clear();
    mCacheRegions.clear();
    mTempOutput = nullptr;
    mTempInputCopy.clear();

    mSingleConvert.type = 0;
    if (des->regions.size() == 1) {
        OpCommonUtils::turnRegion2Convert(des->regions[0], output, mSingleConvert);
        if (mSingleConvert.type == 1) {
            auto input = inputs[0];
            auto srcDev = HexagonBackend::getDevicePtr(input);
            auto dstDev = HexagonBackend::getDevicePtr(output);

            int batch = mSingleConvert.batch;
            int channel = mSingleConvert.channel;
            int area = mSingleConvert.area;

            auto source = TensorUtils::getDescribe(input)->dimensionFormat;
            auto dest = TensorUtils::getDescribe(output)->dimensionFormat;

            int convertType;
            if (input->dimensions() > 4 && output->dimensions() > 4) {
                convertType = 2;
            } else if (input->dimensions() <= 1 || source == dest) {
                convertType = 2;
                if (source == MNN_DATA_FORMAT_NC4HW4) {
                    auto pack = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info().vectorSize;
                    if (pack <= 0) pack = 4;
                    channel = UP_DIV(channel, pack) * pack;
                }
            } else {
                convertType = (source == MNN_DATA_FORMAT_NC4HW4) ? 0 : 1;
            }
            int params[] = {batch, area, channel, mBytes, convertType};
#ifdef HEXAGON_DEBUG
            MNN_PRINT("HexagonRaster single convert cmd params: batch=%d, area=%d, channel=%d, mBytes=%d, convertType=%d\n", batch, area, channel, mBytes, convertType);
#endif
            std::vector<std::pair<int, int>> inputFds = {srcDev};
            std::vector<std::pair<int, int>> outputFds = {dstDev};

            std::vector<Tensor*> cmdInputs = {input};
            std::vector<Tensor*> cmdOutputs = {output};
            dst.emplace_back();
            dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_TENSOR_CONVERT, params, sizeof(params),
                             inputFds,  outputFds,  cmdInputs, cmdOutputs);
            return NO_ERROR;
        }
    }
    mRegionCount = (int)des->regions.size();
    if (mRegionCount == 0) {
        return NO_ERROR;
    }

    auto midFormat = MNN_DATA_FORMAT_NCHW;

    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && output->dimensions() == 4 && des->regions.size() == 1) {
        auto& slice = des->regions[0];
        auto origin = slice.origin;
        if (origin != nullptr) {
            auto originDes = TensorUtils::getDescribe(origin);
            int srcArea = 1;
            int dstArea = 1;
            for (int d = 2; d < origin->dimensions(); ++d) {
                srcArea *= origin->length(d);
            }
            for (int d = 2; d < output->dimensions(); ++d) {
                dstArea *= output->length(d);
            }
            const int inner = slice.size[2];
            const bool supportFlattenCW =
                originDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                origin->batch() == 1 && dstArea == 1 && inner > 0 &&
                origin->channel() % pack == 0 && output->channel() == origin->channel() * inner &&
                output->batch() == slice.size[0] && srcArea == slice.size[0] * inner &&
                slice.src.offset == 0 && slice.dst.offset == 0 &&
                slice.src.stride[0] == origin->width() && slice.src.stride[1] == srcArea && slice.src.stride[2] == 1 &&
                slice.dst.stride[0] == output->channel() && slice.dst.stride[1] == inner && slice.dst.stride[2] == 1;
            if (supportFlattenCW) {
                auto srcDev = HexagonBackend::getDevicePtr(origin);
                auto dstDev = HexagonBackend::getDevicePtr(output);
                int params[] = {output->batch(), inner, origin->channel(), mBytes, 3};
#ifdef HEXAGON_DEBUG
                MNN_PRINT("HexagonRaster flatten-cw convert cmd params: batch=%d, inner=%d, channel=%d, mBytes=%d, convertType=3\n",
                          params[0], params[1], params[2], params[3]);
#endif
                std::vector<std::pair<int, int>> inputFds = {srcDev};
                std::vector<std::pair<int, int>> outputFds = {dstDev};
                std::vector<Tensor*> cmdInputs = {origin};
                std::vector<Tensor*> cmdOutputs = {output};
                dst.emplace_back();
                dst.back().build(hexagonBackend, DSP_OP_TENSOR_CONVERT, params, sizeof(params),
                                 inputFds, outputFds, cmdInputs, cmdOutputs);
                return NO_ERROR;
            }
        }
    }

    if (MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat &&
        (output->dimensions() == 3 || output->dimensions() == 4)) {
        mTempOutput.reset(new Tensor);
        TensorUtils::setupTensorInfo(output, mTempOutput.get(), midFormat);
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            releaseDynamicTemps();
            return OUT_OF_MEMORY;
        }
    }

    TensorUtils::FuseWrap fuseUtils;
    for (int i = 0; i < des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        auto origin = slice.origin;
        if (nullptr == origin) {
            continue;
        }

        auto originDesc = TensorUtils::getDescribe(origin);
        if (originDesc == nullptr) {
            mTempInputCopy.emplace_back(std::make_pair(origin, &slice));
            continue;
        }
        if (originDesc->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            mTempInputCopy.emplace_back(std::make_pair(origin, &slice));
            continue;
        }

        int channel = origin->channel();
        int batch = origin->batch();
        int area = 1;
        for (int d = 2; d < origin->dimensions(); d++) {
            area *= origin->length(d);
        }

        if (batch > 0 && area == 1 && channel % pack == 0 && output->channel() % pack == 0) {
            Tensor::InsideDescribe::Region regionTmp;
            regionTmp.src.offset = 0;
            regionTmp.src.stride[0] = batch * pack;
            regionTmp.src.stride[1] = pack;
            regionTmp.src.stride[2] = 1;
            regionTmp.dst.offset = 0;
            regionTmp.dst.stride[0] = pack;
            regionTmp.dst.stride[1] = channel;
            regionTmp.dst.stride[2] = 1;
            regionTmp.size[0] = channel / pack;
            regionTmp.size[1] = batch;
            regionTmp.size[2] = pack;
            regionTmp.origin = slice.origin;
            bool merge = fuseUtils.match(regionTmp, slice);
            if (merge) {
                std::shared_ptr<Tensor::InsideDescribe::Region> newSlice(new Tensor::InsideDescribe::Region);
                *newSlice = slice;
                fuseUtils.apply(regionTmp, *newSlice);
                mTempInputCopy.emplace_back(std::make_pair(origin, newSlice.get()));
                mCacheRegions.emplace_back(newSlice);
                continue;
            }
        }

        if (batch == 1 && channel % pack == 0 && output->channel() % pack == 0) {
            Tensor::InsideDescribe::Region regionTmp;
            regionTmp.src.offset = 0;
            regionTmp.src.stride[0] = area * pack;
            regionTmp.src.stride[1] = 1;
            regionTmp.src.stride[2] = pack;
            regionTmp.dst.offset = 0;
            regionTmp.dst.stride[0] = area * pack;
            regionTmp.dst.stride[1] = area;
            regionTmp.dst.stride[2] = 1;
            regionTmp.size[0] = channel / pack;
            regionTmp.size[1] = pack;
            regionTmp.size[2] = area;
            regionTmp.origin = slice.origin;
            bool merge = fuseUtils.match(regionTmp, slice);
            if (merge) {
                std::shared_ptr<Tensor::InsideDescribe::Region> newSlice(new Tensor::InsideDescribe::Region);
                *newSlice = slice;
                fuseUtils.apply(regionTmp, *newSlice);
                mTempInputCopy.emplace_back(std::make_pair(origin, newSlice.get()));
                mCacheRegions.emplace_back(newSlice);
                continue;
            }
        }

        auto tempTensor = mTempInput.find(origin);
        if (tempTensor == mTempInput.end()) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(origin, newTensor.get());
            TensorUtils::getDescribe(newTensor.get())->dimensionFormat = midFormat;
            TensorUtils::getDescribe(newTensor.get())->quantAttr = TensorUtils::getDescribe(origin)->quantAttr;
            TensorUtils::getDescribe(newTensor.get())->applyQuant = TensorUtils::getDescribe(origin)->applyQuant;
            newTensor->buffer().type = origin->getType();
            TensorUtils::setLinearLayout(newTensor.get());
            mTempInput.insert(std::make_pair(origin, newTensor));
            auto res = backend()->onAcquireBuffer(newTensor.get(), Backend::DYNAMIC);
            if (!res) {
                releaseDynamicTemps();
                return OUT_OF_MEMORY;
            }
            TensorUtils::getDescribe(newTensor.get())->useCount = TensorUtils::getDescribe(origin)->useCount;
            tempTensor = mTempInput.find(origin);
        }
        mTempInputCopy.emplace_back(std::make_pair(tempTensor->second.get(), &slice));
    }

    for (auto& iter : mTempInput) {
        auto input = iter.first;
        auto output = iter.second.get();
        auto& subIb = input->buffer();
        auto source = TensorUtils::getDescribe(input)->dimensionFormat;
        auto dest = TensorUtils::getDescribe(output)->dimensionFormat;

        int dims = subIb.dimensions;
        int batch = (dims > 0) ? subIb.dim[0].extent : 1;
        int channel = (dims > 1) ? subIb.dim[1].extent : 1;
        int height = (dims > 2) ? subIb.dim[2].extent : 1;
        int width = (dims > 3) ? subIb.dim[3].extent : 1;
        int area = height * width;

        auto srcDev = HexagonBackend::getDevicePtr(input);
        auto dstDev = HexagonBackend::getDevicePtr(output);

        int convertType;
        if (subIb.dimensions <= 1 || source == dest) {
            convertType = 2;
            if (source == MNN_DATA_FORMAT_NC4HW4) {
                auto pack = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info().vectorSize;
                if (pack <= 0) pack = 4;
                channel = UP_DIV(channel, pack) * pack;
            }
        } else {
            convertType = (source == MNN_DATA_FORMAT_NC4HW4) ? 0 : 1;
        }
        int params[] = {batch, area, channel, mBytes, convertType};
#ifdef HEXAGON_DEBUG
        MNN_PRINT("HexagonRaster pre convert cmd params: batch=%d, area=%d, channel=%d, mBytes=%d, convertType=%d\n", batch, area, channel, mBytes, convertType);
#endif
        std::vector<std::pair<int, int>> inputFds = {srcDev};
        std::vector<std::pair<int, int>> outputFds = {dstDev};

        std::vector<Tensor*> cmdInputs = {input};
        std::vector<Tensor*> cmdOutputs = {output};
            dst.emplace_back();
            dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_TENSOR_CONVERT, params, sizeof(params),
                             inputFds,  outputFds,  cmdInputs, cmdOutputs);
    }

    auto dstDev = std::make_pair(0, 0);
    if (mTempOutput != nullptr) {
        dstDev = HexagonBackend::getDevicePtr(mTempOutput.get());
    } else {
        dstDev = HexagonBackend::getDevicePtr(output);
    }

    if (mNeedZero) {
        auto zeroTarget = mTempOutput != nullptr ? mTempOutput.get() : output;
        appendZeroCmd(static_cast<HexagonBackend*>(backend()), zeroTarget, dst);
    }
    mRegionCount = (int)mTempInputCopy.size();
    std::vector<RasterRegion> regions(mRegionCount);

    std::map<std::pair<int, int>, int> srcFdToIndex;
    std::vector<std::pair<int, int>> uniqueSrcFds;
    std::vector<Tensor*> uniqueSrcTensors;

    for (int i = 0; i < mRegionCount; ++i) {
        auto& iter = mTempInputCopy[i];
        auto& slice = *(iter.second);
        auto srcDev = HexagonBackend::getDevicePtr(iter.first);

        auto it = srcFdToIndex.find(srcDev);
        int srcIndex = 0;
        if (it == srcFdToIndex.end()) {
            srcIndex = (int)uniqueSrcFds.size();
            srcFdToIndex[srcDev] = srcIndex;
            uniqueSrcFds.push_back(srcDev);
            uniqueSrcTensors.push_back(iter.first);
        } else {
            srcIndex = it->second;
        }

        auto& region = regions[i];
        region.srcIndex = srcIndex;
        region.srcOffset = slice.src.offset;
        region.dstOffset = slice.dst.offset;

        for (int d = 0; d < 3; ++d) {
            region.size[d] = slice.size[d];
            region.srcStride[d] = slice.src.stride[d];
            region.dstStride[d] = slice.dst.stride[d];
        }
#ifdef HEXAGON_DEBUG
        auto& reg = regions[i];
        MNN_PRINT("i:%d, size: %d, %d, %d, srcIndex: %d, srcOffset:%d, dstOffset:%d, srcStride: %d, %d, %d, dstStride: %d, %d, %d\n", i, reg.size[0], reg.size[1], reg.size[2], reg.srcIndex, reg.srcOffset, reg.dstOffset, reg.srcStride[0], reg.srcStride[1], reg.srcStride[2], reg.dstStride[0], reg.dstStride[1], reg.dstStride[2]);
#endif
    }

    const int MAX_SRC_PER_CMD = 10;
    int totalSrcCount = (int)uniqueSrcFds.size();
    int numGroups = (totalSrcCount + MAX_SRC_PER_CMD - 1) / MAX_SRC_PER_CMD;

    for (int g = 0; g < numGroups; ++g) {
        int startIdx = g * MAX_SRC_PER_CMD;
        int endIdx = std::min(startIdx + MAX_SRC_PER_CMD, totalSrcCount);
        int groupSrcCount = endIdx - startIdx;

        std::vector<std::pair<int, int>> groupSrcFds(uniqueSrcFds.begin() + startIdx, uniqueSrcFds.begin() + endIdx);

        std::vector<RasterRegion> groupRegions;
        for (int i = 0; i < mRegionCount; ++i) {
            int origSrcIndex = regions[i].srcIndex;
            if (origSrcIndex >= startIdx && origSrcIndex < endIdx) {
                RasterRegion adjustedRegion = regions[i];
                adjustedRegion.srcIndex = origSrcIndex - startIdx;
                groupRegions.push_back(adjustedRegion);
            }
        }

        if (groupRegions.empty()) {
            continue;
        }

        struct MergedRasterParam {
            int regionCount;
            int bytes;
            int srcNumber;
            // Variable length array of regions follows
        } __attribute__((packed));

        size_t paramSize = sizeof(MergedRasterParam) + groupRegions.size() * sizeof(RasterRegion);
        std::vector<uint8_t> paramData(paramSize);
        MergedRasterParam* params = reinterpret_cast<MergedRasterParam*>(paramData.data());
        params->regionCount = groupRegions.size();
        params->bytes = mBytes;
        params->srcNumber = groupSrcCount;
        memcpy(paramData.data() + sizeof(MergedRasterParam), groupRegions.data(), groupRegions.size() * sizeof(RasterRegion));

#ifdef HEXAGON_DEBUG
        MNN_PRINT("HexagonRaster raster blit cmd params: regionCount=%d, mBytes=%d, src_number=%d\n", (int)groupRegions.size(), mBytes, groupSrcCount);
#endif
        std::vector<std::pair<int, int>> inputFds = groupSrcFds;
        std::vector<std::pair<int, int>> outputFds = {dstDev};
        auto cmdOutputTensor = mTempOutput != nullptr ? mTempOutput.get() : outputs[0];

        std::vector<Tensor*> cmdInputs;
        for (int i = startIdx; i < endIdx; ++i) {
            cmdInputs.push_back(uniqueSrcTensors[i]);
        }
        std::vector<Tensor*> cmdOutputs = {cmdOutputTensor};

        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_RASTER_BLIT, paramData.data(), paramSize,
                         inputFds,  outputFds,  cmdInputs, cmdOutputs);
    }
    if (nullptr != mTempOutput) {
        auto input = mTempOutput.get();
        auto outputTensor = outputs[0];
        auto& subIb = input->buffer();
        auto source = TensorUtils::getDescribe(input)->dimensionFormat;
        auto dest = TensorUtils::getDescribe(outputTensor)->dimensionFormat;

        int dims = subIb.dimensions;
        int batch = (dims > 0) ? subIb.dim[0].extent : 1;
        int channel = (dims > 1) ? subIb.dim[1].extent : 1;
        int height = (dims > 2) ? subIb.dim[2].extent : 1;
        int width = (dims > 3) ? subIb.dim[3].extent : 1;
        int area = height * width;

        auto srcDev2 = HexagonBackend::getDevicePtr(input);
        auto dstDev2 = HexagonBackend::getDevicePtr(outputTensor);

        int convertType;
        if (subIb.dimensions <= 1 || source == dest) {
            convertType = 2;
            if (source == MNN_DATA_FORMAT_NC4HW4) {
                auto pack = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info().vectorSize;
                if (pack <= 0) pack = 4;
                channel = UP_DIV(channel, pack) * pack;
            }
        } else {
            convertType = (source == MNN_DATA_FORMAT_NC4HW4) ? 0 : 1;
        }
        int params2[] = {batch, area, channel, mBytes, convertType};
#ifdef HEXAGON_DEBUG
        MNN_PRINT("HexagonRaster post convert cmd params: batch=%d, area=%d, channel=%d, mBytes=%d, convertType=%d\n", batch, area, channel, mBytes, convertType);
#endif
        std::vector<std::pair<int, int>> inputFds2 = {srcDev2};
        std::vector<std::pair<int, int>> outputFds2 = {dstDev2};

        std::vector<Tensor*> cmdInputs2 = {input};
        std::vector<Tensor*> cmdOutputs2 = {outputTensor};
        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_TENSOR_CONVERT, params2, sizeof(params2),
                         inputFds2,  outputFds2,  cmdInputs2, cmdOutputs2);
    }

    releaseDynamicTemps();
    return NO_ERROR;
}

HexagonRaster* HexagonRaster::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_Raster) {
        return nullptr;
    }
    return new HexagonRaster(backend);
}

} // namespace MNN
