//
//  GeometryTensorArray.cpp
//  MNN
//
//  Created by MNN on 2020/12/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
// get a pair <ElemOffset, ElemSize>
static std::pair<int, int> getElemSize(const Tensor* t, int index) {
    auto des = TensorUtils::getDescribe(t);
    const auto& shapes = des->tensorArrayAttr->elemShape;
    int elemSize = 1;
    if (index < 0) {
        index = index + shapes.size();
    }
    if (!des->tensorArrayAttr->isIdenticalShape && shapes.size() > index) {
        int offset = 0;
        for (int i = 0; i <= index; i++) {
            elemSize = 1;
            std::for_each(shapes[i].begin(), shapes[i].end(), [&elemSize](int x) { elemSize *= x; });
            offset += elemSize;
        }
        return {offset - elemSize, elemSize};
    } else if (shapes.size() >= 1) {
        elemSize = 1;
        std::for_each(shapes[0].begin(), shapes[0].end(), [&elemSize](int x) { elemSize *= x; });
        return {index * elemSize, elemSize};
    } else {
        MNN_ASSERT(false);
        return {0, 0};
    }
}

static bool isFirstWrite(const Tensor::InsideDescribe::NativeInsideDescribe* des) {
    if (des->tensorArrayAttr->elemShape.empty()) {
        return true;
    }
    for (const auto& dim : des->tensorArrayAttr->elemShape[0]) {
        if (dim < 0) {
            return  true;
        }
    }
    return false;
}

class GeometryTensorArray : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (TensorUtils::getDescribe(outputs[1])->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        if (TensorUtils::getDescribe(outputs[1])->tensorArrayAttr->arraySize > 0) {
            auto type = outputs[1]->getType();
            auto zeroConst = context.allocConst(op, {}, type);
            if (type == halide_type_of<float>()) {
                zeroConst->host<float>()[0] = 0.0;
            } else {
                zeroConst->host<int>()[0] = 0;
            }
            for (int i = 0; i < 2; i++) {
                auto des = TensorUtils::getDescribe(outputs[i]);
                des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                auto& regions = des->regions;
                regions.resize(1);
                regions[0].origin = zeroConst.get();
                regions[0].size[0] = outputs[1]->elementSize();
                regions[0].src.stride[0] = 0;
            }
        }
        return true;
    }
};
class GeometryTensorArraySize : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[1];
        if (TensorUtils::getDescribe(tensorArrayInput)->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        if (!context.allocTensor(outputs[0])) {
            return false;
        }
        outputs[0]->host<int>()[0] = TensorUtils::getDescribe(tensorArrayInput)->tensorArrayAttr->arraySize;
        return true;
    }
};

class GeometryTensorArrayRead : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[2];
        if (TensorUtils::getDescribe(tensorArrayInput)->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        auto output    = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.resize(1);
        auto& reg = outputDes->regions[0];
        auto index = inputs[1]->host<uint32_t>()[0];
        auto elemSize = getElemSize(tensorArrayInput, index);
        reg.origin = tensorArrayInput;
        reg.src.offset = elemSize.first;
        reg.src.stride[0] = 1;
        reg.src.stride[1] = 1;
        reg.src.stride[2] = 1;
        reg.dst.offset = 0;
        reg.dst.stride[0] = 1;
        reg.dst.stride[1] = 1;
        reg.dst.stride[2] = 1;
        reg.size[0] = elemSize.second;
        reg.size[1] = 1;
        reg.size[2] = 1;
        return true;
    }
};

class GeometryTensorArrayWrite : public GeometryComputer {
public:
    // tensor(index < seq_length) will insert instead of overwrite when onnxInsert=true
    GeometryTensorArrayWrite(bool insertMode) : mInsertMode(insertMode) { }
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[3];
        auto inDes = TensorUtils::getDescribe(tensorArrayInput);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        auto output    = outputs[0];
        auto outDes = TensorUtils::getDescribe(output);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        int oldSize = inDes->tensorArrayAttr->arraySize;
        int writeIndex = inputs[1]->host<uint32_t>()[0];
        // mInsertMode=true mean onnx mode, which position tensor is int32 instead of uint32
        if (mInsertMode) {
            writeIndex = inputs[1]->host<int32_t>()[0];
            writeIndex += (writeIndex < 0 ? inDes->tensorArrayAttr->arraySize: 0); // [-n, n]
        }
        auto elemSize = getElemSize(output, writeIndex);
        outDes->regions.clear();
        // support insertMode=true/false, easier to understand
        int regionSize = (writeIndex > 0) + 1 + (writeIndex < outDes->tensorArrayAttr->arraySize - 1);
        outDes->regions.reserve(regionSize);
        /*
         src: [leftData][writeIndex][rightData]
         dst: [leftData][writeTensor][rightData]
         */
        // 1. write Tensor to dst TensorArray [must]
        if (elemSize.second == 0) {
            return true;
        }
        {
            Tensor::InsideDescribe::Region writeTensorRegion;
            writeTensorRegion.origin = inputs[2];
            writeTensorRegion.src.offset = 0;
            writeTensorRegion.src.stride[0] = 1;
            writeTensorRegion.src.stride[1] = 1;
            writeTensorRegion.src.stride[2] = 1;
            writeTensorRegion.dst.offset = elemSize.first;
            writeTensorRegion.dst.stride[0] = 1;
            writeTensorRegion.dst.stride[1] = 1;
            writeTensorRegion.dst.stride[2] = 1;
            writeTensorRegion.size[0] = elemSize.second;
            writeTensorRegion.size[1] = 1;
            writeTensorRegion.size[2] = 1;
            MNN_ASSERT(elemSize.second > 0);
            outDes->regions.emplace_back(std::move(writeTensorRegion));
        }
        if (regionSize == 1) {
            return true;
        }
        // first write data, set pre zero
        bool firstWrite = isFirstWrite(inDes);
        if (firstWrite) {
            auto type = tensorArrayInput->getType();
            auto zeroConst = context.allocConst(op, {}, type);
            if (type == halide_type_of<float>()) {
                zeroConst->host<float>()[0] = 0.0;
            } else {
                zeroConst->host<int>()[0] = 0;
            }
            tensorArrayInput = zeroConst.get();
        }
        // 2. copy TensorArray leftData [optional]
        if (writeIndex > 0 && elemSize.first > 0) {
            Tensor::InsideDescribe::Region leftDataRegion;
            leftDataRegion.origin = tensorArrayInput;
            leftDataRegion.src.offset = 0;
            leftDataRegion.src.stride[0] = !firstWrite;
            leftDataRegion.src.stride[1] = 1;
            leftDataRegion.src.stride[2] = 1;
            leftDataRegion.dst.offset = 0;
            leftDataRegion.dst.stride[0] = 1;
            leftDataRegion.dst.stride[1] = 1;
            leftDataRegion.dst.stride[2] = 1;
            leftDataRegion.size[0] = elemSize.first;
            leftDataRegion.size[1] = 1;
            leftDataRegion.size[2] = 1;
            outDes->regions.emplace_back(std::move(leftDataRegion));
        }
        // 3. copy TensorArray rightData [optional]
        int rightSize = oldSize - writeIndex - (mInsertMode ? 0 : 1);
        if (rightSize > 0) {
            auto last = getElemSize(inputs[0], oldSize-1);
            int totalSize = last.first + last.second;
            int offset = elemSize.first + elemSize.second;
            int offsetSrc = offset - (mInsertMode ? elemSize.second: 0);
            int rightRegionSize = totalSize - offsetSrc;
            if (rightRegionSize > 0) {
                Tensor::InsideDescribe::Region rightDataRegion;
                rightDataRegion.origin = tensorArrayInput;
                rightDataRegion.src.offset = (!firstWrite) * offsetSrc;
                rightDataRegion.src.stride[0] = !firstWrite;
                rightDataRegion.src.stride[1] = 1;
                rightDataRegion.src.stride[2] = 1;
                rightDataRegion.dst.offset = offset;
                rightDataRegion.dst.stride[0] = 1;
                rightDataRegion.dst.stride[1] = 1;
                rightDataRegion.dst.stride[2] = 1;
                rightDataRegion.size[0] = rightRegionSize;
                rightDataRegion.size[1] = 1;
                rightDataRegion.size[2] = 1;
                outDes->regions.emplace_back(std::move(rightDataRegion));
            }
        }
        return true;
    }
private:
    bool mInsertMode;
};

class GeometryTensorArrayGather : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[2];
        auto inDes = TensorUtils::getDescribe(tensorArrayInput);
        if (inDes->tensorArrayAttr == nullptr) {
            return false;
        }
        auto indicesTensor = inputs[1];
        std::vector<int> indices(indicesTensor->elementSize());
        for (int i = 0; i < indices.size(); i++) {
            indices[i] = indicesTensor->host<int>()[i];
        }
        auto output    = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.resize(indices.size());
        int arraySize = inDes->tensorArrayAttr->arraySize;
        int dstOffset = 0;
        for (int i = 0; i < indices.size(); i++) {
            MNN_ASSERT(indices[i] < arraySize);
            auto elemSize = getElemSize(tensorArrayInput, indices[i]);
            auto& reg = outputDes->regions[i];
            reg.origin = tensorArrayInput;
            reg.src.offset = elemSize.first;
            reg.src.stride[0] = 1;
            reg.src.stride[1] = 1;
            reg.src.stride[2] = 1;
            reg.dst.offset = dstOffset;
            reg.dst.stride[0] = 1;
            reg.dst.stride[1] = 1;
            reg.dst.stride[2] = 1;
            reg.size[0] = elemSize.second;
            reg.size[1] = 1;
            reg.size[2] = 1;
            dstOffset += elemSize.second;
        }
        return true;
    }
};

class GeometryTensorArrayScatter : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[3];
        auto inDes = TensorUtils::getDescribe(tensorArrayInput);
        if (inDes->tensorArrayAttr == nullptr) {
            return false;
        }
        int oldSize = inDes->tensorArrayAttr->arraySize;
        auto output    = outputs[0];
        int elemSize = getElemSize(output, 0).second;
        auto indicesTensor = inputs[1];
        // tag index write or not
        std::vector<bool> isWrite(oldSize, false);
        // write index
        std::vector<int> indices(indicesTensor->elementSize());
        // not write index
        std::vector<int> remains;
        for (int i = 0; i < indices.size(); i++) {
            indices[i] = indicesTensor->host<int>()[i];
            if (i < oldSize) {
                isWrite[i] = true;
            }
        }
        for (int i = 0; i < oldSize; i++) {
            if (!isWrite[i]) {
                remains.push_back(i);
            }
        }
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.resize(indices.size() + remains.size());
        // write value by indices
        for (int i = 0; i < indices.size(); i++) {
            MNN_ASSERT(indices[i] < outputDes->tensorArrayAttr->arraySize);
            auto& reg = outputDes->regions[i];
            reg.origin = inputs[2];
            reg.src.offset = i * elemSize;
            reg.src.stride[0] = 1;
            reg.src.stride[1] = 1;
            reg.src.stride[2] = 1;
            reg.dst.offset = indices[i] * elemSize;
            reg.dst.stride[0] = 1;
            reg.dst.stride[1] = 1;
            reg.dst.stride[2] = 1;
            reg.size[0] = elemSize;
            reg.size[1] = 1;
            reg.size[2] = 1;
        }
        if (remains.empty()) {
            return true;
        }
        // first write data, set zero
        bool firstWrite = isFirstWrite(inDes);
        if (firstWrite) {
            auto type = tensorArrayInput->getType();
            auto zeroConst = context.allocConst(op, {}, type);
            if (type == halide_type_of<float>()) {
                zeroConst->host<float>()[0] = 0.0;
            } else {
                zeroConst->host<int>()[0] = 0;
            }
            tensorArrayInput = zeroConst.get();
        }
        // copy not write value by remains
        for (int i = 0; i < remains.size(); i++) {
            auto& reg = outputDes->regions[indices.size() + i];
            reg.origin = tensorArrayInput;
            reg.src.offset = (!firstWrite) * remains[i] * elemSize;
            reg.src.stride[0] = !firstWrite;
            reg.src.stride[1] = 1;
            reg.src.stride[2] = 1;
            reg.dst.offset = remains[i] * elemSize;
            reg.dst.stride[0] = 1;
            reg.dst.stride[1] = 1;
            reg.dst.stride[2] = 1;
            reg.size[0] = elemSize;
            reg.size[1] = 1;
            reg.size[2] = 1;
        }
        return true;
    }
};

class GeometryTensorArraySplit : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto shape = inputs[1]->shape();
        int splitAxis = (op->main_as_TensorArray()->axis() + shape.size()) % shape.size();
        auto outside = std::accumulate(shape.begin(), shape.begin() + splitAxis, 1,
                                      [](int a, int b) { return a * b; });
        auto inside = std::accumulate(shape.begin() + splitAxis + 1, shape.end(), 1,
                                       [](int a, int b) { return a * b; });

        auto value = inputs[1], lengths = inputs[2];
        bool scalarSplit = (lengths->elementSize() == 1);
        int totalLen = value->shape()[splitAxis];
        int splitNum = (scalarSplit ? UP_DIV(totalLen, lengths->host<int>()[0]) : lengths->length(0));
        auto output    = outputs[0];
        auto outDes = TensorUtils::getDescribe(output);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outDes->regions.clear();
        for (int i = 0, splitSum = 0, splitLast = -1; i < splitNum; ++i) {
            int splitLen;
            if (scalarSplit) {
                splitLen = ALIMIN(lengths->host<int>()[0], totalLen - splitSum);
            } else {
                splitLen = lengths->host<int>()[i];
            }
            if (splitLast == splitLen) {
                outDes->regions[outDes->regions.size() - 1].size[0] += 1;
                splitSum += splitLen;
                splitLast = splitLen;
                continue;
            }
            Tensor::InsideDescribe::Region reg;
            reg.origin = value;
            reg.src.offset = inside * splitSum;
            reg.src.stride[0] = inside * splitLen;
            reg.src.stride[1] = inside * shape[splitAxis];
            reg.dst.offset = inside * outside * splitSum;
            reg.dst.stride[0] = inside * outside * splitLen;
            reg.dst.stride[1] = inside * splitLen;
            reg.size[1] = outside;
            reg.size[2] = inside * splitLen;
            outDes->regions.push_back(reg);
            splitSum += splitLen;
            splitLast = splitLen;
        }
        return true;
    }
};

class GeometryTensorArrayConcat : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto output    = outputs[0];
        auto outDes = TensorUtils::getDescribe(output);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outDes->regions.clear();
        auto attr = TensorUtils::getDescribe(inputs[1])->tensorArrayAttr;
        auto tpParam = op->main_as_TensorArray();
        int concatAxis = tpParam->axis(), newAxis = tpParam->new_axis();
        int outputDimensions = output->dimensions();
        concatAxis = (concatAxis + outputDimensions) % outputDimensions;
        int outside = 1;
        int inside = 1;
        for (int i=0; i<concatAxis; ++i) {
            outside *= output->length(i);
        }
        for (int i=concatAxis+1; i<output->dimensions(); ++i) {
            inside *= output->length(i);
        }
        int concatFinal = output->length(concatAxis);
        for (int i = 0, concatSum = 0, concatLast = -1; i < attr->arraySize; ++i) {
            int shapeIndex = i;
            if (attr->isIdenticalShape) {
                shapeIndex = 0;
            }
            int concatLen = 1;
            if (newAxis == 0) {
                concatLen = attr->elemShape[shapeIndex][concatAxis];
            }
            if (concatLast == concatLen) {
                outDes->regions[outDes->regions.size() - 1].size[0] += 1;
                continue;
            }
            Tensor::InsideDescribe::Region reg;
            reg.origin = inputs[1];
            reg.src.offset = inside * outside * concatSum;
            reg.src.stride[0] = outside * inside * concatLen;
            reg.src.stride[1] = inside * concatLen;
            reg.dst.offset = inside * concatSum;
            reg.dst.stride[0] = inside * concatLen;
            reg.dst.stride[1] = inside * concatFinal;
            reg.size[1] = outside;
            reg.size[2] = inside * concatLen;
            outDes->regions.push_back(reg);
            concatSum += concatLen;
            concatLast = concatLen;
        }
        return true;
    }
};

class GeometryTensorArrayErase : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[2];
        auto inDes = TensorUtils::getDescribe(tensorArrayInput);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        auto output    = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        int eraseIndex = inputs[1]->host<int32_t>()[0], oldSize = inDes->tensorArrayAttr->arraySize;
        eraseIndex += (eraseIndex < 0 ? oldSize: 0);
        auto eleSize = getElemSize(tensorArrayInput, eraseIndex);
        outputDes->regions.clear();
        if (eraseIndex > 0) {
            Tensor::InsideDescribe::Region reg;
            reg.origin = tensorArrayInput;
            reg.src.offset = 0;
            reg.src.stride[0] = reg.src.stride[1] = reg.src.stride[2] = 1;
            reg.dst.offset = 0;
            reg.dst.stride[0] = reg.dst.stride[1] = reg.dst.stride[2] = 1;
            reg.size[0] = eleSize.first;
            reg.size[1] = reg.size[2] = 1;
            outputDes->regions.push_back(reg);
        }
        if (eraseIndex < oldSize - 1) {
            int offset = eleSize.first + eleSize.second;
            Tensor::InsideDescribe::Region reg;
            reg.origin = tensorArrayInput;
            reg.src.offset = offset;
            reg.src.stride[0] = reg.src.stride[1] = reg.src.stride[2] = 1;
            reg.dst.offset = eleSize.first;
            reg.dst.stride[0] = reg.dst.stride[1] = reg.dst.stride[2] = 1;
            reg.size[0] = tensorArrayInput->elementSize() - offset;
            reg.size[1] = reg.size[2] = 1;
            outputDes->regions.push_back(reg);
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp0(new GeometryTensorArray);
    GeometryComputer::registerGeometryComputer(comp0, {OpType_TensorArray});
    std::shared_ptr<GeometryComputer> comp1(new GeometryTensorArraySize);
    GeometryComputer::registerGeometryComputer(comp1, {OpType_TensorArraySize});
    std::shared_ptr<GeometryComputer> comp2(new GeometryTensorArrayRead);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_TensorArrayRead});
    std::shared_ptr<GeometryComputer> comp3(new GeometryTensorArrayWrite(false));
    GeometryComputer::registerGeometryComputer(comp3, {OpType_TensorArrayWrite});
    std::shared_ptr<GeometryComputer> comp4(new GeometryTensorArrayGather);
    GeometryComputer::registerGeometryComputer(comp4, {OpType_TensorArrayGather});
    std::shared_ptr<GeometryComputer> comp5(new GeometryTensorArrayScatter);
    GeometryComputer::registerGeometryComputer(comp5, {OpType_TensorArrayScatter});
    std::shared_ptr<GeometryComputer> comp6(new GeometryTensorArraySplit);
    GeometryComputer::registerGeometryComputer(comp6, {OpType_TensorArraySplit});
    std::shared_ptr<GeometryComputer> comp7(new GeometryTensorArrayConcat);
    GeometryComputer::registerGeometryComputer(comp7, {OpType_TensorArrayConcat});
    std::shared_ptr<GeometryComputer> comp8(new GeometryTensorArrayWrite(true));
    GeometryComputer::registerGeometryComputer(comp8, {OpType_TensorArrayInsert});
    std::shared_ptr<GeometryComputer> comp9(new GeometryTensorArrayErase);
    GeometryComputer::registerGeometryComputer(comp9, {OpType_TensorArrayErase});
}

REGISTER_GEOMETRY(GeometryTensorArray, _create);
} // namespace MNN
