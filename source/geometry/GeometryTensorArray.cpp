//
//  GeometryTensorArray.cpp
//  MNN
//
//  Created by MNN on 2020/12/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
// get a pair <ElemOffset, ElemSize>
static std::pair<int, int> getElemSize(const Tensor* t, int index) {
    auto des = TensorUtils::getDescribe(t);
    auto shapes = des->tensorArrayAttr->elemShape;
    int elemSize = 1;
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

static bool isFirstWrite(const Tensor::InsideDescribe* des) {
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
        auto elemSize = getElemSize(output, writeIndex);
        int regionSize = (writeIndex > 0) + 1 + (oldSize - writeIndex - 1 > 0);
        outDes->regions.resize(regionSize);
        /*
         src: [leftData][writeIndex][rightData]
         dst: [leftData][writeTensor][rightData]
         */
        // 1. write Tensor to dst TensorArray [must]
        auto& writeTensorRegion = outDes->regions[0];
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
        if (writeIndex > 0) {
            auto& leftDataRegion = outDes->regions[1];
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
        }
        // 3. copy TensorArray rightData [optional]
        int rightSize = oldSize - writeIndex - 1;
        if (rightSize > 0) {
            auto last = getElemSize(output, oldSize-1);
            int totalSize = last.first + last.second;
            int offset = elemSize.first + elemSize.second;
            auto& rightDataRegion = outDes->regions[1 + (writeIndex > 0)];
            rightDataRegion.origin = tensorArrayInput;
            rightDataRegion.src.offset = (!firstWrite) * offset;
            rightDataRegion.src.stride[0] = !firstWrite;
            rightDataRegion.src.stride[1] = 1;
            rightDataRegion.src.stride[2] = 1;
            rightDataRegion.dst.offset = offset;
            rightDataRegion.dst.stride[0] = 1;
            rightDataRegion.dst.stride[1] = 1;
            rightDataRegion.dst.stride[2] = 1;
            rightDataRegion.size[0] = totalSize - offset;
            rightDataRegion.size[1] = 1;
            rightDataRegion.size[2] = 1;
        }
        return true;
    }
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
        auto output    = outputs[0];
        auto outDes = TensorUtils::getDescribe(output);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outDes->regions.resize(1);
        auto& reg = outDes->regions[0];
        reg.origin = inputs[1];
        reg.src.offset = 0;
        reg.src.stride[0] = 1;
        reg.src.stride[1] = 1;
        reg.src.stride[2] = 1;
        reg.dst.offset = 0;
        reg.dst.stride[0] = 1;
        reg.dst.stride[1] = 1;
        reg.dst.stride[2] = 1;
        reg.size[0] = inputs[1]->elementSize();
        reg.size[1] = 1;
        reg.size[2] = 1;
        return true;
    }
};

class GeometryTensorArrayConcat : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto tensorArrayInput = inputs[1];
        auto inDes = TensorUtils::getDescribe(tensorArrayInput);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        auto output    = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.resize(1);
        auto& reg = outputDes->regions[0];
        reg.origin = tensorArrayInput;
        reg.src.offset = 0;
        reg.src.stride[0] = 1;
        reg.src.stride[1] = 1;
        reg.src.stride[2] = 1;
        reg.dst.offset = 0;
        reg.dst.stride[0] = 1;
        reg.dst.stride[1] = 1;
        reg.dst.stride[2] = 1;
        reg.size[0] = tensorArrayInput->elementSize();
        reg.size[1] = 1;
        reg.size[2] = 1;
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
    std::shared_ptr<GeometryComputer> comp3(new GeometryTensorArrayWrite);
    GeometryComputer::registerGeometryComputer(comp3, {OpType_TensorArrayWrite});
    std::shared_ptr<GeometryComputer> comp4(new GeometryTensorArrayGather);
    GeometryComputer::registerGeometryComputer(comp4, {OpType_TensorArrayGather});
    std::shared_ptr<GeometryComputer> comp5(new GeometryTensorArrayScatter);
    GeometryComputer::registerGeometryComputer(comp5, {OpType_TensorArrayScatter});
    std::shared_ptr<GeometryComputer> comp6(new GeometryTensorArraySplit);
    GeometryComputer::registerGeometryComputer(comp6, {OpType_TensorArraySplit});
    std::shared_ptr<GeometryComputer> comp7(new GeometryTensorArrayConcat);
    GeometryComputer::registerGeometryComputer(comp7, {OpType_TensorArrayConcat});
}

REGISTER_GEOMETRY(GeometryTensorArray, _create);
} // namespace MNN
