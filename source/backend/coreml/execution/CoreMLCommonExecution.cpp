//
//  CoreMLCommonExecution.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLCommonExecution.hpp"
namespace MNN {

CoreMLCommonExecution::CoreMLCommonExecution(Backend* backend, const Op* op) : Execution(backend), mOp(op) {
    mCoreMLBackend = (CoreMLBackend*)backend;
    // Layer names and weight addresses must not change the shared program key.
    mName = "kernel_" + std::to_string(static_cast<int>(op->type()));
}

ErrorCode CoreMLCommonExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode CoreMLCommonExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

CoreMLCommonExecution::Operation::Operation(CoreMLCommonExecution& execution, const std::string& type)
    : mExecution(execution), mBuilder(new MilOperationBuilder(type)) {
    MNN_ASSERT(!type.empty());
}

CoreMLCommonExecution::Operation& CoreMLCommonExecution::Operation::input(const std::string& parameter,
                                                                          const Tensor* tensor) {
    return input(parameter, mExecution.mCoreMLBackend->getTensorName(tensor));
}

CoreMLCommonExecution::Operation& CoreMLCommonExecution::Operation::input(const std::string& parameter,
                                                                          const std::string& value) {
    MNN_ASSERT(mBuilder != nullptr);
    mBuilder->addInput(parameter, value);
    return *this;
}

void CoreMLCommonExecution::Operation::output(const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(mBuilder != nullptr);
    mExecution.addOperation(std::move(mBuilder), outputs);
}

std::string CoreMLCommonExecution::Operation::output(const std::vector<int>& shape, const std::string& name,
                                                     CoreML__Specification__MILSpec__DataType dtype) {
    MNN_ASSERT(mBuilder != nullptr);
    return mExecution.addOperation(std::move(mBuilder), shape, name, dtype);
}

CoreMLCommonExecution::Operation& CoreMLCommonExecution::Operation::inputs(const std::string& parameter,
                                                                           const std::vector<std::string>& values) {
    MNN_ASSERT(mBuilder != nullptr);
    mBuilder->addInputs(parameter, values);
    return *this;
}

CoreMLCommonExecution::Operation& CoreMLCommonExecution::Operation::inputs(const std::string& parameter,
                                                                           const std::vector<Tensor*>& tensors) {
    std::vector<std::string> names;
    names.reserve(tensors.size());
    for (auto tensor : tensors) {
        names.push_back(mExecution.mCoreMLBackend->getTensorName(tensor));
    }
    return inputs(parameter, names);
}

CoreMLCommonExecution::Operation& CoreMLCommonExecution::Operation::shape(const std::string& parameter,
                                                                          const std::vector<int>& values,
                                                                          const std::string& name) {
    return input(parameter, mExecution.buildShape(values, name.empty() ? parameter : name));
}

CoreMLCommonExecution::Operation&
CoreMLCommonExecution::Operation::scalarFloat32(const std::string& parameter, float value, const std::string& name) {
    return input(parameter, mExecution.mCoreMLBackend->buildScalarFloat32(
                                value, mExecution.mName + "_" + (name.empty() ? parameter : name)));
}

void CoreMLCommonExecution::Operation::finish(const std::string& name) {
    MNN_ASSERT(mBuilder != nullptr && !name.empty());
    mBuilder->setName(mExecution.mName + "_" + name);
    mExecution.mCoreMLBackend->addOperation(std::move(mBuilder));
}
void CoreMLCommonExecution::addOperation(std::unique_ptr<MilOperationBuilder> builder,
                                         const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(!builder->getType().empty());
    builder->setName(mName);
    for (auto output : outputs) {
        const auto& name = mCoreMLBackend->getTensorName(output);
        const auto shape = mCoreMLBackend->tensorShape(output);
        const auto& dtype = output->getType();
        std::vector<int64_t> mil_shape(shape.begin(), shape.end());
        auto mil_dtype = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
        if (dtype == halide_type_of<int>()) {
            mil_dtype = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32;
        }
        builder->addOutput(name, mil_dtype, mil_shape);
    }
    mCoreMLBackend->addOperation(std::move(builder));
}

std::string CoreMLCommonExecution::addOperation(std::unique_ptr<MilOperationBuilder> builder,
                                                const std::vector<int>& shape, const std::string& name,
                                                CoreML__Specification__MILSpec__DataType dtype) {
    auto opName = mName + "_" + builder->getType();
    if (!name.empty()) {
        opName += "_" + name;
    }
    builder->setName(opName);
    std::vector<int64_t> mil_shape(shape.begin(), shape.end());
    builder->addOutput(builder->getName(), dtype, mil_shape);
    mCoreMLBackend->addOperation(std::move(builder));
    return opName;
}

}; // namespace MNN
