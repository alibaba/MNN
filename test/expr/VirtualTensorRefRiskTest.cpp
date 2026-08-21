//
//  VirtualTensorRefRiskTest.cpp
//  MNNTests
//

#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/Module.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static std::shared_ptr<Module> loadVirtualRefModule(const std::vector<int8_t>& buffer) {
    Module::Config config;
    config.rearrange = false;
    config.shapeMutable = true;
    return std::shared_ptr<Module>(
        Module::load({"input"}, {"output"}, reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(), &config),
        Module::destroy);
}

static VARP makeVirtualRefInput(int round = 0) {
    auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
    auto inputPtr = input->writeMap<float>();
    if (inputPtr == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < input->getInfo()->size; ++i) {
        inputPtr[i] = static_cast<float>((i + round * 2) % 7 - 3);
    }
    input->unMap();
    return input;
}

class VirtualTensorRefInputRebindTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Square(_Split(input, {64, 64}, 1)[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        for (int round = 0; round < 3; ++round) {
            auto outputs = module->onForward({makeVirtualRefInput(round)});
            if (outputs.size() != 1 || outputs[0] == nullptr) {
                return false;
            }
            auto outputPtr = outputs[0]->readMap<float>();
            if (outputPtr == nullptr) {
                return false;
            }
            for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
                const auto value = static_cast<float>((i + 64 * 16 + round * 2) % 7 - 3);
                const auto expected = value * value;
                if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                    MNN_ERROR(
                        "VirtualTensorRefInputRebindTest error at "
                        "round=%d index=%d: %f != %f\n",
                        round, i, outputPtr[i], expected);
                    return false;
                }
            }
            outputs[0]->unMap();
        }
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefInputRebindTest, "expr/VirtualTensorRefInputRebind");

class VirtualTensorRefDirectOutputTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto origin = _Square(input);
            auto output = _Split(origin, {64, 64}, 1)[0];
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        auto outputs = module->onForward({makeVirtualRefInput()});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            MNN_ERROR("VirtualTensorRefDirectOutputTest can't map output\n");
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto value = static_cast<float>(i % 7 - 3);
            const auto expected = value * value;
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefDirectOutputTest error at index=%d: "
                    "%f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefDirectOutputTest, "expr/VirtualTensorRefDirectOutput");

class VirtualTensorRefInputDirectOutputTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Split(input, {64, 64}, 1)[1];
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        auto outputs0 = module->onForward({makeVirtualRefInput(0)});
        auto outputs1 = module->onForward({makeVirtualRefInput(1)});
        if (outputs0.size() != 1 || outputs1.size() != 1) {
            return false;
        }
        for (int round = 0; round < 2; ++round) {
            auto output = round == 0 ? outputs0[0] : outputs1[0];
            auto outputPtr = output->readMap<float>();
            if (outputPtr == nullptr) {
                return false;
            }
            for (int i = 0; i < output->getInfo()->size; ++i) {
                const auto expected = static_cast<float>((i + 64 * 16 + round * 2) % 7 - 3);
                if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                    MNN_ERROR(
                        "VirtualTensorRefInputDirectOutputTest error at "
                        "round=%d index=%d: %f != %f\n",
                        round, i, outputPtr[i], expected);
                    return false;
                }
            }
            output->unMap();
        }
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefInputDirectOutputTest, "expr/VirtualTensorRefInputDirectOutput");

class VirtualTensorRefNestedSliceTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto outer = _Split(_Square(input), {64, 64}, 1);
            auto inner = _Split(outer[1], {32, 32}, 1);
            auto output = _Abs(inner[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        auto outputs = module->onForward({makeVirtualRefInput()});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto value = static_cast<float>((i + 96 * 16) % 7 - 3);
            const auto expected = value * value;
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefNestedSliceTest error at index=%d: "
                    "%f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefNestedSliceTest, "expr/VirtualTensorRefNestedSlice");

class VirtualTensorRefSiblingLifetimeTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto slices = _Split(_Square(input), {64, 64}, 1);
            auto output = _Concat({_Abs(slices[0]), _Negative(slices[1])}, 1);
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        auto outputs = module->onForward({makeVirtualRefInput()});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto value = static_cast<float>(i % 7 - 3);
            const auto squared = value * value;
            const auto expected = i < 64 * 16 ? squared : -squared;
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefSiblingLifetimeTest error at index=%d: "
                    "%f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefSiblingLifetimeTest, "expr/VirtualTensorRefSiblingLifetime");

class VirtualTensorRefContiguousTransitionTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 64, 1, 16}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Square(_Transpose(input, {0, 1, 3, 2}));
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        const std::vector<std::pair<int, int>> shapes = {{1, 16}, {4, 4}, {2, 8}, {1, 16}};
        for (const auto& shape : shapes) {
            const auto height = shape.first;
            const auto width = shape.second;
            auto input = _Input({1, 64, height, width}, NC4HW4, halide_type_of<float>());
            auto inputPtr = input->writeMap<float>();
            if (inputPtr == nullptr) {
                return false;
            }
            for (int i = 0; i < input->getInfo()->size; ++i) {
                inputPtr[i] = static_cast<float>(i % 11 - 5);
            }
            input->unMap();

            auto outputs = module->onForward({input});
            if (outputs.size() != 1 || outputs[0] == nullptr ||
                outputs[0]->getInfo()->dim != std::vector<int>({1, 64, width, height})) {
                return false;
            }
            auto outputPtr = outputs[0]->readMap<float>();
            if (outputPtr == nullptr) {
                return false;
            }
            for (int c = 0; c < 64; ++c) {
                for (int outputY = 0; outputY < width; ++outputY) {
                    for (int outputX = 0; outputX < height; ++outputX) {
                        const auto outputIndex = ((c / 4) * width * height + outputY * height + outputX) * 4 + c % 4;
                        const auto inputIndex = ((c / 4) * height * width + outputX * width + outputY) * 4 + c % 4;
                        const auto value = static_cast<float>(inputIndex % 11 - 5);
                        const auto expected = value * value;
                        if (std::fabs(outputPtr[outputIndex] - expected) > 0.001f) {
                            MNN_ERROR(
                                "VirtualTensorRefContiguousTransitionTest "
                                "error at shape=%dx%d index=%d: %f != "
                                "%f\n",
                                height, width, outputIndex, outputPtr[outputIndex], expected);
                            return false;
                        }
                    }
                }
            }
            outputs[0]->unMap();
        }
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefContiguousTransitionTest, "expr/VirtualTensorRefContiguousTransition");

class VirtualTensorRefSameTypePrecisionTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (precision != BackendConfig::Precision_High) {
            return true;
        }
        auto runtime = ExecutorScope::Current()->getRuntime();
        if (runtime.first.find(MNN_FORWARD_METAL) == runtime.first.end()) {
            return true;
        }

        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Square(_Split(input, {64, 64}, 1)[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }

        auto hostInput = makeVirtualRefInput();
        auto normalPrecisionMetalInput = _Square(hostInput);
        auto normalPrecisionPtr = normalPrecisionMetalInput->readMap<float>();
        if (normalPrecisionPtr == nullptr) {
            return false;
        }
        normalPrecisionMetalInput->unMap();

        BackendConfig lowPrecisionConfig;
        lowPrecisionConfig.precision = BackendConfig::Precision_Low;
        auto lowPrecisionExecutor = Executor::newExecutor(MNN_FORWARD_METAL, lowPrecisionConfig, 1);
        ExecutorScope lowPrecisionScope(lowPrecisionExecutor);
        Module::Config moduleConfig;
        moduleConfig.rearrange = false;
        moduleConfig.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    &moduleConfig),
                                       Module::destroy);
        if (module == nullptr) {
            return false;
        }

        auto outputs = module->onForward({normalPrecisionMetalInput});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto inputValue = static_cast<float>((i + 64 * 16) % 7 - 3);
            const auto squared = inputValue * inputValue;
            const auto expected = squared * squared;
            if (std::fabs(outputPtr[i] - expected) > 0.01f) {
                MNN_ERROR(
                    "VirtualTensorRefSameTypePrecisionTest error at index=%d: "
                    "%f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefSameTypePrecisionTest, "expr/VirtualTensorRefSameTypePrecision");

class VirtualTensorRefCrossCPUPrecisionTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (precision != BackendConfig::Precision_High || getCurrentType() != MNN_FORWARD_CPU) {
            return true;
        }

        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Square(_Split(input, {64, 64}, 1)[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }

        auto highInput = _Square(makeVirtualRefInput());
        auto highInputPtr = highInput->readMap<float>();
        if (highInputPtr == nullptr) {
            return false;
        }
        highInput->unMap();

        BackendConfig lowPrecisionConfig;
        lowPrecisionConfig.precision = BackendConfig::Precision_Low;
        auto lowPrecisionExecutor = Executor::newExecutor(MNN_FORWARD_CPU, lowPrecisionConfig, 1);
        ExecutorScope lowPrecisionScope(lowPrecisionExecutor);
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        auto outputs = module->onForward({highInput});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto inputValue = static_cast<float>((i + 64 * 16) % 7 - 3);
            const auto squared = inputValue * inputValue;
            const auto expected = squared * squared;
            if (std::fabs(outputPtr[i] - expected) > 0.01f) {
                MNN_ERROR(
                    "VirtualTensorRefCrossCPUPrecisionTest error at index=%d: "
                    "%f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefCrossCPUPrecisionTest, "expr/VirtualTensorRefCrossCPUPrecision");

class VirtualTensorRefDynamicOutputTest : public MNNTestCase {
public:
    bool run(int precision) override { return runCase(Executor::LAZY_CONTENT); }

private:
    static bool runCase(uint32_t lazyMode) {
        auto executor = ExecutorScope::Current();
        const auto oldLazyMode = executor->getLazyMode();
        executor->setLazyComputeMode(lazyMode);
        auto input = makeVirtualRefInput();
        auto square = _Square(input);
        auto slices = _Split(square, {64, 64}, 1);
        auto output = slices[1];
        auto outputPtr = output->readMap<float>();
        executor->setLazyComputeMode(oldLazyMode);
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < output->getInfo()->size; ++i) {
            const auto value = static_cast<float>((i + 64 * 16) % 7 - 3);
            const auto expected = value * value;
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefDynamicOutputTest error at mode=%u "
                    "index=%d: %f != %f\n",
                    lazyMode, i, outputPtr[i], expected);
                return false;
            }
        }
        output->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefDynamicOutputTest, "expr/VirtualTensorRefDynamicOutput");

class VirtualTensorRefDynamicInputOutputTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto executor = ExecutorScope::Current();
        const auto oldLazyMode = executor->getLazyMode();
        executor->setLazyComputeMode(Executor::LAZY_CONTENT);
        auto input = makeVirtualRefInput();
        auto slices = _Split(input, {64, 64}, 1);
        auto output = slices[1];
        auto outputPtr = output->readMap<float>();
        executor->setLazyComputeMode(oldLazyMode);
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < output->getInfo()->size; ++i) {
            const auto expected = static_cast<float>((i + 64 * 16) % 7 - 3);
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefDynamicInputOutputTest error at index=%d: "
                    "%f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        output->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefDynamicInputOutputTest, "expr/VirtualTensorRefDynamicInputOutput");

class VirtualTensorRefEagerAllocatorTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Abs(_Split(_Square(input), {64, 64}, 1)[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }

        ScheduleConfig scheduleConfig;
        scheduleConfig.type = getCurrentType();
        scheduleConfig.numThread = 1;
        std::shared_ptr<Executor::RuntimeManager> runtimeManager(
            Executor::RuntimeManager::createRuntimeManager(scheduleConfig), Executor::RuntimeManager::destroy);
        runtimeManager->setHint(Interpreter::MEM_ALLOCATOR_TYPE, Runtime::Allocator_Eager);
        Module::Config config;
        config.rearrange = false;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    runtimeManager, &config),
                                       Module::destroy);
        if (module == nullptr) {
            return false;
        }

        auto outputs = module->onForward({makeVirtualRefInput()});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto value = static_cast<float>((i + 64 * 16) % 7 - 3);
            const auto expected = value * value;
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefEagerAllocatorTest error at index=%d: %f "
                    "!= %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefEagerAllocatorTest, "expr/VirtualTensorRefEagerAllocator");

class VirtualTensorRefFallbackOriginTest : public MNNTestCase {
public:
    bool run(int precision) override {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({}, NCHW, halide_type_of<float>());
            input->setName("input");
            auto linspace = _LinSpace(input, _Scalar<float>(1.0f), _Scalar<int32_t>(2048));
            auto reshaped = _Reshape(linspace, {1, 2048, 1, 1}, NCHW);
            auto packed = _Convert(reshaped, NC4HW4);
            auto slice = _Split(packed, {1024, 1024}, 1)[1];
            auto output = _Square(slice);
            output->setName("output");
            buffer = Variable::save({output});
        }
        int rasterCount = 0;
        auto executor = ExecutorScope::Current();
        executor->setCallBack(TensorCallBackWithInfo([&](const std::vector<Tensor*>&, const OperatorInfo* info) {
                                  if (info != nullptr && info->type() == "Raster") {
                                      ++rasterCount;
                                  }
                                  return true;
                              }),
                              [](const std::vector<Tensor*>&, const OperatorInfo*) { return true; });
        ScheduleConfig scheduleConfig;
        scheduleConfig.type = getCurrentType();
        std::shared_ptr<Executor::RuntimeManager> runtimeManager(
            Executor::RuntimeManager::createRuntimeManager(scheduleConfig), Executor::RuntimeManager::destroy);
        runtimeManager->setMode(Interpreter::Session_Debug);
        Module::Config config;
        config.rearrange = false;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    runtimeManager, &config),
                                       Module::destroy);
        if (module == nullptr) {
            executor->setCallBack(nullptr, nullptr);
            return false;
        }

        auto input = _Input({}, NCHW, halide_type_of<float>());
        auto inputPtr = input->writeMap<float>();
        if (inputPtr == nullptr) {
            return false;
        }
        inputPtr[0] = 0.0f;
        input->unMap();

        auto outputs = module->onForward({input});
        executor->setCallBack(nullptr, nullptr);
        MNN_PRINT("VirtualTensorRefFallbackOriginTest Raster count: %d\n", rasterCount);
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto value = static_cast<float>(i + 1024) / 2047.0f;
            const auto expected = value * value;
            if (std::fabs(outputPtr[i] - expected) > 0.01f) {
                MNN_ERROR(
                    "VirtualTensorRefFallbackOriginTest error at index=%d: %f "
                    "!= %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefFallbackOriginTest, "expr/VirtualTensorRefFallbackOrigin");

class VirtualTensorRefRandomFallbackOriginTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (getCurrentType() == MNN_FORWARD_CPU) {
            return true;
        }
        std::vector<int8_t> buffer;
        {
            auto shape = _Input({4}, NCHW, halide_type_of<int32_t>());
            shape->setName("shape");
            auto random = _RandomUnifom(shape, halide_type_of<float>(), 1.0f, 2.0f, 19, 23);
            random = _Convert(random, NC4HW4);
            auto slice = _Split(random, {64, 64}, 1)[1];
            auto output = _Square(slice);
            output->setName("output");
            buffer = Variable::save({output});
        }
        Module::Config config;
        config.rearrange = false;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"shape"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    &config),
                                       Module::destroy);
        if (module == nullptr) {
            return false;
        }

        auto shape = _Input({4}, NCHW, halide_type_of<int32_t>());
        auto shapePtr = shape->writeMap<int32_t>();
        if (shapePtr == nullptr) {
            return false;
        }
        shapePtr[0] = 1;
        shapePtr[1] = 128;
        shapePtr[2] = 1;
        shapePtr[3] = 1;
        shape->unMap();
        auto outputs = module->onForward({shape});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            if (!(outputPtr[i] >= 1.0f && outputPtr[i] < 4.0f)) {
                MNN_ERROR(
                    "VirtualTensorRefRandomFallbackOriginTest error at index=%d: "
                    "%f is outside [1, 4)\n",
                    i, outputPtr[i]);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefRandomFallbackOriginTest, "expr/VirtualTensorRefRandomFallbackOrigin");

class VirtualTensorRefDequantFallbackOriginTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (getCurrentType() == MNN_FORWARD_CPU) {
            return true;
        }
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<uint8_t>());
            input->setName("input");
            std::unique_ptr<OpT> op(new OpT);
            op->type = OpType_Dequantize;
            op->main.type = OpParameter_Dequantize;
            auto param = new DequantizeT;
            param->inputQuantizedParam.reset(new QuantizedParamT);
            param->inputQuantizedParam->zeroPoint = 0;
            param->inputQuantizedParam->scale = 1.0f;
            param->mode = QuantizeMode_MIN_COMBINED;
            param->modelFormat = ModeFormat_TFLITE;
            param->type = DataType_DT_QUINT8;
            op->main.value = param;
            auto dequantized = Variable::create(Expr::create(std::move(op), {input}));
            auto slice = _Split(dequantized, {64, 64}, 1)[1];
            auto output = _Square(slice);
            output->setName("output");
            buffer = Variable::save({output});
        }
        auto module = loadVirtualRefModule(buffer);
        if (module == nullptr) {
            return false;
        }

        auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<uint8_t>());
        auto inputPtr = input->writeMap<uint8_t>();
        if (inputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < input->getInfo()->size; ++i) {
            inputPtr[i] = static_cast<uint8_t>(i % 7);
        }
        input->unMap();
        auto outputs = module->onForward({input});
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
            const auto value = static_cast<float>((i + 64 * 16) % 7);
            const auto expected = value * value;
            if (std::fabs(outputPtr[i] - expected) > 0.01f) {
                MNN_ERROR(
                    "VirtualTensorRefDequantFallbackOriginTest error at "
                    "index=%d: %f != %f\n",
                    i, outputPtr[i], expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefDequantFallbackOriginTest, "expr/VirtualTensorRefDequantFallbackOrigin");
