//
//  VirtualTensorRefResizeTest.cpp
//  MNNTests
//

#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

class VirtualTensorRefResizeTest : public MNNTestCase {
public:
    bool run(int precision) override { return runCase(false) && runCase(true); }

private:
    static bool runCase(bool directInputSlice) {
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 48, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto origin = directInputSlice ? input : _Square(input);
            auto slices = _Split(origin, {64, 64}, 1);
            // Use direct backend ops so GeometryComputer caches their Commands.
            // makeRaster must not replace the cached inputs with temporary refs, and multiple
            // consumers of the same slice must reuse the in-place virtual ref.
            auto output0 = _Square(slices[0]);
            auto output1 = _Square(slices[1]);
            auto output2 = _Abs(slices[0]);
            output0->setName("output0");
            output1->setName("output1");
            output2->setName("output2");
            buffer = Variable::save({output0, output1, output2});
        }

        Module::Config config;
        config.rearrange = false;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output0", "output1", "output2"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    &config),
                                       Module::destroy);
        if (module == nullptr) {
            return false;
        }

        for (int height : {48, 16, 32}) {
            auto input = _Input({1, 128, height, 1}, NC4HW4, halide_type_of<float>());
            auto inputPtr = input->writeMap<float>();
            if (inputPtr == nullptr) {
                return false;
            }
            for (int i = 0; i < input->getInfo()->size; ++i) {
                inputPtr[i] = static_cast<float>(i % 5 - 2);
            }
            input->unMap();

            auto outputs = module->onForward({input});
            if (outputs.size() != 3 || !checkOutput(outputs[0], height, directInputSlice, 0, true) ||
                !checkOutput(outputs[1], height, directInputSlice, 1, true) ||
                !checkOutput(outputs[2], height, directInputSlice, 0, false)) {
                return false;
            }
        }
        return true;
    }

    static bool checkOutput(VARP output, int height, bool directInputSlice, int sliceIndex, bool squareConsumer) {
        if (output == nullptr || output->getInfo() == nullptr ||
            output->getInfo()->dim != std::vector<int>({1, 64, height, 1})) {
            return false;
        }
        auto outputPtr = output->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < output->getInfo()->size; ++i) {
            const auto inputIndex = i + sliceIndex * 64 * height;
            const auto inputValue = static_cast<float>(inputIndex % 5 - 2);
            const auto originValue = directInputSlice ? inputValue : inputValue * inputValue;
            const auto expected = squareConsumer ? originValue * originValue : std::fabs(originValue);
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR("VirtualTensorRefResizeTest error at height=%d index=%d: %f != %f\n", height, i, outputPtr[i],
                          expected);
                return false;
            }
        }
        output->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefResizeTest, "expr/VirtualTensorRefResize");

class VirtualTensorRefExecutionFallbackTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto runtime = ExecutorScope::Current()->getRuntime();
        if (runtime.first.find(MNN_FORWARD_METAL) == runtime.first.end()) {
            return true;
        }
        auto executor = cloneCurrentExecutor();
        ExecutorScope scope(executor);

        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto origin = _Square(input);
            auto slices = _Split(origin, {64, 64}, 1);
            auto output = _Histogram(slices[0], 4, 0, 4);
            output->setName("output");
            buffer = Variable::save({output});
        }

        Module::Config config;
        config.rearrange = true;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    &config),
                                       Module::destroy);
        if (module == nullptr) {
            return false;
        }

        auto hostInput = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
        auto inputPtr = hostInput->writeMap<float>();
        if (inputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < hostInput->getInfo()->size; ++i) {
            inputPtr[i] = static_cast<float>(i % 4);
        }
        hostInput->unMap();
        auto outputs = module->onForward({hostInput});
        const float expected[] = {256.0f, 256.0f, 0.0f, 256.0f};
        if (outputs.size() != 1 || outputs[0] == nullptr) {
            MNN_ERROR("virtual tensor ref backend fallback result is incorrect\n");
            return false;
        }
        auto outputPtr = outputs[0]->readMap<float>();
        if (outputPtr == nullptr || !checkVector<float>(outputPtr, expected, 4, 0.001f)) {
            MNN_ERROR("virtual tensor ref backend fallback result is incorrect\n");
            return false;
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefExecutionFallbackTest, "expr/VirtualTensorRefExecutionFallback");

class VirtualTensorRefLazyContentTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto executor = ExecutorScope::Current();
        const auto oldLazyMode = executor->getLazyMode();
        executor->setLazyComputeMode(Executor::LAZY_CONTENT);
        const bool result = runCase(false) && runCase(true) && runMultiOutputCase();
        executor->setLazyComputeMode(oldLazyMode);
        return result;
    }

private:
    static bool runCase(bool directInputSlice) {
        auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
        auto inputPtr = input->writeMap<float>();
        if (inputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < input->getInfo()->size; ++i) {
            inputPtr[i] = static_cast<float>(i % 7 - 3);
        }
        input->unMap();

        auto origin = directInputSlice ? input : _Square(input);
        auto output = _Split(origin, {64, 64}, 1)[1];
        output = _Abs(output);
        auto outputPtr = output->readMap<float>();
        if (outputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < output->getInfo()->size; ++i) {
            const auto inputValue = static_cast<float>((i + 64 * 16) % 7 - 3);
            const auto expected = directInputSlice ? std::fabs(inputValue) : inputValue * inputValue;
            if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                MNN_ERROR(
                    "VirtualTensorRefLazyContentTest error at "
                    "directInputSlice=%d index=%d: %f != %f\n",
                    directInputSlice, i, outputPtr[i], expected);
                return false;
            }
        }
        output->unMap();
        return true;
    }

    static bool runMultiOutputCase() {
        auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
        auto inputPtr = input->writeMap<float>();
        if (inputPtr == nullptr) {
            return false;
        }
        for (int i = 0; i < input->getInfo()->size; ++i) {
            inputPtr[i] = static_cast<float>(i % 7 - 3);
        }
        input->unMap();

        auto outputs = _Split(_Square(input), {64, 64}, 1);
        for (int outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
            auto outputPtr = outputs[outputIndex]->readMap<float>();
            if (outputPtr == nullptr) {
                return false;
            }
            for (int i = 0; i < outputs[outputIndex]->getInfo()->size; ++i) {
                const auto inputValue = static_cast<float>((i + outputIndex * 64 * 16) % 7 - 3);
                const auto expected = inputValue * inputValue;
                if (std::fabs(outputPtr[i] - expected) > 0.001f) {
                    MNN_ERROR(
                        "VirtualTensorRefLazyContentTest multi-output error at "
                        "output=%d index=%d: %f != %f\n",
                        outputIndex, i, outputPtr[i], expected);
                    return false;
                }
            }
            outputs[outputIndex]->unMap();
        }
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefLazyContentTest, "expr/VirtualTensorRefLazyContent");

class VirtualTensorRefGeometryMaskTest : public MNNTestCase {
public:
    bool run(int precision) override {
        const auto type = getCurrentType();
        if (type != MNN_FORWARD_CPU && type != MNN_FORWARD_METAL && type != MNN_FORWARD_VULKAN) {
            return true;
        }
        std::vector<int8_t> buffer;
        {
            auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
            input->setName("input");
            auto output = _Abs(_Split(_Square(input), {64, 64}, 1)[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }

        const int allMask = Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_ALL;
        const int rasterWithRef = countRaster(buffer, allMask);
        const int rasterWithoutRef =
            countRaster(buffer, allMask & ~Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_VIRTUAL_TENSOR_REF);
        if (rasterWithRef < 0 || rasterWithoutRef <= rasterWithRef) {
            MNN_ERROR("VirtualTensorRefGeometryMaskTest Raster count: %d -> %d\n", rasterWithRef, rasterWithoutRef);
            return false;
        }
        return true;
    }

private:
    static VARP makeInput() {
        auto input = _Input({1, 128, 16, 1}, NC4HW4, halide_type_of<float>());
        auto inputPtr = input->writeMap<float>();
        if (inputPtr == nullptr) {
            return nullptr;
        }
        for (int i = 0; i < input->getInfo()->size; ++i) {
            inputPtr[i] = static_cast<float>(i % 5 - 2);
        }
        input->unMap();
        return input;
    }

    static int countRaster(const std::vector<int8_t>& buffer, int geometryMask) {
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
        scheduleConfig.numThread = 1;
        std::shared_ptr<Executor::RuntimeManager> runtimeManager(
            Executor::RuntimeManager::createRuntimeManager(scheduleConfig), Executor::RuntimeManager::destroy);
        runtimeManager->setMode(Interpreter::Session_Debug);
        runtimeManager->setHint(Interpreter::HintMode::GEOMETRY_COMPUTE_MASK, geometryMask);

        Module::Config config;
        config.rearrange = false;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    runtimeManager, &config),
                                       Module::destroy);
        auto input = makeInput();
        auto result = module == nullptr || input == nullptr ? std::vector<VARP>{} : module->onForward({input});
        if (result.size() != 1 || result[0] == nullptr) {
            executor->setCallBack(nullptr, nullptr);
            return -1;
        }
        auto outputPtr = result[0]->readMap<float>();
        if (outputPtr == nullptr) {
            executor->setCallBack(nullptr, nullptr);
            return -1;
        }
        for (int i = 0; i < result[0]->getInfo()->size; ++i) {
            const auto inputValue = static_cast<float>((i + 64 * 16) % 5 - 2);
            if (std::fabs(outputPtr[i] - inputValue * inputValue) > 0.001f) {
                result[0]->unMap();
                executor->setCallBack(nullptr, nullptr);
                return -1;
            }
        }
        result[0]->unMap();
        executor->setCallBack(nullptr, nullptr);
        return rasterCount;
    }
};

MNNTestSuiteRegister(VirtualTensorRefGeometryMaskTest, "expr/VirtualTensorRefGeometryMask");

class VirtualTensorRefDequantFallbackTest : public MNNTestCase {
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
            auto output = _Square(_Split(dequantized, {64, 64}, 1)[1]);
            output->setName("output");
            buffer = Variable::save({output});
        }

        Module::Config config;
        config.rearrange = false;
        config.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"input"}, {"output"},
                                                    reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                                    &config),
                                       Module::destroy);
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
                MNN_ERROR("VirtualTensorRefDequantFallbackTest error at index=%d: %f != %f\n", i, outputPtr[i],
                          expected);
                return false;
            }
        }
        outputs[0]->unMap();
        return true;
    }
};

MNNTestSuiteRegister(VirtualTensorRefDequantFallbackTest, "expr/VirtualTensorRefDequantFallback");
