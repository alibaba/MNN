#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/TensorUtils.hpp"

using namespace MNN;
using namespace MNN::Express;

static VARPS makeComplexGraph(VARP x) {
    // Input: NCHW float, shape {1, 4, 32, 32}
    // Graph intent:
    // - Introduce multiple ops (convert/conv/pool/concat/transpose) to increase
    //   intermediate allocations.
    // - Keep an early large tensor as an output (Aux) to make later output tensor
    //   more likely to be allocated with non-zero offset on Metal.

    auto x4 = _Convert(x, NC4HW4);

    auto c0 = _Conv(0.01f, 0.0f, x4, {4, 8}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    c0      = _Relu(c0);

    auto maxP = _MaxPool(c0, {2, 2}, {2, 2}, VALID);
    auto aveP = _AvePool(c0, {2, 2}, {2, 2}, VALID);

    auto aux = _Concat({maxP, aveP}, 1);
    aux->setName("Aux");

    auto c1 = _Conv(0.02f, 0.01f, aux, {16, 4}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    c1      = _Relu6(c1);

    auto y = _Convert(c1, NCHW);
    y      = _Transpose(y, {0, 2, 3, 1});
    y      = _Transpose(y, {0, 3, 1, 2});
    y      = y + _Scalar<float>(1.0f);
    y = _ReduceSum(y, {2}, true);
    y->setName("Output");

    auto s = _Shape(y);
    s->setName("Shape");

    return {aux, y, s};
}

static VARP makeInput(float base) {
    auto x   = _Input({1, 4, 32, 32}, NCHW, halide_type_of<float>());
    auto ptr = x->writeMap<float>();
    for (int i = 0; i < x->getInfo()->size; ++i) {
        ptr[i] = base + (float)i * 0.001f;
    }
    x->unMap();
    return x;
}

static bool isMetalRuntime() {
    auto rtInfo = Express::ExecutorScope::Current()->getRuntime();
    if (rtInfo.first.empty()) {
        return false;
    }
    return rtInfo.first.begin()->first == MNN_FORWARD_METAL;
}

class StaticModuleOutputReuseTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto executor = cloneCurrentExecutor();
        ExecutorScope scope(executor);

        std::vector<int8_t> buffer;
        {
            auto x = _Input({1, 4, 32, 32}, NCHW, halide_type_of<float>());
            x->setName("Input");
            auto outputs = makeComplexGraph(x);
            buffer       = Variable::save(outputs);
        }

        Module::Config config;
        config.shapeMutable = true;

        std::shared_ptr<Module> module(
            Module::load({"Input"}, {"Aux", "Output", "Shape"}, (const uint8_t*)buffer.data(), buffer.size(), &config),
            Module::destroy);
        if (nullptr == module) {
            return false;
        }
        {
            auto x   = _Input({1, 4, 320, 320}, NCHW, halide_type_of<float>());
            auto ptr = x->writeMap<float>();
            module->onForward({x});
        }

        auto input0   = makeInput(0.0f);
        auto outputs0 = module->onForward({input0});
        if (outputs0.size() != 3) {
            return false;
        }

        auto aux0Info   = outputs0[0]->getInfo();
        auto output0Info = outputs0[1]->getInfo();
        auto shape0Info = outputs0[2]->getInfo();
        if (nullptr == aux0Info || nullptr == output0Info || nullptr == shape0Info) {
            return false;
        }
        if (aux0Info->dim != std::vector<int>({1, 16, 16, 16})) {
            return false;
        }
        if (output0Info->dim != std::vector<int>({1, 4, 1, 16})) {
            return false;
        }
        if (shape0Info->dim != std::vector<int>({4})) {
            return false;
        }

        int expectedShape[4] = {1, 4, 1, 16};
        auto shapeResult0    = outputs0[2]->readMap<int>();
        if (!checkVector(shapeResult0, expectedShape, 4, 0)) {
            return false;
        }

        // Snapshot first inference results. We'll verify they remain unchanged after
        // the next inference (to catch output buffer reuse issues).
        std::vector<float> aux0Snapshot(aux0Info->size);
        std::vector<float> output0Snapshot(output0Info->size);
        std::vector<int> shape0Snapshot(4);
        {
            auto auxPtr = outputs0[0]->readMap<float>();
            auto outPtr = outputs0[1]->readMap<float>();
            for (int i = 0; i < aux0Info->size; ++i) {
                aux0Snapshot[i] = auxPtr[i];
            }
            for (int i = 0; i < output0Info->size; ++i) {
                output0Snapshot[i] = outPtr[i];
            }
            for (int i = 0; i < 4; ++i) {
                shape0Snapshot[i] = shapeResult0[i];
            }
        }

        auto aux0Tensor    = outputs0[0]->getTensor();
        auto output0Tensor = outputs0[1]->getTensor();
        auto shape0Tensor  = outputs0[2]->getTensor();
        if (nullptr == aux0Tensor || nullptr == output0Tensor || nullptr == shape0Tensor) {
            return false;
        }

        auto aux0DescribeOrigin    = TensorUtils::getDescribeOrigin(aux0Tensor);
        auto output0DescribeOrigin = TensorUtils::getDescribeOrigin(output0Tensor);
        auto shape0DescribeOrigin  = TensorUtils::getDescribeOrigin(shape0Tensor);
        if (nullptr == aux0DescribeOrigin || nullptr == output0DescribeOrigin || nullptr == shape0DescribeOrigin) {
            return false;
        }

        int aux0Offset    = aux0DescribeOrigin->offset;
        int output0Offset = output0DescribeOrigin->offset;
        int shape0Offset  = shape0DescribeOrigin->offset;

        if (isMetalRuntime() && output0Offset <= 0) {
            // Not a hard assert: offset depends on allocator strategy / platform.
            // But the model is designed to make output buffer more likely to be a
            // non-zero slice inside a shared MTLBuffer.
            MNN_PRINT("[StaticModuleOutputReuseTest] Metal output offset=%d (aux=%d, shape=%d)\n", output0Offset, aux0Offset,
                      shape0Offset);
        }

        auto input1   = makeInput(10.0f);
        auto outputs1 = module->onForward({input1});
        for (int i=0; i<10; ++i) {
            outputs1 = module->onForward({input1});
        }
        if (outputs1.size() != 3) {
            return false;
        }

        auto aux1Info    = outputs1[0]->getInfo();
        auto output1Info = outputs1[1]->getInfo();
        auto shape1Info  = outputs1[2]->getInfo();
        if (nullptr == aux1Info || nullptr == output1Info || nullptr == shape1Info) {
            return false;
        }
        if (aux1Info->dim != std::vector<int>({1, 16, 16, 16})) {
            return false;
        }
        if (output1Info->dim != std::vector<int>({1, 4, 1, 16})) {
            return false;
        }
        if (shape1Info->dim != std::vector<int>({4})) {
            return false;
        }

        auto shapeResult1 = outputs1[2]->readMap<int>();
        if (!checkVector(shapeResult1, expectedShape, 4, 0)) {
            return false;
        }


        // Ensure the previous forward's outputs are still valid, and their offsets
        // stay unchanged after the next forward.
        auto aux0InfoAfter    = outputs0[0]->getInfo();
        auto output0InfoAfter = outputs0[1]->getInfo();
        auto shape0InfoAfter  = outputs0[2]->getInfo();
        if (nullptr == aux0InfoAfter || nullptr == output0InfoAfter || nullptr == shape0InfoAfter) {
            return false;
        }
        if (aux0InfoAfter->dim != std::vector<int>({1, 16, 16, 16})) {
            return false;
        }
        if (output0InfoAfter->dim != std::vector<int>({1, 4, 1, 16})) {
            return false;
        }
        if (shape0InfoAfter->dim != std::vector<int>({4})) {
            return false;
        }

        // Ensure the previous inference results remain unchanged after the next forward.
        if (!checkVector(outputs0[0]->readMap<float>(), aux0Snapshot.data(), (int)aux0Snapshot.size(), 1e-4f)) {
            return false;
        }
        if (!checkVector(outputs0[1]->readMap<float>(), output0Snapshot.data(), (int)output0Snapshot.size(), 1e-4f)) {
            return false;
        }
        if (!checkVector(outputs0[2]->readMap<int>(), shape0Snapshot.data(), (int)shape0Snapshot.size(), 0)) {
            return false;
        }

        auto aux0DescribeAfterOrigin    = TensorUtils::getDescribeOrigin(outputs0[0]->getTensor());
        auto output0DescribeAfterOrigin = TensorUtils::getDescribeOrigin(outputs0[1]->getTensor());
        auto shape0DescribeAfterOrigin  = TensorUtils::getDescribeOrigin(outputs0[2]->getTensor());
        if (nullptr == aux0DescribeAfterOrigin || nullptr == output0DescribeAfterOrigin || nullptr == shape0DescribeAfterOrigin) {
            return false;
        }

        if (aux0Offset != aux0DescribeAfterOrigin->offset) {
            return false;
        }
        if (output0Offset != output0DescribeAfterOrigin->offset) {
            return false;
        }
        if (shape0Offset != shape0DescribeAfterOrigin->offset) {
            return false;
        }

        return true;
    }
};

MNNTestSuiteRegister(StaticModuleOutputReuseTest, "expr/StaticModuleOutputReuseTest");
