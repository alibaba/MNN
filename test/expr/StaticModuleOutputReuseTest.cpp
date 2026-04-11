#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/TensorUtils.hpp"

using namespace MNN;
using namespace MNN::Express;

class StaticModuleOutputReuseTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto executor = cloneCurrentExecutor();
        ExecutorScope scope(executor);

        std::vector<int8_t> buffer;
        {
            auto x = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            x->setName("Input");
            auto y = x + _Scalar<float>(1.0f);
            y->setName("Output");
            auto s = _Shape(x);
            s->setName("Shape");
            buffer = Variable::save({y, s});
        }

        Module::Config config;
        config.shapeMutable = true;

        std::shared_ptr<Module> module(Module::load({"Input"}, {"Output", "Shape"}, (const uint8_t*)buffer.data(), buffer.size(), &config), Module::destroy);
        if (nullptr == module) {
            return false;
        }

        auto makeInput = [](float base) {
            auto x = _Input({1, 1, 1, 4}, NCHW, halide_type_of<float>());
            auto ptr = x->writeMap<float>();
            for (int i = 0; i < 4; ++i) {
                ptr[i] = base + (float)i;
            }
            return x;
        };

        auto input0 = makeInput(0.0f);
        auto outputs0 = module->onForward({input0});
        if (outputs0.size() != 2) {
            return false;
        }

        auto output0Info = outputs0[0]->getInfo();
        auto shape0Info = outputs0[1]->getInfo();
        if (nullptr == output0Info || nullptr == shape0Info) {
            return false;
        }
        if (output0Info->dim != std::vector<int>({1, 1, 1, 4})) {
            return false;
        }
        if (shape0Info->dim != std::vector<int>({4})) {
            return false;
        }

        float expected0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        int expectedShape[4] = {1, 1, 1, 4};

        auto result0 = outputs0[0]->readMap<float>();
        auto shapeResult0 = outputs0[1]->readMap<int>();
        if (!checkVector(result0, expected0, 4, 1e-6f)) {
            return false;
        }
        if (!checkVector(shapeResult0, expectedShape, 4, 0)) {
            return false;
        }

        auto output0Tensor = outputs0[0]->getTensor();
        auto shape0Tensor = outputs0[1]->getTensor();
        if (nullptr == output0Tensor || nullptr == shape0Tensor) {
            return false;
        }
        auto output0DescribeOrigin = TensorUtils::getDescribeOrigin(output0Tensor);
        auto shape0DescribeOrigin = TensorUtils::getDescribeOrigin(shape0Tensor);
        if (nullptr == output0DescribeOrigin || nullptr == shape0DescribeOrigin) {
            return false;
        }
        int output0Offset = output0DescribeOrigin->offset;
        int shape0Offset = shape0DescribeOrigin->offset;

        auto input1 = makeInput(10.0f);
        auto outputs1 = module->onForward({input1});
        if (outputs1.size() != 2) {
            return false;
        }

        auto output1Info = outputs1[0]->getInfo();
        auto shape1Info = outputs1[1]->getInfo();
        if (nullptr == output1Info || nullptr == shape1Info) {
            return false;
        }
        if (output1Info->dim != std::vector<int>({1, 1, 1, 4})) {
            return false;
        }
        if (shape1Info->dim != std::vector<int>({4})) {
            return false;
        }

        float expected1[4] = {11.0f, 12.0f, 13.0f, 14.0f};

        auto result1 = outputs1[0]->readMap<float>();
        auto shapeResult1 = outputs1[1]->readMap<int>();
        if (!checkVector(result1, expected1, 4, 1e-6f)) {
            return false;
        }
        if (!checkVector(shapeResult1, expectedShape, 4, 0)) {
            return false;
        }

        auto output0InfoAfter = outputs0[0]->getInfo();
        auto shape0InfoAfter = outputs0[1]->getInfo();
        if (nullptr == output0InfoAfter || nullptr == shape0InfoAfter) {
            return false;
        }
        if (output0InfoAfter->dim != std::vector<int>({1, 1, 1, 4})) {
            return false;
        }
        if (shape0InfoAfter->dim != std::vector<int>({4})) {
            return false;
        }

        auto output0DescribeAfterOrigin = TensorUtils::getDescribeOrigin(outputs0[0]->getTensor());
        auto shape0DescribeAfterOrigin = TensorUtils::getDescribeOrigin(outputs0[1]->getTensor());
        if (nullptr == output0DescribeAfterOrigin || nullptr == shape0DescribeAfterOrigin) {
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
