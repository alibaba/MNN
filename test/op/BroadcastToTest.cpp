//
//  BroadcastToTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "MNN_generated.h"

using namespace MNN::Express;
using namespace MNN;

class BroadcastToTest : public MNNTestCase {
    virtual ~BroadcastToTest() = default;

    virtual bool run(int precision) {
        {
            const float tensorData[]   = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            const int shapeData[]      = {2, 3, 2, 2};
            const float expectedData[] = {
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0,
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0,
            };

            auto tensor = _Const(tensorData, {1, 3, 1, 2}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 24) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({2, 3, 2, 2})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0};
            const int shapeData[]      = {3, 3};
            const float expectedData[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

            auto tensor = _Const(tensorData, {1, 3}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {2}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 9) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({3, 3})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0};
            const int shapeData[]      = {3, 3};
            const float expectedData[] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};

            auto tensor = _Const(tensorData, {3, 1}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {2}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 9) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({3, 3})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0};
            const int shapeData[]      = {2, 3, 2, 2};
            const float expectedData[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};

            auto tensor = _Const(tensorData, {1, 1, 1, 2}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 24) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({2, 3, 2, 2})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0};
            const int shapeData[]      = {2, 3, 2, 2};
            const float expectedData[] = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                          1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0};

            auto tensor = _Const(tensorData, {1, 3, 1, 1}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 24) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({2, 3, 2, 2})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            const int shapeData[]      = {1, 1, 1, 1};
            const float expectedData[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

            auto tensor = _Const(tensorData, {1, 3, 1, 2}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 6) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({1, 3, 1, 2})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0};
            const int shapeData[]      = {2, 1, 2};
            const float expectedData[] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
                                          1.0, 1.0, 2.0, 2.0, 3.0, 3.0};

            auto tensor = _Const(tensorData, {3, 1}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {3}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            if (size != 12) {
                return false;
            }
            auto& dims = result->getInfo()->dim;
            if (dims != std::vector<int>({2, 3, 2})) {
                return false;
            }
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }
        return true;
    }
};

class BinaryBroadcastTest : public MNNTestCase {
    virtual ~BinaryBroadcastTest() = default;

    virtual bool run(int precision) {
        auto X = _Input({2, 5, 2}, NHWC, halide_type_of<float>());
        X->setName("X");
        auto y0 = _Input({}, NHWC, halide_type_of<float>());
        y0->writeMap<float>()[0] = 1.0f;
        auto y1 = _Input({1, 1, 2}, NHWC, halide_type_of<float>());
        y1->writeMap<float>()[0] = 1.0f;
        y1->writeMap<float>()[1] = 2.0f;

        auto y2 = _Input({2, 1, 2}, NHWC, halide_type_of<float>());
        y2->writeMap<float>()[0] = 1.0f;
        y2->writeMap<float>()[1] = 2.0f;
        y2->writeMap<float>()[2] = 3.0f;
        y2->writeMap<float>()[3] = 4.0f;
        y0.fix(VARP::CONSTANT);
        y1.fix(VARP::CONSTANT);
        y2.fix(VARP::CONSTANT);
        // Run as Module
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        {
            auto z0 = _Add(X, y0);
            z0->setName("z0");
            auto z1 = _Add(X, y1);
            z1->setName("z1");
            auto z2 = _Add(X, y2);
            z2->setName("z2");
            auto z3 = _Add(y2, X);
            z3->setName("z3");
            std::unique_ptr<MNN::NetT> net(new NetT);
            Variable::save({z0, z1, z2, z3}, net.get());
            auto len = MNN::Net::Pack(builderOutput, net.get());
            builderOutput.Finish(len);
        }
        float error = (precision <= MNN::BackendConfig::Precision_High ? 1 : 100) * 0.0005f;
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<MNN::Express::Module> module(Module::load(std::vector<std::string>{"X"}, std::vector<std::string>{"z0", "z1", "z2", "z3"}, bufferOutput, sizeOutput));
        // First
        {
            auto x0 = _Input({2, 1, 2}, NHWC, halide_type_of<float>());
            auto size = x0->getInfo()->size;
            auto ptr = x0->writeMap<float>();
            for (int i=0; i<size; ++i) {
                ptr[i] = (float)(i+1) * 0.1f;
            }
            auto z = module->onForward({x0});
            std::vector<float> z0Target = {
                1.1f, 1.2f, 1.3f, 1.4f
            };
            std::vector<float> z1Target = {
                1.1f, 2.2f, 1.3f, 2.4f
            };
            std::vector<float> z2Target = {
                1.1f, 2.2f, 3.3f, 4.4f
            };
            if (!checkVector(z[0]->readMap<float>(), z0Target.data(), 4, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[1]->readMap<float>(), z1Target.data(), 4, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[2]->readMap<float>(), z2Target.data(), 4, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[3]->readMap<float>(), z2Target.data(), 4, error)) {
                FUNC_PRINT(1);
                return false;
            }
        }
        {
            auto x0 = _Input({2, 5, 2}, NHWC, halide_type_of<float>());
            auto size = x0->getInfo()->size;
            auto ptr = x0->writeMap<float>();
            for (int i=0; i<size; ++i) {
                ptr[i] = (float)(i+1) * 0.1f;
            }
            auto z = module->onForward({x0});
            std::vector<float> z0Target(2 * 5 * 2);
            std::vector<float> z1Target(2 * 5 * 2);
            std::vector<float> z2Target(2 * 5 * 2);
            for (int i=0; i<2; ++i) {
                for (int j=0; j<5; ++j) {
                    for (int k=0; k<2; ++k) {
                        auto index = i*10+j*2+k;
                        z0Target[index] = (float)(index+1) * 0.1f + 1.0f;
                        z1Target[index] = (float)(index+1) * 0.1f + (float)(k + 1) * 1.0f;
                        z2Target[index] = (float)(index+1) * 0.1f + (float)(k + i * 2 + 1) * 1.0f;
                    }
                }
            }
            auto tsize = 2 * 5 * 2;
            if (!checkVector(z[0]->readMap<float>(), z0Target.data(), tsize, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[1]->readMap<float>(), z1Target.data(), tsize, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[2]->readMap<float>(), z2Target.data(), tsize, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[3]->readMap<float>(), z2Target.data(), 4, error)) {
                FUNC_PRINT(1);
                return false;
            }
        }
        {
            auto x0 = _Input({2, 3, 2}, NHWC, halide_type_of<float>());
            auto size = x0->getInfo()->size;
            auto ptr = x0->writeMap<float>();
            for (int i=0; i<size; ++i) {
                ptr[i] = (float)(i+1) * 0.1f;
            }
            auto z = module->onForward({x0});
            std::vector<float> z0Target(2 * 3 * 2);
            std::vector<float> z1Target(2 * 3 * 2);
            std::vector<float> z2Target(2 * 3 * 2);
            for (int i=0; i<2; ++i) {
                for (int j=0; j<3; ++j) {
                    for (int k=0; k<2; ++k) {
                        auto index = i*6+j*2+k;
                        z0Target[index] = (float)(index+1) * 0.1f + 1.0f;
                        z1Target[index] = (float)(index+1) * 0.1f + (float)(k + 1) * 1.0f;
                        z2Target[index] = (float)(index+1) * 0.1f + (float)(k + i * 2 + 1) * 1.0f;
                    }
                }
            }
            auto tsize = 2 * 3 * 2;
            if (!checkVector(z[0]->readMap<float>(), z0Target.data(), tsize, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[1]->readMap<float>(), z1Target.data(), tsize, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[2]->readMap<float>(), z2Target.data(), tsize, error)) {
                FUNC_PRINT(1);
                return false;
            }
            if (!checkVector(z[3]->readMap<float>(), z2Target.data(), 4, error)) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(BroadcastToTest, "op/BroadcastToTest");
MNNTestSuiteRegister(BinaryBroadcastTest, "op/BinaryBroadcastTest");
