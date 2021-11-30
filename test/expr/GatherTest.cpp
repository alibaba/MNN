//
//  GatherTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
 Test Case From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/gather-nd
 */
#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
using namespace MNN;
using namespace MNN::Express;

class GatherExprTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::unique_ptr<MNN::OpT> gatherOp(new MNN::OpT);
        gatherOp->type = MNN::OpType_GatherND;
        auto parameter = _Input({2, 2}, NHWC, halide_type_of<int32_t>());
        parameter->setName("param");
        auto indice    = _Input({2, 2}, NHWC, halide_type_of<int32_t>());
        indice->setName("indice");
        auto y         = Variable::create(Expr::create(gatherOp.get(), {parameter, indice}));
        y->setName("y");
        {
            parameter->resize({2, 2});
            auto ptr = parameter->writeMap<float>();
            ptr[0]   = 7.0;
            ptr[1]   = 2.0;
            ptr[2]   = 4.0;
            ptr[3]   = 6.0;
        }
        {
            auto indicePtr = indice->writeMap<int32_t>();
            indicePtr[0]   = 0;
            indicePtr[1]   = 0;
            indicePtr[2]   = 1;
            indicePtr[3]   = 1;
            auto size      = y->getInfo()->size;
            if (size != 2) {
                return false;
            }
            auto yPtr = y->readMap<float>();
            if (fabs(yPtr[0] - 7.0) > 0.001 || fabs(yPtr[1] - 6.0) > 0.001) {
                return false;
            }
        }
        {
            indice->resize({2, 1});
            auto indicePtr = indice->writeMap<int32_t>();
            indicePtr[0]   = 1;
            indicePtr[1]   = 0;
            auto size      = y->getInfo()->size;
            if (4 != size) {
                return false;
            }
            auto yPtr = y->readMap<float>();
            if (fabs(yPtr[0] - 4.0) > 0.001 || fabs(yPtr[1] - 6.0) > 0.001 || fabs(yPtr[2] - 7.0) > 0.001 ||
                fabs(yPtr[3] - 2.0) > 0.001) {
                return false;
            }
        }
        {
            indice->resize({1, 1});
            auto indicePtr = indice->writeMap<int32_t>();
            indicePtr[0]   = 1;
            parameter->resize({2, 2, 2});
            auto parameterPtr = parameter->writeMap<float>();
            for (int i = 0; i < parameter->getInfo()->size; ++i) {
                parameterPtr[i] = 1.0 * i;
            }
            auto size = y->getInfo()->size;
            if (4 != size) {
                return false;
            }
            auto yPtr = y->readMap<float>();
            for (int i = 0; i < size; ++i) {
                if (fabs(yPtr[i] - 4.0 - i) > 0.001) {
                    return false;
                }
            }
        }
        // Run as Module
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        {
            std::unique_ptr<MNN::NetT> net(new NetT);
            Variable::save({y}, net.get());
            y = nullptr;
            auto len = MNN::Net::Pack(builderOutput, net.get());
            builderOutput.Finish(len);
        }
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<MNN::Express::Module> module(Module::load(std::vector<std::string>{"param", "indice"}, std::vector<std::string>{"y"}, bufferOutput, sizeOutput));
        {
            {
                parameter->resize({2, 2});
                auto ptr = parameter->writeMap<float>();
                ptr[0]   = 7.0;
                ptr[1]   = 2.0;
                ptr[2]   = 4.0;
                ptr[3]   = 6.0;
            }
            {
                indice->resize({2, 2});
                auto indicePtr = indice->writeMap<int32_t>();
                indicePtr[0]   = 0;
                indicePtr[1]   = 0;
                indicePtr[2]   = 1;
                indicePtr[3]   = 1;
                auto y2 = module->onForward({parameter, indice})[0];
                auto size      = y2->getInfo()->size;
                if (size != 2) {
                    return false;
                }
                auto yPtr = y2->readMap<float>();
                if (fabs(yPtr[0] - 7.0) > 0.001 || fabs(yPtr[1] - 6.0) > 0.001) {
                    return false;
                }
            }
            {
                indice->resize({2, 1});
                auto indicePtr = indice->writeMap<int32_t>();
                indicePtr[0]   = 1;
                indicePtr[1]   = 0;
                auto y2 = module->onForward({parameter, indice})[0];
                auto size      = y2->getInfo()->size;
                if (4 != size) {
                    return false;
                }
                auto yPtr = y2->readMap<float>();
                if (fabs(yPtr[0] - 4.0) > 0.001 || fabs(yPtr[1] - 6.0) > 0.001 || fabs(yPtr[2] - 7.0) > 0.001 ||
                    fabs(yPtr[3] - 2.0) > 0.001) {
                    return false;
                }
            }
            {
                indice->resize({1, 1});
                auto indicePtr = indice->writeMap<int32_t>();
                indicePtr[0]   = 1;
                parameter->resize({2, 2, 2});
                auto parameterPtr = parameter->writeMap<float>();
                for (int i = 0; i < parameter->getInfo()->size; ++i) {
                    parameterPtr[i] = 1.0 * i;
                }
                auto y2 = module->onForward({parameter, indice})[0];
                auto yPtr = y2->readMap<float>();
                auto size = y2->getInfo()->size;
                if (4 != size) {
                    return false;
                }
                for (int i = 0; i < size; ++i) {
                    if (fabs(yPtr[i] - 4.0 - i) > 0.001) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(GatherExprTest, "expr/Gather");
