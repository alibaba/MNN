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
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
using namespace MNN::Express;

class GatherTest : public MNNTestCase {
public:
    virtual bool run() {
        std::unique_ptr<MNN::OpT> gatherOp(new MNN::OpT);
        gatherOp->type = MNN::OpType_GatherND;
        auto parameter = _Input({2, 2}, NHWC, halide_type_of<int32_t>());
        auto indice = _Input({2, 2}, NHWC, halide_type_of<int32_t>());
        auto y = Variable::create(Expr::create(gatherOp.get(), {parameter, indice}));
        {
            parameter->resize({2, 2});
            auto ptr = parameter->writeMap<int32_t>();
            ptr[0] = 7;
            ptr[1] = 2;
            ptr[2] = 4;
            ptr[3] = 6;
        }
        {
            auto indicePtr = indice->writeMap<int32_t>();
            indicePtr[0] = 0;
            indicePtr[1] = 0;
            indicePtr[2] = 1;
            indicePtr[3] = 1;
            auto size = y->getInfo()->size;
            if (size != 2) {
                return false;
            }
            auto yPtr = y->readMap<int32_t>();
            if (yPtr[0] != 7 || yPtr[1] != 6) {
                return false;
            }
        }
        {
            indice->resize({2, 1});
            auto indicePtr = indice->writeMap<int32_t>();
            indicePtr[0] = 1;
            indicePtr[1] = 0;
            auto size = y->getInfo()->size;
            if (4 != size) {
                return false;
            }
            auto yPtr = y->readMap<int32_t>();
            if (yPtr[0] != 4 || yPtr[1] != 6 || yPtr[2] != 7 || yPtr[3] != 2) {
                return false;
            }
        }
        {
            indice->resize({1, 1});
            auto indicePtr = indice->writeMap<int32_t>();
            indicePtr[0] = 1;
            parameter->resize({2, 2, 2});
            auto parameterPtr = parameter->writeMap<int32_t>();
            for (int i=0; i<parameter->getInfo()->size; ++i) {
                parameterPtr[i] = i;
            }
            auto size = y->getInfo()->size;
            if (4 != size) {
                return false;
            }
            auto yPtr = y->readMap<int32_t>();
            for (int i=0; i<size; ++i) {
                if (yPtr[i] != 4 + i) {
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(GatherTest, "expr/Gather");
