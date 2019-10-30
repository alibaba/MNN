//
//  PaddingTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ExprCreator.hpp"
#include "MNNTestSuite.h"
#include "MNN_generated.h"
using namespace MNN::Express;
using namespace MNN;

static void fillVar(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr = x->writeMap<int32_t>();
    for (int i=0; i<size; ++i) {
        ptr[i] = i + 1;
    }
}
static void printVar(VARP x) {
    auto size = x->getInfo()->size;
    auto ptr = x->readMap<int32_t>();
    for (int i=0; i<size; ++i) {
        MNN_PRINT("%d, ", ptr[i]);
    }
    MNN_PRINT("\n");
}

class PaddingTest : public MNNTestCase {
public:
    virtual bool run() {
        std::unique_ptr<OpT> padding(new OpT);
        padding->type = OpType_Padding;
        {
            auto x = _Input({4, 6}, NCHW, halide_type_of<int32_t>());
            auto pad = _Input({4}, NCHW, halide_type_of<int32_t>());
            auto paddingPtr = pad->writeMap<int32_t>();
            paddingPtr[0] = 0;
            paddingPtr[1] = 1;
            paddingPtr[2] = 1;
            paddingPtr[3] = 1;
            fillVar(x);
            auto y = Variable::create(Expr::create(padding.get(), {x, pad}));
            {
                auto size = y->getInfo()->dim;
                auto ptr = y->readMap<int32_t>();
                for (int i=0; i<size[0]; ++i) {
                    for (int j=0; j<size[1]; ++j) {
                        auto compute = ptr[i*8+j];
                        auto expect = i*6 + (j-1) + 1;
                        if (i >= 4 || j < 1 || j >= 7) {
                            expect = 0;
                        }
                        if (compute != expect) {
                            FUNC_PRINT(1);
                            return false;
                        }
                    }
                }
            }
        }
        {
            auto x = _Input({1, 3, 4, 6}, NCHW, halide_type_of<int32_t>());
            auto convert = _Convert(x, NC4HW4);
            auto pad = _Input({8}, NCHW, halide_type_of<int32_t>());
            auto paddingPtr = pad->writeMap<int32_t>();
            paddingPtr[0] = 0;
            paddingPtr[1] = 1;
            paddingPtr[2] = 0;
            paddingPtr[3] = 0;
            paddingPtr[4] = 1;
            paddingPtr[5] = 1;
            paddingPtr[6] = 1;
            paddingPtr[7] = 1;
            fillVar(x);
            auto y = Variable::create(Expr::create(padding.get(), {x, pad}));
            auto yC4 = _Convert(Variable::create(Expr::create(padding.get(), {convert, pad})), NCHW);
            {
                auto info = y->getInfo();
                auto info2 = yC4->getInfo();
                if(info->size != info2->size) {
                    FUNC_PRINT(1);
                    return false;
                }
                auto ptr0 = y->readMap<int32_t>();
                auto ptr1 = yC4->readMap<int32_t>();
                for (int i=0; i<info->size; ++i) {
                    if (ptr0[i] != ptr1[i]) {
                        FUNC_PRINT(1);
                        return false;
                    }
                }
            }
            
            paddingPtr = pad->writeMap<int32_t>();
            paddingPtr[0] = 0;
            paddingPtr[1] = 1;
            paddingPtr[2] = 0;
            paddingPtr[3] = 1;
            paddingPtr[4] = 1;
            paddingPtr[5] = 1;
            paddingPtr[6] = 1;
            paddingPtr[7] = 1;
            {
                auto info = y->getInfo();
                auto info2 = yC4->getInfo();
                if(info->size != info2->size) {
                    FUNC_PRINT(1);
                    return false;
                }
                auto ptr0 = y->readMap<int32_t>();
                auto ptr1 = yC4->readMap<int32_t>();
                for (int i=0; i<info->size; ++i) {
                    if (ptr0[i] != ptr1[i]) {
                        FUNC_PRINT(1);
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(PaddingTest, "expr/Padding");
