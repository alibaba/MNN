//
//  RasterOutputTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/12/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/expr/Module.hpp>
#include "TestUtils.h"

using namespace MNN::Express;
using namespace MNN;

static std::shared_ptr<Module> _createModel() {
    auto x = _Input({1, 3, 224, 224}, NCHW, halide_type_of<int>());
    x->setName("Input");
    auto y = _Transpose(x, {0, 1, 3, 2});
    auto z = _Add(y, _Scalar<int>(1));
    z->setName("Add");
    auto q = _Negative(y);
    auto p = _Transpose(q, {0, 3, 1, 2});
    p->setName("Transpose");
    std::unique_ptr<NetT> net(new NetT);
    Variable::save({z, p}, net.get());
    flatbuffers::FlatBufferBuilder builder;
    auto len = MNN::Net::Pack(builder, net.get());
    builder.Finish(len);
    return std::shared_ptr<Module>(Module::load({"Input"}, {"Add", "Transpose"}, builder.GetBufferPointer(), builder.GetSize()));
}

class RasterOutputTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto net = _createModel();
        auto x = _Input({1, 3, 224, 224}, NCHW, halide_type_of<int>());
        auto y = _Transpose(x, {0, 1, 3, 2});
        auto z = _Add(y, _Scalar<int>(1));
        auto q = _Negative(y);
        auto p = _Transpose(q, {0, 3, 1, 2});
        {
            auto xPtr = x->writeMap<int>();
            for (int v = 0; v < 1 * 3 * 224 * 224; ++ v) {
                xPtr[v] = v;
            }
            x->unMap();
        }
        auto outputs = net->onForward({x});
        {
            auto dPtr = outputs[0]->readMap<int>();
            auto zPtr = z->readMap<int>();
            auto size = z->getInfo()->size;
            for (int v = 0; v < size; ++v) {
                if (zPtr[v] != dPtr[v]) {
                    return false;
                }
            }
        }
        {
            auto dPtr = outputs[1]->readMap<int>();
            auto zPtr = p->readMap<int>();
            auto size = p->getInfo()->size;
            for (int v = 0; v < size; ++v) {
                if (zPtr[v] != dPtr[v]) {
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RasterOutputTest, "expr/RasterOutput");
