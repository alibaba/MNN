//
//  linearRegression.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include <random>
#include "DemoUnit.hpp"
#include "SGD.hpp"
using namespace MNN::Express;
using namespace MNN::Train;
std::random_device gRandom;
class LinearRegress : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        VARP w = _TrainableParam(0.3f, {}, NHWC);
        VARP b = _TrainableParam(0.1f, {}, NHWC);
        std::shared_ptr<Module> _m(Module::createEmpty({w, b}));
        std::shared_ptr<SGD> opt(new SGD(_m));
        opt->setLearningRate(0.1f);

        const int number = 10;
        const int limit  = 300;
        for (int i = 0; i < limit; ++i) {
            VARP x = _Input({number}, NHWC);
            // Fill x
            auto xPtr = x->writeMap<float>();
            for (int v = 0; v < number; ++v) {
                xPtr[v] = (gRandom() % 10000) / 10000.0f;
            }
            VARP label = _Input({number}, NHWC);
            // Fill label
            auto ptr = label->writeMap<float>();
            for (int v = 0; v < number; ++v) {
                ptr[v] = xPtr[v] * 0.8f + 0.7f;
            }
            VARP y = x * w + b;

            VARP diff = y - label;
            VARP loss = (diff * diff).mean({});

            if (i == limit - 1) {
                MNN_PRINT("w = %f, b = %f, Target w = 0.8f, Target b = 0.7f\n", w->readMap<float>()[0],
                          b->readMap<float>()[0]);
                Variable::save({y}, "linear.mnn");
            } else {
                opt->step(loss);
            }
        }
        return 0;
    }
};

DemoUnitSetRegister(LinearRegress, "LinearRegress");
