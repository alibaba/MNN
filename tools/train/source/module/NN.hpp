//
//  NN.hpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Train_NN_hpp
#define MNN_Train_NN_hpp
#include <MNN/expr/ExprCreator.hpp>
#include "Distributions.hpp"
#include "Module.hpp"
namespace MNN {
namespace Train {
class Initializer;

class MNN_PUBLIC NN {
public:
    struct ConvOption {
        Express::INTS kernelSize     = {1, 1};
        Express::INTS channel        = {0, 0};
        Express::INTS stride         = {1, 1};
        Express::INTS dilate         = {1, 1};
        Express::PaddingMode padMode = Express::VALID;
        Express::INTS pads           = {0, 0};
        bool depthwise               = false;

        void reset(int size = 2);
    };
    static std::shared_ptr<Module> Conv(const ConvOption& option, bool bias = true,
                                        std::shared_ptr<Initializer> weightInit = nullptr,
                                        std::shared_ptr<Initializer> biasInit   = nullptr);
    static std::shared_ptr<Module> ConvTranspose(const ConvOption& option, bool bias = true,
                                                 std::shared_ptr<Initializer> weightInit = nullptr,
                                                 std::shared_ptr<Initializer> biasInit   = nullptr);
    static std::shared_ptr<Module> Linear(int l, int t, bool hasBias = true,
                                          std::shared_ptr<Initializer> weightInit = nullptr,
                                          std::shared_ptr<Initializer> biasInit   = nullptr);
    static std::shared_ptr<Module> Dropout(const float dropRatio);
    static std::shared_ptr<Module> BatchNorm(const int channels, const int dims = 4,
                                            const float m = 0.999, const float e = 1e-5);
};

} // namespace Train
} // namespace MNN

#endif
