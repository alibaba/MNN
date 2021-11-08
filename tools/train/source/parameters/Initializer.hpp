//
//  Initializer.hpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Initializer_hpp
#define Initializer_hpp

#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace Train {
class RandomGenerator;
class MNN_PUBLIC Initializer {
public:
    Initializer()          = default;
    virtual ~Initializer() = default;
    Express::VARP createConstVar(Express::INTS dim, Express::Dimensionformat format = Express::NCHW);
    virtual void onExecute(Express::VARP p) = 0;

    static Initializer* constValue(float value);
    static Initializer* uniform(float minValue = 0.0f, float maxValue = 1.0f);

    enum VarianceNorm {
        FANIN,
        FANOUT,
        AVERAGE,
    };

    static Initializer* xavier(VarianceNorm norm = FANIN);
    static Initializer* gauss(float mean = 0.0f, float std = 1.0f);
    static Initializer* MSRA(VarianceNorm norm = FANIN);
    static Initializer* bilinear();
    static Initializer* positiveUnitball();
};

} // namespace Train
} // namespace MNN

#endif // Initializer_hpp
