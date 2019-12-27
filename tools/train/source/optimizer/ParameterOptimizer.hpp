//
//  ParameterOptimizer.hpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ParameterOptimizer_hpp
#define ParameterOptimizer_hpp
#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace Train {

class MNN_PUBLIC ParameterOptimizer {
public:
    ParameterOptimizer()          = default;
    virtual ~ParameterOptimizer() = default;
    bool step(Express::VARP loss);
    virtual std::map<Express::VARP, Express::VARP> onGetNextParameter(Express::VARP loss) = 0;
};

} // namespace Train
} // namespace MNN

#endif
