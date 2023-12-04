//
//  OpGrad.hpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpGrad_hpp
#define OpGrad_hpp
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <map>
#include <vector>
#include "MNN_generated.h"

namespace MNN {
class MNN_PUBLIC OpGrad {
public:
    enum Type { LINEAR, SEMI_LINEAR, NO_LINEAR };

    OpGrad()          = default;
    virtual ~OpGrad() = default;

    Type type() const {
        return mType;
    }
    static Express::VARP divideAvoidZero(MNN::Express::VARP y, MNN::Express::VARP x);

    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) = 0;

    static OpGrad* get(int type);
    static void insert(int type, OpGrad* creator);
    static std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>> gradCommon(std::vector<Express::VARP> outputs, std::vector<Express::VARP> outputDiff, std::vector<Express::VARP> parameters);

    static std::vector<Express::VARP> gradLinear(Express::VARP loss, const std::vector<Express::VARP>& parameters, const std::vector<Express::VARP>& outputDiff, const std::vector<std::string> blockExpr = {});
    static std::map<Express::VARP, Express::VARP> gradCommon(std::vector<Express::VARP> outputs, const std::set<Express::VARP>& parameters, std::map<Express::EXPRP, std::vector<Express::VARP>>& backwardMap, const std::vector<std::string> blockExpr = {});
    static std::map<Express::VARP, Express::VARP> grad(Express::VARP loss, const std::set<Express::VARP>& parameters, const std::vector<std::string> blockExpr = {});

protected:
    Type mType = LINEAR;
};
} // namespace MNN

#endif
