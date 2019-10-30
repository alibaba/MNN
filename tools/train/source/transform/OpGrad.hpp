//
//  OpGrad.hpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpGrad_hpp
#define OpGrad_hpp
#include <map>
#include <vector>
#include "Expr.hpp"
#include "ExprCreator.hpp"
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
        
        virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) = 0;
        
        static OpGrad* get(int type);
        static void insert(int type, OpGrad* creator);
        
    protected:
        Type mType = LINEAR;
    };
}


MNN_PUBLIC std::string numberToString(int index);
#endif
