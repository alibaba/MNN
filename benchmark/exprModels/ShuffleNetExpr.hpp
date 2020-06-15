//
//  ShuffleNetExpr.hpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1707.01083.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ShuffleNetExpr_hpp
#define ShuffleNetExpr_hpp

#include <MNN/expr/Expr.hpp>

MNN::Express::VARP shuffleNetExpr(int group, int numClass);

#endif // ShuffleNetExpr_hpp
