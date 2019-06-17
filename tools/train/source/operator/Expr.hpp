//
//  Expr.hpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Expr_hpp
#define Expr_hpp

#include <stdio.h>
#include <vector>
#include "converter/source/IR/MNN_generated.h"

namespace MNN {
class Expr {
public:
    Expr();
    ~Expr() {
    }
    void render(NetT* dest);
    void addChild(std::shared_ptr<Expr> expr);
    std::vector<int> getInputs();
    std::vector<int> getOutputs();

private:
    std::unique_ptr<OpT> mOp;
    std::vector<std::shared_ptr<Expr>> mSubExpr;
};
} // namespace MNN

#endif /* Expr_hpp */
