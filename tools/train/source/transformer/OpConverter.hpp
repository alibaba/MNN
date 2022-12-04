//
//  OpConverter.hpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpConverter_hpp
#define OpConverter_hpp
#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include "MNN_generated.h"

class MNN_PUBLIC OpConverter {
public:
    OpConverter() = default;

    static MNN::Express::EXPRP convert(MNN::Express::EXPRP source, std::map<std::string, MNN::Express::VARP>& helpInfo);

    virtual ~OpConverter() = default;
    static OpConverter* get(MNN::OpType type);
    static void insert(MNN::OpType type, OpConverter* converter);

    struct ReductResult {
        std::vector<int> needDeleteOpIndexes;
    };
    virtual ReductResult onReduct(int opIndex, MNN::OpT* op, MNN::NetT* net) = 0;
};
#endif
