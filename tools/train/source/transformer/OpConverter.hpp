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
namespace MNN {
struct TrainInfo {
    std::map<std::string, Express::VARP> bnVariables;
    std::map<std::string, std::pair<std::string, std::string>> convolutionVariables;
    std::map<std::string, std::string> trainables;
};
class MNN_PUBLIC OpConverter {
public:
    OpConverter() = default;

    static MNN::Express::EXPRP convert(MNN::Express::EXPRP source, TrainInfo& helpInfo);

    virtual ~OpConverter() = default;
    static OpConverter* get(int type);
    static void insert(int type, OpConverter* converter);
};
};
#endif
