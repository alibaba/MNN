//
//  ResNetExpr.hpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1512.03385.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ResNetExpr_hpp
#define ResNetExpr_hpp

#include <map>
#include <string>
#include <MNN/expr/Expr.hpp>

enum ResNetType {
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
};

static inline ResNetType EnumResNetTypeByString(const std::string& key) {
    auto resNetTypeMap = std::map<std::string, ResNetType>({
        {"18", ResNet18},
        {"34", ResNet34},
        {"50", ResNet50},
        {"101", ResNet101},
        {"152", ResNet152}});
    auto resNetTypeIter = resNetTypeMap.find(key);
    if (resNetTypeIter == resNetTypeMap.end()) {
        return (ResNetType)(-1);
    }
    return resNetTypeIter->second;
}

MNN::Express::VARP resNetExpr(ResNetType resNetType, int numClass);

#endif //ResNetExpr_hpp
