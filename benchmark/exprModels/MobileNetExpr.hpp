//
//  MobileNetExpr.hpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1704.04861.pdf https://arxiv.org/pdf/1801.04381.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobileNetExpr_hpp
#define MobileNetExpr_hpp

#include <map>
#include <string>
#include <MNN/expr/Expr.hpp>

enum MobileNetWidthType {
    MobileNet_100, MobileNet_075, MobileNet_050, MobileNet_025
};

enum MobileNetResolutionType {
    MobileNet_224, MobileNet_192, MobileNet_160, MobileNet_128
};

static inline MobileNetWidthType EnumMobileNetWidthTypeByString(const std::string& key) {
    auto mobileNetWidthTypeMap = std::map<std::string, MobileNetWidthType>({
        {"1.0", MobileNet_100},
        {"0.75", MobileNet_075},
        {"0.5", MobileNet_050},
        {"0.25", MobileNet_025}});
    auto mobileNetWidthTypeIter = mobileNetWidthTypeMap.find(key);
    if (mobileNetWidthTypeIter == mobileNetWidthTypeMap.end()) {
        return (MobileNetWidthType)(-1);
    }
    return mobileNetWidthTypeIter->second;
}

static inline MobileNetResolutionType EnumMobileNetResolutionTypeByString(const std::string& key) {
    auto mobileNetResolutionTypeMap = std::map<std::string, MobileNetResolutionType>({
        {"224", MobileNet_224},
        {"192", MobileNet_192},
        {"160", MobileNet_160},
        {"128", MobileNet_128}});
    auto mobileNetResolutionTypeIter = mobileNetResolutionTypeMap.find(key);
    if (mobileNetResolutionTypeIter == mobileNetResolutionTypeMap.end()) {
        return (MobileNetResolutionType)(-1);
    }
    return mobileNetResolutionTypeIter->second;
}

MNN::Express::VARP mobileNetV1Expr(MobileNetWidthType alpha, MobileNetResolutionType beta, int numClass);
MNN::Express::VARP mobileNetV2Expr(int numClass);

#endif //MobileNetExpr_hpp
