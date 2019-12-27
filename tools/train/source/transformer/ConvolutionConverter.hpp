//
//  ConvolutionConverter.hpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionConverter_hpp
#define ConvolutionConverter_hpp

#include <stdio.h>
#include "OpConverter.hpp"
class ConvolutionConverter : public OpConverter {
public:
    virtual ReductResult onReduct(int opIndex, MNN::OpT* op, MNN::NetT* net) override;
};

#endif /* ConvolutionConverter_hpp */
