//
//  ConvolutionCommon.hpp
//  MNN
//
//  Created by MNN on b'2020/03/02'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvolutionCommon_hpp
#define ConvolutionCommon_hpp
#include "MNN_generated.h"
#include "Execution.hpp"
#include "AutoStorage.h"
namespace MNN {
class MNN_PUBLIC ConvolutionCommon : public Execution {
public:
    struct Int8Common {
        AutoStorage<int8_t> weight;
        AutoStorage<float> alpha;
        AutoStorage<float> weightFloat;
        const IDSTQuan* quan;
    };
    static std::shared_ptr<Int8Common> load(const IDSTQuan *quan, bool forceFloat = false);
    
    // Return padX, padY
    static std::pair<int, int> convolutionPad(const Tensor* input, const Tensor* output, const Convolution2DCommon* common);
    static std::pair<int, int> convolutionTransposePad(const Tensor* input, const Tensor* output, const Convolution2DCommon* common);
};
}
#endif
