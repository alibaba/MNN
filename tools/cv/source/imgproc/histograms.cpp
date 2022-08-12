//
//  histograms.cpp
//  MNN
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/imgproc/histograms.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>

namespace MNN {
namespace CV {
VARP calcHist(VARPS images, const std::vector<int>& channels, VARP mask,
              const std::vector<int>& histSize, const std::vector<float>& ranges, bool accumulate) {
    // last bin deal different with opencv, same as torch and numpy
    return _Histogram(images[0], histSize[0], int(ranges[0]), int(ranges[1]), channels[0]);
}
} // CV
} // MNN
