//
//  calib3d.hpp
//  MNN
//
//  Created by MNN on 2022/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CALIB3D_HPP
#define CALIB3D_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "types.hpp"

namespace MNN {
namespace CV {

MNN_PUBLIC VARP Rodrigues(VARP src);

MNN_PUBLIC std::pair<VARP, VARP> solvePnP(VARP objectPoints, VARP imagePoints, VARP cameraMatrix, VARP distCoeffs,
                                          bool useExtrinsicGuess = false);

} // CV
} // MNN
#endif // CALIB3D_HPP
