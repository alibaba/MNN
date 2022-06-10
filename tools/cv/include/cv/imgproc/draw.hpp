//
//  draw.hpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DRAW_HPP
#define DRAW_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "../types.hpp"

namespace MNN {
namespace CV {

enum LineTypes {
  FILLED = -1,
  LINE_4 = 4,
  LINE_8 = 8,
  LINE_AA = 16
};

MNN_PUBLIC void arrowedLine(VARP& img, Point pt1, Point pt2, const Scalar& color,
                            int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);
MNN_PUBLIC void circle(VARP& img, Point center, int radius, const Scalar& color,
                       int thickness=1, int line_type=8, int shift=0);

MNN_PUBLIC void line(VARP& img, Point pt1, Point pt2, const Scalar& color,
                     int thickness = 1, int lineType = LINE_8, int shift = 0);

MNN_PUBLIC void rectangle(VARP& img, Point pt1, Point pt2, const Scalar& color,
                          int thickness = 1, int lineType = LINE_8, int shift = 0);

MNN_PUBLIC void drawContours(VARP& img, std::vector<std::vector<Point>> _contours, int contourIdx, const Scalar& color,
                             int thickness = 1, int lineType = LINE_8);

MNN_PUBLIC void fillPoly(VARP& img, std::vector<std::vector<Point>> pts, const Scalar& color,
                         int line_type = LINE_8, int shift = 0, Point offset = {0, 0});
} // CV
} // MNN
#endif // DRAW_HPP
