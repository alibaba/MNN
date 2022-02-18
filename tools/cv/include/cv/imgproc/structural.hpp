//
//  structural.hpp
//  MNN
//
//  Created by MNN on 2021/12/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef STRUCTURAL_HPP
#define STRUCTURAL_HPP

#include <MNN/MNNDefine.h>
#include "cv/types.hpp"

namespace MNN {
namespace CV {

enum RetrievalModes {
    RETR_EXTERNAL = 0,
    RETR_LIST = 1,
    RETR_CCOMP = 2,
    RETR_TREE = 3,
    RETR_FLOODFILL = 4
};

enum ContourApproximationModes {
    CHAIN_APPROX_NONE = 1,
    CHAIN_APPROX_SIMPLE = 2,
    CHAIN_APPROX_TC89_L1 = 3,
    CHAIN_APPROX_TC89_KCOS = 4
};

class RotatedRect
{
public:
    //! default constructor
    RotatedRect() {}
    //! returns the rectangle mass center
    Point2f center;
    //! returns width and height of the rectangle
    Size2f size;
    //! returns the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
    float angle;
};
typedef std::vector<Point> POINTS;

MNN_PUBLIC std::vector<VARP> findContours(VARP image, int mode, int method, Point offset = {0, 0});
MNN_PUBLIC double contourArea(VARP _contour, bool oriented = false);
MNN_PUBLIC std::vector<int> convexHull(VARP _points, bool clockwise = false, bool returnPoints = true);
MNN_PUBLIC RotatedRect minAreaRect(VARP _points);
MNN_PUBLIC Rect2i boundingRect(VARP points);
MNN_PUBLIC int connectedComponentsWithStats(VARP image, VARP& labels, VARP& statsv, VARP& centroids, int connectivity = 8);
MNN_PUBLIC VARP boxPoints(RotatedRect box);
} // CV
} // MNN
#endif // STRUCTURAL_HPP
