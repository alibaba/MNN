//
//  draw.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ImageProcess.hpp>
#include "cv/imgproc/draw.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <cmath>

namespace MNN {
namespace CV {

// help functions
// TODO: replace this function with an Op.
void bresenham(uint8_t* ptr, int h, int w, int c, int x1, int y1, int x2, int y2, Scalar color) {
    int x = x1;
    int y = y1;
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int s1 = x2 > x1 ? 1 : -1;
    int s2 = y2 > y1 ? 1 : -1;
    bool interchange = false;
    if (dy > dx) {
        std::swap(dx, dy);
        interchange = true;
    }
    int p = 2 * dy - dx;
    for(int i = 0; i <= dx; i++) {
        // printf("[%d, %d]\n", x, y);
        memcpy(ptr + (y * w + x) * c, &color, c);
        if (p >= 0) {
            if (interchange) {
                x += s1;
            } else {
                y += s2;
            }
            p -= 2 * dx;
        }
        if (interchange) {
            y += s2;
        } else {
            x += s1;
        }
        p += 2 * dy;
    }
}

std::vector<int> getPoints(Point pt1, Point pt2, int thickness) {
    int x1 = pt1.fX, y1 = pt1.fY, x2 = pt2.fX, y2 = pt2.fY;
    std::vector<int> pts { x1, y1, x2, y2 };
    for (int i = 0; i < thickness; i++) {
        // x - i;
    }
    return pts;
}

void arrowedLine(VARP& img, Point pt1, Point pt2, const Scalar& color,
                 int thickness, int line_type, int shift, double tipLength) {
    // line
    line(img, pt1, pt2, color, thickness, line_type, shift);
    float deltaX = pt1.fX - pt2.fX, deltaY = pt1.fY - pt2.fY;
    const double tipSize = std::sqrt(deltaX * deltaX + deltaY * deltaY) * tipLength;
    const double angle = atan2(pt1.fY - pt2.fY, pt1.fX - pt2.fX);
    // arrawed edge 1
    Point p;
    p.fX = std::round(pt2.fX + tipSize * cos(angle + MNN_PI / 4));
    p.fY = std::round(pt2.fY + tipSize * sin(angle + MNN_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
    // arrawed edge 2
    p.fX = std::round(pt2.fX + tipSize * cos(angle - MNN_PI / 4));
    p.fY = std::round(pt2.fY + tipSize * sin(angle - MNN_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
}

void line(VARP& img, Point pt1, Point pt2, const Scalar& color,
          int thickness, int lineType, int shift) {
    int h = 0, w = 0, c = 0;
    getVARPSize(img, &h, &w, &c);
    auto ptr = img->writeMap<uint8_t>();
    int x1 = static_cast<int>(pt1.fX), y1 = static_cast<int>(pt1.fY);
    int x2 = static_cast<int>(pt2.fX), y2 = static_cast<int>(pt2.fY);
    for (int i = 0; i < thickness; i++) {
        // bresenham(ptr, h, w, c, x1[i], y1[i], x2[i], y2[i], color);
    }
    bresenham(ptr, h, w, c, x1, y1, x2, y2, color);
}

void rectangle(VARP& img, Point pt1, Point pt2, const Scalar& color,
               int thickness, int lineType, int shift) {
    // top
    line(img, pt1, {pt2.fX, pt1.fY}, color, thickness, lineType);
    // left
    line(img, pt1, {pt1.fX, pt2.fY}, color, thickness, lineType);
    // right
    line(img, {pt2.fX, pt1.fY}, pt2, color, thickness, lineType);
    // bottom
    line(img, {pt1.fX, pt2.fY}, pt2, color, thickness, lineType);
}

} // CV
} // MNN
