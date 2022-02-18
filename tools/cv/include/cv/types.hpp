//
//  types.hpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TYPES_HPP
#define TYPES_HPP

#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace CV {

using namespace Express;

#define MNN_PI 3.1415926535897932384626433832795

typedef signed char schar;
typedef unsigned char uchar;

// Size Start
template<typename _Tp> class Size_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Size_();
    Size_(_Tp _width, _Tp _height);
    Size_(const Size_& sz);
    Size_(Size_&& sz);

    Size_& operator = (const Size_& sz);
    Size_& operator = (Size_&& sz);
    //! the area (width*height)
    _Tp area() const;
    //! aspect ratio (width/height)
    double aspectRatio() const;
    //! true if empty
    bool empty() const;

    //! conversion of another data type.
    template<typename _Tp2> operator Size_<_Tp2>() const;

    _Tp width; //!< the width
    _Tp height; //!< the height
};

typedef Size_<int> Size2i;
typedef Size_<int64_t> Size2l;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

template<typename _Tp> inline
Size_<_Tp>::Size_()
    : width(0), height(0) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(_Tp _width, _Tp _height)
    : width(_width), height(_height) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(const Size_& sz)
    : width(sz.width), height(sz.height) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(Size_&& sz)
    : width(std::move(sz.width)), height(std::move(sz.height)) {}

template<typename _Tp> template<typename _Tp2> inline
Size_<_Tp>::operator Size_<_Tp2>() const
{
    return Size_<_Tp2>(static_cast<_Tp2>(width), static_cast<_Tp2>(height));
}

template<typename _Tp> inline
Size_<_Tp>& Size_<_Tp>::operator = (const Size_<_Tp>& sz)
{
    width = sz.width; height = sz.height;
    return *this;
}

template<typename _Tp> inline
Size_<_Tp>& Size_<_Tp>::operator = (Size_<_Tp>&& sz)
{
    width = std::move(sz.width); height = std::move(sz.height);
    return *this;
}

template<typename _Tp> inline
_Tp Size_<_Tp>::area() const
{
    return width * height;
}

template<typename _Tp> inline
bool Size_<_Tp>::empty() const
{
    return width <= 0 || height <= 0;
}

template<typename _Tp> static inline
Size_<_Tp>& operator *= (Size_<_Tp>& a, _Tp b)
{
    a.width *= b;
    a.height *= b;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator * (const Size_<_Tp>& a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp *= b;
    return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator /= (Size_<_Tp>& a, _Tp b)
{
    a.width /= b;
    a.height /= b;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator / (const Size_<_Tp>& a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator += (Size_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator + (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    Size_<_Tp> tmp(a);
    tmp += b;
    return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator -= (Size_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width -= b.width;
    a.height -= b.height;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator - (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    Size_<_Tp> tmp(a);
    tmp -= b;
    return tmp;
}

template<typename _Tp> static inline
bool operator == (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    return a.width == b.width && a.height == b.height;
}

template<typename _Tp> static inline
bool operator != (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    return !(a == b);
}
// Size End
// Point Start
template<typename _Tp> class Point_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Point_();
    Point_(_Tp _x, _Tp _y);
    Point_(const Point_& pt);
    Point_(Point_&& pt);
    Point_(const Size_<_Tp>& sz);

    Point_& operator = (const Point_& pt);
    Point_& operator = (Point_&& pt);
    template<typename _Tp2> operator Point_<_Tp2>() const;

    _Tp x; //!< x coordinate of the point
    _Tp y; //!< y coordinate of the point
};

typedef Point_<int> Point2i;
typedef Point_<int64_t> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;

template<typename _Tp> inline
Point_<_Tp>::Point_()
    : x(0), y(0) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(_Tp _x, _Tp _y)
    : x(_x), y(_y) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(const Point_& pt)
    : x(pt.x), y(pt.y) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(Point_&& pt)
    : x(std::move(pt.x)), y(std::move(pt.y)) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(const Size_<_Tp>& sz)
    : x(sz.width), y(sz.height) {}

template<typename _Tp> inline
Point_<_Tp>& Point_<_Tp>::operator = (const Point_& pt)
{
    x = pt.x; y = pt.y;
    return *this;
}

template<typename _Tp> inline
Point_<_Tp>& Point_<_Tp>::operator = (Point_&& pt)
{
    x = std::move(pt.x); y = std::move(pt.y);
    return *this;
}

template<typename _Tp> template<typename _Tp2> inline
Point_<_Tp>::operator Point_<_Tp2>() const
{
    return Point_<_Tp2>(static_cast<_Tp2>(x), static_cast<_Tp2>(y));
}

template<typename _Tp> static inline
Point_<_Tp>& operator += (Point_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename _Tp> static inline
Point_<_Tp> operator - (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( static_cast<_Tp>(a.x - b.x), static_cast<_Tp>(a.y - b.y) );
}

template<typename _Tp> static inline
bool operator != (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return a.x != b.x || a.y != b.y;
}
// Point End
// Rect Start
template<typename _Tp> class Rect_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
    Rect_(const Rect_& r);
    Rect_(Rect_&& r);
    Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz);
    Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2);

    Rect_& operator = ( const Rect_& r );
    Rect_& operator = ( Rect_&& r );
    //! the top-left corner
    Point_<_Tp> tl() const;
    //! the bottom-right corner
    Point_<_Tp> br() const;

    //! size (width, height) of the rectangle
    Size_<_Tp> size() const;
    //! area (width*height) of the rectangle
    _Tp area() const;
    //! true if empty
    bool empty() const;

    _Tp x; //!< x coordinate of the top-left corner
    _Tp y; //!< y coordinate of the top-left corner
    _Tp width; //!< width of the rectangle
    _Tp height; //!< height of the rectangle
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;

template<typename _Tp> inline
Rect_<_Tp>::Rect_()
    : x(0), y(0), width(0), height(0) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    : x(_x), y(_y), width(_width), height(_height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Rect_<_Tp>& r)
    : x(r.x), y(r.y), width(r.width), height(r.height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(Rect_<_Tp>&& r)
    : x(std::move(r.x)), y(std::move(r.y)), width(std::move(r.width)), height(std::move(r.height)) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz)
    : x(org.x), y(org.y), width(sz.width), height(sz.height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2)
{
    x = std::min(pt1.x, pt2.x);
    y = std::min(pt1.y, pt2.y);
    width = std::max(pt1.x, pt2.x) - x;
    height = std::max(pt1.y, pt2.y) - y;
}

template<typename _Tp> inline
Rect_<_Tp>& Rect_<_Tp>::operator = ( const Rect_<_Tp>& r )
{
    x = r.x;
    y = r.y;
    width = r.width;
    height = r.height;
    return *this;
}

template<typename _Tp> inline
Rect_<_Tp>& Rect_<_Tp>::operator = ( Rect_<_Tp>&& r )
{
    x = std::move(r.x);
    y = std::move(r.y);
    width = std::move(r.width);
    height = std::move(r.height);
    return *this;
}

template<typename _Tp> inline
Point_<_Tp> Rect_<_Tp>::tl() const
{
    return Point_<_Tp>(x,y);
}

template<typename _Tp> inline
Point_<_Tp> Rect_<_Tp>::br() const
{
    return Point_<_Tp>(x + width, y + height);
}

template<typename _Tp> inline
Size_<_Tp> Rect_<_Tp>::size() const
{
    return Size_<_Tp>(width, height);
}

template<typename _Tp> inline
_Tp Rect_<_Tp>::area() const
{
    const _Tp result = width * height;
    return result;
}

template<typename _Tp> inline
bool Rect_<_Tp>::empty() const
{
    return width <= 0 || height <= 0;
}
// Rect
// Scalar Start
template<typename _Tp> class Scalar_ {
public:
    //! default constructor
    Scalar_();
    Scalar_(_Tp _r, _Tp _g, _Tp _b) {
        val[0] = _r;
        val[1] = _g;
        val[2] = _b;
        val[3] = 255;
    };
    Scalar_(_Tp _r, _Tp _g, _Tp _b, _Tp _a) {
        val[0] = _r;
        val[1] = _g;
        val[2] = _b;
        val[3] = _a;
    };
    _Tp val[4];
};
typedef Scalar_<double> Scalar;
// Scalar End

static void getVARPSize(VARP var, int* height, int* width, int* channel) {
    auto info = var->getInfo();
    auto dims = info->dim;
    int num = dims.size();
    if (num < 2) return;
    if (num == 2) {
        *height  = dims[0];
        *width   = dims[1];
        *channel = 1;
    } else if (num == 3) {
        *height  = dims[0];
        *width   = dims[1];
        *channel = dims[2];
    } else if (info->order == NHWC) {
        *channel = dims[num - 1];
        *width   = dims[num - 2];
        *height  = dims[num - 3];
    } else { // NCHW
        *width   = dims[num - 1];
        *height  = dims[num - 2];
        *channel = dims[num - 3];
    }
}
static int getVARPHeight(VARP var) {
    int h, w, c;
    getVARPSize(var, &h, &w, &c);
    return h;
}
static int getVARPWidth(VARP var) {
    int h, w, c;
    getVARPSize(var, &h, &w, &c);
    return w;
}
static int getVARPChannel(VARP var) {
    int h, w, c;
    getVARPSize(var, &h, &w, &c);
    return c;
}
static int getVARPByte(VARP var) {
    return var->getInfo()->type.bytes();
}
} // CV
} // MNN
#endif // TYPES_HPP
