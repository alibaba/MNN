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

// Size
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

template<typename _Tp> class Scalar_ {
public:
    //! default constructor
    Scalar_();
    Scalar_(_Tp _r, _Tp _g, _Tp _b) : r(_r), g(_g), b(_b), a(255) {};
    Scalar_(_Tp _r, _Tp _g, _Tp _b, _Tp _a) : r(_r), g(_g), b(_b), a(_a) {};
    _Tp r, g, b, a;
};
typedef Scalar_<uint8_t> Scalar;

static void getVARPSize(VARP var, int* height, int* width, int* channel) {
    auto info = var->getInfo();
    auto dims = info->dim;
    int num = dims.size();
    if (num < 3) return;
    if (info->order == NHWC) {
        *channel = dims[num - 1];
        *width   = dims[num - 2];
        *height  = dims[num - 3];
    } else { // NCHW
        *width   = dims[num - 1];
        *height  = dims[num - 2];
        *channel = dims[num - 3];
    }
}

} // CV
} // MNN
#endif // TYPES_HPP
