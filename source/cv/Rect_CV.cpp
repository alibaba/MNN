#include <MNN/Rect.h>

namespace MNN {
namespace CV {

void Point::set(float x, float y) {	
    fX = x;	
    fY = y;	
}

bool Rect::isEmpty() const {
    // We write it as the NOT of a non-empty rect, so we will return true if any values
    // are NaN.
    return !(fLeft < fRight && fTop < fBottom);
}
bool Rect::isSorted() const {
    return fLeft <= fRight && fTop <= fBottom;
}
void Rect::setEmpty() {
    *this = MakeEmpty();
}
void Rect::set(float left, float top, float right, float bottom) {
    fLeft   = left;
    fTop    = top;
    fRight  = right;
    fBottom = bottom;
}
void Rect::setLTRB(float left, float top, float right, float bottom) {
    this->set(left, top, right, bottom);
}
void Rect::iset(int left, int top, int right, int bottom) {
    fLeft   = (float)(left);
    fTop    = (float)(top);
    fRight  = (float)(right);
    fBottom = (float)(bottom);
}
void Rect::isetWH(int width, int height) {
    fLeft = fTop = 0;
    fRight       = (float)(width);
    fBottom      = (float)(height);
}
void Rect::setXYWH(float x, float y, float width, float height) {
    fLeft   = x;
    fTop    = y;
    fRight  = x + width;
    fBottom = y + height;
}
void Rect::setWH(float width, float height) {
    fLeft   = 0;
    fTop    = 0;
    fRight  = width;
    fBottom = height;
}

Rect Rect::makeOffset(float dx, float dy) const {
    return MakeLTRB(fLeft + dx, fTop + dy, fRight + dx, fBottom + dy);
}
Rect Rect::makeInset(float dx, float dy) const {
    return MakeLTRB(fLeft + dx, fTop + dy, fRight - dx, fBottom - dy);
}
Rect Rect::makeOutset(float dx, float dy) const {
    return MakeLTRB(fLeft - dx, fTop - dy, fRight + dx, fBottom + dy);
}
void Rect::offset(float dx, float dy) {
    fLeft += dx;
    fTop += dy;
    fRight += dx;
    fBottom += dy;
}
void Rect::offsetTo(float newX, float newY) {
    fRight += newX - fLeft;
    fBottom += newY - fTop;
    fLeft = newX;
    fTop  = newY;
}
void Rect::inset(float dx, float dy) {
    fLeft += dx;
    fTop += dy;
    fRight -= dx;
    fBottom -= dy;
}
void Rect::outset(float dx, float dy) {
    this->inset(-dx, -dy);
}

bool Rect::Intersects(float al, float at, float ar, float ab, float bl, float bt, float br, float bb) {
    float L = std::max(al, bl);
    float R = std::min(ar, br);
    float T = std::max(at, bt);
    float B = std::min(ab, bb);
    return L < R && T < B;
}

bool Rect::intersects(float left, float top, float right, float bottom) const {
    return Intersects(fLeft, fTop, fRight, fBottom, left, top, right, bottom);
}
bool Rect::intersects(const Rect& r) const {
    return Intersects(fLeft, fTop, fRight, fBottom, r.fLeft, r.fTop, r.fRight, r.fBottom);
}
bool Rect::Intersects(const Rect& a, const Rect& b) {
    return Intersects(a.fLeft, a.fTop, a.fRight, a.fBottom, b.fLeft, b.fTop, b.fRight, b.fBottom);
}
void Rect::joinNonEmptyArg(const Rect& r) {
    MNN_ASSERT(!r.isEmpty());
    // if we are empty, just assign
    if (fLeft >= fRight || fTop >= fBottom) {
        *this = r;
    } else {
        this->joinPossiblyEmptyRect(r);
    }
}
void Rect::joinPossiblyEmptyRect(const Rect& r) {
    fLeft   = std::min(fLeft, r.left());
    fTop    = std::min(fTop, r.top());
    fRight  = std::max(fRight, r.right());
    fBottom = std::max(fBottom, r.bottom());
}
bool Rect::contains(float x, float y) const {
    return x >= fLeft && x < fRight && y >= fTop && y < fBottom;
}
void Rect::sort() {
    using std::swap;
    if (fLeft > fRight) {
        swap(fLeft, fRight);
    }
    if (fTop > fBottom) {
        swap(fTop, fBottom);
    }
}
Rect Rect::makeSorted() const {
    return MakeLTRB(std::min(fLeft, fRight), std::min(fTop, fBottom), std::max(fLeft, fRight),
                    std::max(fTop, fBottom));
}
const float* Rect::asScalars() const {
    return &fLeft;
}

} // namespace CV
} // namespace MNN
