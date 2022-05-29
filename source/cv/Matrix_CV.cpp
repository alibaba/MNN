/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <math.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <MNN/Matrix.h>
#include "cv/SkNx.h"

namespace MNN {
namespace CV {
static inline float sk_ieee_float_divide(float x, float y) {
    return x / y;
}
static inline bool checkForZero(float x) {
    return x * x == 0;
}

static inline int32_t SkScalarAs2sCompliment(float xFloat) {
    union SkFloatIntUnion {
        float fFloat;
        int32_t fSignBitInt;
    };
    SkFloatIntUnion xu;
    xu.fFloat = xFloat;
    int32_t x = xu.fSignBitInt;
    if (x < 0) {
        x &= 0x7FFFFFFF;
        x = -x;
    }
    return x;
}
#define SK_ScalarPI 3.14159265f
#define SkDegreesToRadians(degrees) ((degrees) * (SK_ScalarPI / 180))
#define SkRadiansToDegrees(radians) ((radians) * (180 / SK_ScalarPI))
#define SK_ScalarNearlyZero (1.0f / (1 << 12))
// In a few places, we performed the following
//      a * b + c * d + e
// as
//      a * b + (c * d + e)
//
// sdot and scross are indended to capture these compound operations into a
// function, with an eye toward considering upscaling the intermediates to
// doubles for more precision (as we do in concat and invert).
//
// However, these few lines that performed the last add before the "dot", cause
// tiny image differences, so we guard that change until we see the impact on
// chrome's layouttests.
//
#define SK_LEGACY_MATRIX_MATH_ORDER
#define floatInvert(x) (1.0f / (x))

/*      [scale-x    skew-x      trans-x]   [X]   [X']
        [skew-y     scale-y     trans-y] * [Y] = [Y']
        [persp-0    persp-1     persp-2]   [1]   [1 ]
*/

void Matrix::reset() {
    fMat[kMScaleX] = fMat[kMScaleY] = fMat[kMPersp2] = 1;
    fMat[kMSkewX] = fMat[kMSkewY] = fMat[kMTransX] = fMat[kMTransY] = fMat[kMPersp0] = fMat[kMPersp1] = 0;
    this->setTypeMask(kIdentity_Mask | kRectStaysRect_Mask);
}

void Matrix::set9(const float buffer[]) {
    memcpy(fMat, buffer, 9 * sizeof(float));

    this->setTypeMask(kUnknown_Mask);
}

void Matrix::setAffine(const float buffer[]) {
    fMat[kMScaleX] = buffer[kAScaleX];
    fMat[kMSkewX]  = buffer[kASkewX];
    fMat[kMTransX] = buffer[kATransX];
    fMat[kMSkewY]  = buffer[kASkewY];
    fMat[kMScaleY] = buffer[kAScaleY];
    fMat[kMTransY] = buffer[kATransY];
    fMat[kMPersp0] = 0;
    fMat[kMPersp1] = 0;
    fMat[kMPersp2] = 1;
    this->setTypeMask(kUnknown_Mask);
}

// this guy aligns with the masks, so we can compute a mask from a varaible 0/1
enum { kTranslate_Shift = 0, kScale_Shift, kAffine_Shift, kPerspective_Shift, kRectStaysRect_Shift };

static const int32_t kScalar1Int = 0x3f800000;

///////////////////////////////////////////////////////////////////////////////

bool operator==(const Matrix& a, const Matrix& b) {
    const float* ma = a.fMat;
    const float* mb = b.fMat;

    return ma[0] == mb[0] && ma[1] == mb[1] && ma[2] == mb[2] && ma[3] == mb[3] && ma[4] == mb[4] && ma[5] == mb[5] &&
           ma[6] == mb[6] && ma[7] == mb[7] && ma[8] == mb[8];
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

static inline float sdot(float a, float b, float c, float d) {
    return a * b + c * d;
}

static inline float scross(float a, float b, float c, float d) {
    return a * b - c * d;
}

void Matrix::setTranslate(float dx, float dy) {
    if ((dx != 0) | (dy != 0)) {
        fMat[kMTransX] = dx;
        fMat[kMTransY] = dy;

        fMat[kMScaleX] = fMat[kMScaleY] = fMat[kMPersp2] = 1;
        fMat[kMSkewX] = fMat[kMSkewY] = fMat[kMPersp0] = fMat[kMPersp1] = 0;

        this->setTypeMask(kTranslate_Mask | kRectStaysRect_Mask);
    } else {
        this->reset();
    }
}

void Matrix::preTranslate(float dx, float dy) {
    const unsigned mask = this->getType();

    if (mask <= kTranslate_Mask) {
        fMat[kMTransX] += dx;
        fMat[kMTransY] += dy;
    } else if (mask & kPerspective_Mask) {
        Matrix m;
        m.setTranslate(dx, dy);
        this->preConcat(m);
        return;
    } else {
        fMat[kMTransX] += sdot(fMat[kMScaleX], dx, fMat[kMSkewX], dy);
        fMat[kMTransY] += sdot(fMat[kMSkewY], dx, fMat[kMScaleY], dy);
    }
    this->updateTranslateMask();
}

void Matrix::postTranslate(float dx, float dy) {
    Matrix m;
    m.setTranslate(dx, dy);
    this->postConcat(m);
}

///////////////////////////////////////////////////////////////////////////////

void Matrix::setScale(float sx, float sy, float px, float py) {
    if (1 == sx && 1 == sy) {
        this->reset();
    } else {
        this->setScaleTranslate(sx, sy, px - sx * px, py - sy * py);
    }
}

void Matrix::setScale(float sx, float sy) {
    if (1 == sx && 1 == sy) {
        this->reset();
    } else {
        fMat[kMScaleX] = sx;
        fMat[kMScaleY] = sy;
        fMat[kMPersp2] = 1;

        fMat[kMTransX] = fMat[kMTransY] = fMat[kMSkewX] = fMat[kMSkewY] = fMat[kMPersp0] = fMat[kMPersp1] = 0;

        this->setTypeMask(kScale_Mask | kRectStaysRect_Mask);
    }
}

void Matrix::preScale(float sx, float sy, float px, float py) {
    if (1 == sx && 1 == sy) {
        return;
    }

    Matrix m;
    m.setScale(sx, sy, px, py);
    this->preConcat(m);
}

void Matrix::preScale(float sx, float sy) {
    if (1 == sx && 1 == sy) {
        return;
    }

    // the assumption is that these multiplies are very cheap, and that
    // a full concat and/or just computing the matrix type is more expensive.
    // Also, the fixed-point case checks for overflow, but the float doesn't,
    // so we can get away with these blind multiplies.

    fMat[kMScaleX] *= sx;
    fMat[kMSkewY] *= sx;
    fMat[kMPersp0] *= sx;

    fMat[kMSkewX] *= sy;
    fMat[kMScaleY] *= sy;
    fMat[kMPersp1] *= sy;

    // Attempt to simplify our type when applying an inverse scale.
    // TODO: The persp/affine preconditions are in place to keep the mask consistent with
    //       what computeTypeMask() would produce (persp/skew always implies kScale).
    //       We should investigate whether these flag dependencies are truly needed.
    if (fMat[kMScaleX] == 1 && fMat[kMScaleY] == 1 && !(fTypeMask & (kPerspective_Mask | kAffine_Mask))) {
        this->clearTypeMask(kScale_Mask);
    } else {
        this->orTypeMask(kScale_Mask);
    }
}

void Matrix::postScale(float sx, float sy, float px, float py) {
    if (1 == sx && 1 == sy) {
        return;
    }
    Matrix m;
    m.setScale(sx, sy, px, py);
    this->postConcat(m);
}

void Matrix::postScale(float sx, float sy) {
    if (1 == sx && 1 == sy) {
        return;
    }
    Matrix m;
    m.setScale(sx, sy);
    this->postConcat(m);
}

// this guy perhaps can go away, if we have a fract/high-precision way to
// scale matrices
bool Matrix::postIDiv(int divx, int divy) {
    if (divx == 0 || divy == 0) {
        return false;
    }

    const float invX = 1.f / divx;
    const float invY = 1.f / divy;

    fMat[kMScaleX] *= invX;
    fMat[kMSkewX] *= invX;
    fMat[kMTransX] *= invX;

    fMat[kMScaleY] *= invY;
    fMat[kMSkewY] *= invY;
    fMat[kMTransY] *= invY;

    this->setTypeMask(kUnknown_Mask);
    return true;
}

////////////////////////////////////////////////////////////////////////////////////

void Matrix::setSinCos(float sinV, float cosV, float px, float py) {
    const float oneMinusCosV = 1 - cosV;
    // MNN_PRINT("%f, %f, \n", sinV, cosV);

    fMat[kMScaleX] = cosV;
    fMat[kMSkewX]  = -sinV;
    fMat[kMTransX] = sdot(sinV, py, oneMinusCosV, px);

    fMat[kMSkewY]  = sinV;
    fMat[kMScaleY] = cosV;
    fMat[kMTransY] = sdot(-sinV, px, oneMinusCosV, py);

    fMat[kMPersp0] = fMat[kMPersp1] = 0;
    fMat[kMPersp2]                  = 1;

    this->setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::setSinCos(float sinV, float cosV) {
    fMat[kMScaleX] = cosV;
    fMat[kMSkewX]  = -sinV;
    fMat[kMTransX] = 0;

    fMat[kMSkewY]  = sinV;
    fMat[kMScaleY] = cosV;
    fMat[kMTransY] = 0;

    fMat[kMPersp0] = fMat[kMPersp1] = 0;
    fMat[kMPersp2]                  = 1;

    this->setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::setRotate(float degrees, float px, float py) {
    auto rad = SkDegreesToRadians(degrees);
    float sinV, cosV;
    sinV = sin(rad);
    cosV = cos(rad);
    this->setSinCos(sinV, cosV, px, py);
}

void Matrix::setRotate(float degrees) {
    auto rad = SkDegreesToRadians(degrees);
    float sinV, cosV;
    sinV = sin(rad);
    cosV = cos(rad);
    this->setSinCos(sinV, cosV);
}

void Matrix::preRotate(float degrees, float px, float py) {
    Matrix m;
    m.setRotate(degrees, px, py);
    this->preConcat(m);
}

void Matrix::preRotate(float degrees) {
    Matrix m;
    m.setRotate(degrees);
    this->preConcat(m);
}

void Matrix::postRotate(float degrees, float px, float py) {
    Matrix m;
    m.setRotate(degrees, px, py);
    this->postConcat(m);
}

void Matrix::postRotate(float degrees) {
    Matrix m;
    m.setRotate(degrees);
    this->postConcat(m);
}

////////////////////////////////////////////////////////////////////////////////////

void Matrix::setSkew(float sx, float sy, float px, float py) {
    fMat[kMScaleX] = 1;
    fMat[kMSkewX]  = sx;
    fMat[kMTransX] = -sx * py;

    fMat[kMSkewY]  = sy;
    fMat[kMScaleY] = 1;
    fMat[kMTransY] = -sy * px;

    fMat[kMPersp0] = fMat[kMPersp1] = 0;
    fMat[kMPersp2]                  = 1;

    this->setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::setSkew(float sx, float sy) {
    fMat[kMScaleX] = 1;
    fMat[kMSkewX]  = sx;
    fMat[kMTransX] = 0;

    fMat[kMSkewY]  = sy;
    fMat[kMScaleY] = 1;
    fMat[kMTransY] = 0;

    fMat[kMPersp0] = fMat[kMPersp1] = 0;
    fMat[kMPersp2]                  = 1;

    this->setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
}

void Matrix::preSkew(float sx, float sy, float px, float py) {
    Matrix m;
    m.setSkew(sx, sy, px, py);
    this->preConcat(m);
}

void Matrix::preSkew(float sx, float sy) {
    Matrix m;
    m.setSkew(sx, sy);
    this->preConcat(m);
}

void Matrix::postSkew(float sx, float sy, float px, float py) {
    Matrix m;
    m.setSkew(sx, sy, px, py);
    this->postConcat(m);
}

void Matrix::postSkew(float sx, float sy) {
    Matrix m;
    m.setSkew(sx, sy);
    this->postConcat(m);
}

///////////////////////////////////////////////////////////////////////////////

bool Matrix::setRectToRect(const Rect& src, const Rect& dst, ScaleToFit align) {
    if (src.isEmpty()) {
        this->reset();
        return false;
    }

    if (dst.isEmpty()) {
        ::memset(fMat, 0, 8 * sizeof(float));
        fMat[kMPersp2] = 1;
        this->setTypeMask(kScale_Mask | kRectStaysRect_Mask);
    } else {
        float tx, sx = dst.width() / src.width();
        float ty, sy = dst.height() / src.height();
        bool xLarger = false;

        if (align != kFill_ScaleToFit) {
            if (sx > sy) {
                xLarger = true;
                sx      = sy;
            } else {
                sy = sx;
            }
        }

        tx = dst.fLeft - src.fLeft * sx;
        ty = dst.fTop - src.fTop * sy;
        if (align == kCenter_ScaleToFit || align == kEnd_ScaleToFit) {
            float diff;

            if (xLarger) {
                diff = dst.width() - src.width() * sy;
            } else {
                diff = dst.height() - src.height() * sy;
            }

            if (align == kCenter_ScaleToFit) {
                diff = 0.5f * (diff);
            }

            if (xLarger) {
                tx += diff;
            } else {
                ty += diff;
            }
        }

        this->setScaleTranslate(sx, sy, tx, ty);
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////

static inline float muladdmul(float a, float b, float c, float d) {
    return (float)((double)a * b + (double)c * d);
}

static inline float rowcol3(const float row[], const float col[]) {
    return row[0] * col[0] + row[1] * col[3] + row[2] * col[6];
}

static bool only_scale_and_translate(unsigned mask) {
    return 0 == (mask & (Matrix::kAffine_Mask | Matrix::kPerspective_Mask));
}

void Matrix::setConcat(const Matrix& a, const Matrix& b) {
    TypeMask aType = a.getType();
    TypeMask bType = b.getType();

    if (a.isTriviallyIdentity()) {
        *this = b;
    } else if (b.isTriviallyIdentity()) {
        *this = a;
    } else if (only_scale_and_translate(aType | bType)) {
        this->setScaleTranslate(a.fMat[kMScaleX] * b.fMat[kMScaleX], a.fMat[kMScaleY] * b.fMat[kMScaleY],
                                a.fMat[kMScaleX] * b.fMat[kMTransX] + a.fMat[kMTransX],
                                a.fMat[kMScaleY] * b.fMat[kMTransY] + a.fMat[kMTransY]);
    } else {
        Matrix tmp;

        if ((aType | bType) & kPerspective_Mask) {
            tmp.fMat[kMScaleX] = rowcol3(&a.fMat[0], &b.fMat[0]);
            tmp.fMat[kMSkewX]  = rowcol3(&a.fMat[0], &b.fMat[1]);
            tmp.fMat[kMTransX] = rowcol3(&a.fMat[0], &b.fMat[2]);
            tmp.fMat[kMSkewY]  = rowcol3(&a.fMat[3], &b.fMat[0]);
            tmp.fMat[kMScaleY] = rowcol3(&a.fMat[3], &b.fMat[1]);
            tmp.fMat[kMTransY] = rowcol3(&a.fMat[3], &b.fMat[2]);
            tmp.fMat[kMPersp0] = rowcol3(&a.fMat[6], &b.fMat[0]);
            tmp.fMat[kMPersp1] = rowcol3(&a.fMat[6], &b.fMat[1]);
            tmp.fMat[kMPersp2] = rowcol3(&a.fMat[6], &b.fMat[2]);

            tmp.setTypeMask(kUnknown_Mask);
        } else {
            tmp.fMat[kMScaleX] = muladdmul(a.fMat[kMScaleX], b.fMat[kMScaleX], a.fMat[kMSkewX], b.fMat[kMSkewY]);

            tmp.fMat[kMSkewX] = muladdmul(a.fMat[kMScaleX], b.fMat[kMSkewX], a.fMat[kMSkewX], b.fMat[kMScaleY]);

            tmp.fMat[kMTransX] =
                muladdmul(a.fMat[kMScaleX], b.fMat[kMTransX], a.fMat[kMSkewX], b.fMat[kMTransY]) + a.fMat[kMTransX];

            tmp.fMat[kMSkewY] = muladdmul(a.fMat[kMSkewY], b.fMat[kMScaleX], a.fMat[kMScaleY], b.fMat[kMSkewY]);

            tmp.fMat[kMScaleY] = muladdmul(a.fMat[kMSkewY], b.fMat[kMSkewX], a.fMat[kMScaleY], b.fMat[kMScaleY]);

            tmp.fMat[kMTransY] =
                muladdmul(a.fMat[kMSkewY], b.fMat[kMTransX], a.fMat[kMScaleY], b.fMat[kMTransY]) + a.fMat[kMTransY];

            tmp.fMat[kMPersp0] = 0;
            tmp.fMat[kMPersp1] = 0;
            tmp.fMat[kMPersp2] = 1;
            // SkDebugf("Concat mat non-persp type: %d\n", tmp.getType());
            // MNN_ASSERT(!(tmp.getType() & kPerspective_Mask));
            tmp.setTypeMask(kUnknown_Mask | kOnlyPerspectiveValid_Mask);
        }
        *this = tmp;
    }
}

void Matrix::preConcat(const Matrix& mat) {
    // check for identity first, so we don't do a needless copy of ourselves
    // to ourselves inside setConcat()
    if (!mat.isIdentity()) {
        this->setConcat(*this, mat);
    }
}

void Matrix::postConcat(const Matrix& mat) {
    // check for identity first, so we don't do a needless copy of ourselves
    // to ourselves inside setConcat()
    if (!mat.isIdentity()) {
        this->setConcat(mat, *this);
    }
}

///////////////////////////////////////////////////////////////////////////////

/*  Matrix inversion is very expensive, but also the place where keeping
    precision may be most important (here and matrix concat). Hence to avoid
    bitmap blitting artifacts when walking the inverse, we use doubles for
    the intermediate math, even though we know that is more expensive.
 */

static inline float scross_dscale(float a, float b, float c, float d, double scale) {
    return (float)(scross(a, b, c, d) * scale);
}

static inline double dcross(double a, double b, double c, double d) {
    return a * b - c * d;
}

static inline float dcross_dscale(double a, double b, double c, double d, double scale) {
    return (float)(dcross(a, b, c, d) * scale);
}

static double sk_inv_determinant(const float mat[9], int isPerspective) {
    double det;

    if (isPerspective) {
        det = mat[Matrix::kMScaleX] *
                  dcross(mat[Matrix::kMScaleY], mat[Matrix::kMPersp2], mat[Matrix::kMTransY], mat[Matrix::kMPersp1]) +
              mat[Matrix::kMSkewX] *
                  dcross(mat[Matrix::kMTransY], mat[Matrix::kMPersp0], mat[Matrix::kMSkewY], mat[Matrix::kMPersp2]) +
              mat[Matrix::kMTransX] *
                  dcross(mat[Matrix::kMSkewY], mat[Matrix::kMPersp1], mat[Matrix::kMScaleY], mat[Matrix::kMPersp0]);
    } else {
        det = dcross(mat[Matrix::kMScaleX], mat[Matrix::kMScaleY], mat[Matrix::kMSkewX], mat[Matrix::kMSkewY]);
    }

    return 1.0 / det;
}

void Matrix::SetAffineIdentity(float affine[6]) {
    affine[kAScaleX] = 1;
    affine[kASkewY]  = 0;
    affine[kASkewX]  = 0;
    affine[kAScaleY] = 1;
    affine[kATransX] = 0;
    affine[kATransY] = 0;
}

bool Matrix::asAffine(float affine[6]) const {
    if (affine) {
        affine[kAScaleX] = this->fMat[kMScaleX];
        affine[kASkewY]  = this->fMat[kMSkewY];
        affine[kASkewX]  = this->fMat[kMSkewX];
        affine[kAScaleY] = this->fMat[kMScaleY];
        affine[kATransX] = this->fMat[kMTransX];
        affine[kATransY] = this->fMat[kMTransY];
    }
    return true;
}

void Matrix::ComputeInv(float dst[9], const float src[9], double invDet, bool isPersp) {
    MNN_ASSERT(src != dst);
    MNN_ASSERT(src && dst);

    if (isPersp) {
        dst[kMScaleX] = scross_dscale(src[kMScaleY], src[kMPersp2], src[kMTransY], src[kMPersp1], invDet);
        dst[kMSkewX]  = scross_dscale(src[kMTransX], src[kMPersp1], src[kMSkewX], src[kMPersp2], invDet);
        dst[kMTransX] = scross_dscale(src[kMSkewX], src[kMTransY], src[kMTransX], src[kMScaleY], invDet);

        dst[kMSkewY]  = scross_dscale(src[kMTransY], src[kMPersp0], src[kMSkewY], src[kMPersp2], invDet);
        dst[kMScaleY] = scross_dscale(src[kMScaleX], src[kMPersp2], src[kMTransX], src[kMPersp0], invDet);
        dst[kMTransY] = scross_dscale(src[kMTransX], src[kMSkewY], src[kMScaleX], src[kMTransY], invDet);

        dst[kMPersp0] = scross_dscale(src[kMSkewY], src[kMPersp1], src[kMScaleY], src[kMPersp0], invDet);
        dst[kMPersp1] = scross_dscale(src[kMSkewX], src[kMPersp0], src[kMScaleX], src[kMPersp1], invDet);
        dst[kMPersp2] = scross_dscale(src[kMScaleX], src[kMScaleY], src[kMSkewX], src[kMSkewY], invDet);
    } else { // not perspective
        dst[kMScaleX] = (float)(src[kMScaleY] * invDet);
        dst[kMSkewX]  = (float)(-src[kMSkewX] * invDet);
        dst[kMTransX] = dcross_dscale(src[kMSkewX], src[kMTransY], src[kMScaleY], src[kMTransX], invDet);

        dst[kMSkewY]  = (float)(-src[kMSkewY] * invDet);
        dst[kMScaleY] = (float)(src[kMScaleX] * invDet);
        dst[kMTransY] = dcross_dscale(src[kMSkewY], src[kMTransX], src[kMScaleX], src[kMTransY], invDet);

        dst[kMPersp0] = 0;
        dst[kMPersp1] = 0;
        dst[kMPersp2] = 1;
    }
}

bool Matrix::invertNonIdentity(Matrix* inv) const {
    MNN_ASSERT(!this->isIdentity());

    TypeMask mask = this->getType();

    if (0 == (mask & ~(kScale_Mask | kTranslate_Mask))) {
        bool invertible = true;
        if (inv) {
            if (mask & kScale_Mask) {
                float invX = fMat[kMScaleX];
                float invY = fMat[kMScaleY];
                if (0 == invX || 0 == invY) {
                    return false;
                }
                invX = floatInvert(invX);
                invY = floatInvert(invY);

                // Must be careful when writing to inv, since it may be the
                // same memory as this.

                inv->fMat[kMSkewX] = inv->fMat[kMSkewY] = inv->fMat[kMPersp0] = inv->fMat[kMPersp1] = 0;

                inv->fMat[kMScaleX] = invX;
                inv->fMat[kMScaleY] = invY;
                inv->fMat[kMPersp2] = 1;
                inv->fMat[kMTransX] = -fMat[kMTransX] * invX;
                inv->fMat[kMTransY] = -fMat[kMTransY] * invY;

                inv->setTypeMask(mask | kRectStaysRect_Mask);
            } else {
                // translate only
                inv->setTranslate(-fMat[kMTransX], -fMat[kMTransY]);
            }
        } else { // inv is nullptr, just check if we're invertible
            if (!fMat[kMScaleX] || !fMat[kMScaleY]) {
                invertible = false;
            }
        }
        return invertible;
    }

    int isPersp   = mask & kPerspective_Mask;
    double invDet = sk_inv_determinant(fMat, isPersp);

    if (invDet == 0) { // underflow
        return false;
    }

    bool applyingInPlace = (inv == this);

    Matrix* tmp = inv;

    Matrix storage;
    if (applyingInPlace || nullptr == tmp) {
        tmp = &storage; // we either need to avoid trampling memory or have no memory
    }

    ComputeInv(tmp->fMat, fMat, invDet, isPersp);
    tmp->setTypeMask(fTypeMask);

    if (applyingInPlace) {
        *inv = storage; // need to copy answer back
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////

void Matrix::Identity_pts(const Matrix& m, Point dst[], const Point src[], int count) {
    MNN_ASSERT(m.getType() == 0);

    if (dst != src && count > 0) {
        memcpy(dst, src, count * sizeof(Point));
    }
}

void Matrix::Trans_pts(const Matrix& m, Point dst[], const Point src[], int count) {
    MNN_ASSERT(m.getType() <= Matrix::kTranslate_Mask);
    if (count > 0) {
        float tx = m.getTranslateX();
        float ty = m.getTranslateY();
        if (count & 1) {
            dst->fX = src->fX + tx;
            dst->fY = src->fY + ty;
            src += 1;
            dst += 1;
        }
        Sk4s trans4(tx, ty, tx, ty);
        count >>= 1;
        if (count & 1) {
            (Sk4s::Load(src) + trans4).store(dst);
            src += 2;
            dst += 2;
        }
        count >>= 1;
        for (int i = 0; i < count; ++i) {
            (Sk4s::Load(src + 0) + trans4).store(dst + 0);
            (Sk4s::Load(src + 2) + trans4).store(dst + 2);
            src += 4;
            dst += 4;
        }
    }
}

void Matrix::Scale_pts(const Matrix& m, Point dst[], const Point src[], int count) {
    MNN_ASSERT(m.getType() <= (Matrix::kScale_Mask | Matrix::kTranslate_Mask));
    if (count > 0) {
        float tx = m.getTranslateX();
        float ty = m.getTranslateY();
        float sx = m.getScaleX();
        float sy = m.getScaleY();
        if (count & 1) {
            dst->fX = src->fX * sx + tx;
            dst->fY = src->fY * sy + ty;
            src += 1;
            dst += 1;
        }
        Sk4s trans4(tx, ty, tx, ty);
        Sk4s scale4(sx, sy, sx, sy);
        count >>= 1;
        if (count & 1) {
            (Sk4s::Load(src) * scale4 + trans4).store(dst);
            src += 2;
            dst += 2;
        }
        count >>= 1;
        for (int i = 0; i < count; ++i) {
            (Sk4s::Load(src + 0) * scale4 + trans4).store(dst + 0);
            (Sk4s::Load(src + 2) * scale4 + trans4).store(dst + 2);
            src += 4;
            dst += 4;
        }
    }
}

void Matrix::Persp_pts(const Matrix& m, Point dst[], const Point src[], int count) {
    if (count > 0) {
        do {
            float sy = src->fY;
            float sx = src->fX;
            src += 1;

            float x = sdot(sx, m.fMat[kMScaleX], sy, m.fMat[kMSkewX]) + m.fMat[kMTransX];
            float y = sdot(sx, m.fMat[kMSkewY], sy, m.fMat[kMScaleY]) + m.fMat[kMTransY];
#ifdef SK_LEGACY_MATRIX_MATH_ORDER
            float z = sx * m.fMat[kMPersp0] + (sy * m.fMat[kMPersp1] + m.fMat[kMPersp2]);
#else
            float z = sdot(sx, m.fMat[kMPersp0], sy, m.fMat[kMPersp1]) + m.fMat[kMPersp2];
#endif
            if (z) {
                z = 1.0f / (z);
            }

            dst->fY = y * z;
            dst->fX = x * z;
            dst += 1;
        } while (--count);
    }
}

void Matrix::Affine_vpts(const Matrix& m, Point dst[], const Point src[], int count) {
    MNN_ASSERT(m.getType() != Matrix::kPerspective_Mask);
    float tx = m.getTranslateX();
    float ty = m.getTranslateY();
    float sx = m.getScaleX();
    float sy = m.getScaleY();
    float kx = m.getSkewX();
    float ky = m.getSkewY();
#ifdef MNN_USE_NEON
    if (count > 4) {
        auto tx4    = vdupq_n_f32(tx);
        auto ty4    = vdupq_n_f32(ty);
        auto sx4    = vdupq_n_f32(sx);
        auto sy4    = vdupq_n_f32(sy);
        auto kx4    = vdupq_n_f32(kx);
        auto ky4    = vdupq_n_f32(ky);
        int countC4 = count / 4;
        for (int i = 0; i < countC4; ++i) {
            const float* s = (const float*)src;
            float* d       = (float*)dst;
            auto sv        = vld2q_f32(s);
            float32x4x2_t dv;
            dv.val[0] = tx4 + sv.val[0] * sx4 + sv.val[1] * kx4;
            dv.val[1] = ty4 + sv.val[0] * ky4 + sv.val[1] * sy4;
            vst2q_f32(d, dv);

            src += 4;
            dst += 4;
        }
        count = count - 4 * countC4;
    }
#endif
    if (count > 0) {
        if (count & 1) {
            dst->set(src->fX * sx + src->fY * kx + tx, src->fX * ky + src->fY * sy + ty);
            src += 1;
            dst += 1;
        }
        Sk4s trans4(tx, ty, tx, ty);
        Sk4s scale4(sx, sy, sx, sy);
        Sk4s skew4(kx, ky, kx, ky); // applied to swizzle of src4
        count >>= 1;
        for (int i = 0; i < count; ++i) {
            Sk4s src4 = Sk4s::Load(src);
            Sk4s swz4 = SkNx_shuffle<1, 0, 3, 2>(src4); // y0 x0, y1 x1
            (src4 * scale4 + swz4 * skew4 + trans4).store(dst);
            src += 2;
            dst += 2;
        }
    }
}

const Matrix::MapPtsProc Matrix::gMapPtsProcs[] = {
    Matrix::Identity_pts, Matrix::Trans_pts, Matrix::Scale_pts, Matrix::Scale_pts, Matrix::Affine_vpts,
    Matrix::Affine_vpts, Matrix::Affine_vpts, Matrix::Affine_vpts,
    // repeat the persp proc 8 times
    Matrix::Persp_pts, Matrix::Persp_pts, Matrix::Persp_pts, Matrix::Persp_pts, Matrix::Persp_pts, Matrix::Persp_pts,
    Matrix::Persp_pts, Matrix::Persp_pts};

static Sk4f sort_as_rect(const Sk4f& ltrb) {
    Sk4f rblt(ltrb[2], ltrb[3], ltrb[0], ltrb[1]);
    Sk4f min = Sk4f::Min(ltrb, rblt);
    Sk4f max = Sk4f::Max(ltrb, rblt);
    // We can extract either pair [0,1] or [2,3] from min and max and be correct, but on
    // ARM this sequence generates the fastest (a single instruction).
    return Sk4f(min[2], min[3], max[0], max[1]);
}

void Matrix::mapRectScaleTranslate(Rect* dst, const Rect& src) const {
    MNN_ASSERT(this->isScaleTranslate());

    float sx = fMat[kMScaleX];
    float sy = fMat[kMScaleY];
    float tx = fMat[kMTransX];
    float ty = fMat[kMTransY];
    Sk4f scale(sx, sy, sx, sy);
    Sk4f trans(tx, ty, tx, ty);
    sort_as_rect(Sk4f::Load(&src.fLeft) * scale + trans).store(&dst->fLeft);
}

bool Matrix::mapRect(Rect* dst, const Rect& src) const {
    if (this->getType() <= kTranslate_Mask) {
        float tx = fMat[kMTransX];
        float ty = fMat[kMTransY];
        Sk4f trans(tx, ty, tx, ty);
        sort_as_rect(Sk4f::Load(&src.fLeft) + trans).store(&dst->fLeft);
        return true;
    }
    if (this->isScaleTranslate()) {
        this->mapRectScaleTranslate(dst, src);
        return true;
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////////

void Matrix::Persp_xy(const Matrix& m, float sx, float sy, Point* pt) {
    float x = sdot(sx, m.fMat[kMScaleX], sy, m.fMat[kMSkewX]) + m.fMat[kMTransX];
    float y = sdot(sx, m.fMat[kMSkewY], sy, m.fMat[kMScaleY]) + m.fMat[kMTransY];
    float z = sdot(sx, m.fMat[kMPersp0], sy, m.fMat[kMPersp1]) + m.fMat[kMPersp2];
    if (z) {
        z = 1.0f / (z);
    }
    pt->fX = x * z;
    pt->fY = y * z;
}

void Matrix::RotTrans_xy(const Matrix& m, float sx, float sy, Point* pt) {
    MNN_ASSERT((m.getType() & (kAffine_Mask | kPerspective_Mask)) == kAffine_Mask);

#ifdef SK_LEGACY_MATRIX_MATH_ORDER
    pt->fX = sx * m.fMat[kMScaleX] + (sy * m.fMat[kMSkewX] + m.fMat[kMTransX]);
    pt->fY = sx * m.fMat[kMSkewY] + (sy * m.fMat[kMScaleY] + m.fMat[kMTransY]);
#else
    pt->fX = sdot(sx, m.fMat[kMScaleX], sy, m.fMat[kMSkewX]) + m.fMat[kMTransX];
    pt->fY = sdot(sx, m.fMat[kMSkewY], sy, m.fMat[kMScaleY]) + m.fMat[kMTransY];
#endif
}

void Matrix::Rot_xy(const Matrix& m, float sx, float sy, Point* pt) {
    MNN_ASSERT((m.getType() & (kAffine_Mask | kPerspective_Mask)) == kAffine_Mask);
    MNN_ASSERT(0 == m.fMat[kMTransX]);
    MNN_ASSERT(0 == m.fMat[kMTransY]);

#ifdef SK_LEGACY_MATRIX_MATH_ORDER
    pt->fX = sx * m.fMat[kMScaleX] + (sy * m.fMat[kMSkewX] + m.fMat[kMTransX]);
    pt->fY = sx * m.fMat[kMSkewY] + (sy * m.fMat[kMScaleY] + m.fMat[kMTransY]);
#else
    pt->fX = sdot(sx, m.fMat[kMScaleX], sy, m.fMat[kMSkewX]) + m.fMat[kMTransX];
    pt->fY = sdot(sx, m.fMat[kMSkewY], sy, m.fMat[kMScaleY]) + m.fMat[kMTransY];
#endif
}

void Matrix::ScaleTrans_xy(const Matrix& m, float sx, float sy, Point* pt) {
    MNN_ASSERT((m.getType() & (kScale_Mask | kAffine_Mask | kPerspective_Mask)) == kScale_Mask);

    pt->fX = sx * m.fMat[kMScaleX] + m.fMat[kMTransX];
    pt->fY = sy * m.fMat[kMScaleY] + m.fMat[kMTransY];
}

void Matrix::Scale_xy(const Matrix& m, float sx, float sy, Point* pt) {
    MNN_ASSERT((m.getType() & (kScale_Mask | kAffine_Mask | kPerspective_Mask)) == kScale_Mask);
    MNN_ASSERT(0 == m.fMat[kMTransX]);
    MNN_ASSERT(0 == m.fMat[kMTransY]);

    pt->fX = sx * m.fMat[kMScaleX];
    pt->fY = sy * m.fMat[kMScaleY];
}

void Matrix::Trans_xy(const Matrix& m, float sx, float sy, Point* pt) {
    MNN_ASSERT(m.getType() == kTranslate_Mask);

    pt->fX = sx + m.fMat[kMTransX];
    pt->fY = sy + m.fMat[kMTransY];
}

void Matrix::Identity_xy(const Matrix& m, float sx, float sy, Point* pt) {
    MNN_ASSERT(0 == m.getType());

    pt->fX = sx;
    pt->fY = sy;
}

const Matrix::MapXYProc Matrix::gMapXYProcs[] = {
    Matrix::Identity_xy, Matrix::Trans_xy, Matrix::Scale_xy, Matrix::ScaleTrans_xy, Matrix::Rot_xy, Matrix::RotTrans_xy,
    Matrix::Rot_xy, Matrix::RotTrans_xy,
    // repeat the persp proc 8 times
    Matrix::Persp_xy, Matrix::Persp_xy, Matrix::Persp_xy, Matrix::Persp_xy, Matrix::Persp_xy, Matrix::Persp_xy,
    Matrix::Persp_xy, Matrix::Persp_xy};

uint8_t Matrix::computeTypeMask() const {
    unsigned mask = 0;

    if (fMat[kMPersp0] != 0 || fMat[kMPersp1] != 0 || fMat[kMPersp2] != 1) {
        // Once it is determined that this is a perspective transform,
        // all other flags are moot as far as optimizations are concerned.
        return (uint8_t)(kORableMasks);
    }

    if (fMat[kMTransX] != 0 || fMat[kMTransY] != 0) {
        mask |= kTranslate_Mask;
    }

    int m00 = SkScalarAs2sCompliment(fMat[Matrix::kMScaleX]);
    int m01 = SkScalarAs2sCompliment(fMat[Matrix::kMSkewX]);
    int m10 = SkScalarAs2sCompliment(fMat[Matrix::kMSkewY]);
    int m11 = SkScalarAs2sCompliment(fMat[Matrix::kMScaleY]);

    if (m01 | m10) {
        // The skew components may be scale-inducing, unless we are dealing
        // with a pure rotation.  Testing for a pure rotation is expensive,
        // so we opt for being conservative by always setting the scale bit.
        // along with affine.
        // By doing this, we are also ensuring that matrices have the same
        // type masks as their inverses.
        mask |= kAffine_Mask | kScale_Mask;

        // For rectStaysRect, in the affine case, we only need check that
        // the primary diagonal is all zeros and that the secondary diagonal
        // is all non-zero.

        // map non-zero to 1
        m01 = m01 != 0;
        m10 = m10 != 0;

        int dp0 = 0 == (m00 | m11); // true if both are 0
        int ds1 = m01 & m10;        // true if both are 1

        mask |= (dp0 & ds1) << kRectStaysRect_Shift;
    } else {
        // Only test for scale explicitly if not affine, since affine sets the
        // scale bit.
        if ((m00 ^ kScalar1Int) | (m11 ^ kScalar1Int)) {
            mask |= kScale_Mask;
        }

        // Not affine, therefore we already know secondary diagonal is
        // all zeros, so we just need to check that primary diagonal is
        // all non-zero.

        // map non-zero to 1
        m00 = m00 != 0;
        m11 = m11 != 0;

        // record if the (p)rimary diagonal is all non-zero
        mask |= (m00 & m11) << kRectStaysRect_Shift;
    }

    // MNN_PRINT("%d\n", mask);
    return (uint8_t)(mask);
}

uint8_t Matrix::computePerspectiveTypeMask() const {
    // Benchmarking suggests that replacing this set of SkScalarAs2sCompliment
    // is a win, but replacing those below is not. We don't yet understand
    // that result.
    if (fMat[kMPersp0] != 0 || fMat[kMPersp1] != 0 || fMat[kMPersp2] != 1) {
        // If this is a perspective transform, we return true for all other
        // transform flags - this does not disable any optimizations, respects
        // the rule that the type mask must be conservative, and speeds up
        // type mask computation.
        return (uint8_t)(kORableMasks);
    }

    return (uint8_t)(kOnlyPerspectiveValid_Mask | kUnknown_Mask);
}
bool Matrix::Poly2Proc(const Point srcPt[], Matrix* dst) {
    dst->fMat[kMScaleX] = srcPt[1].fY - srcPt[0].fY;
    dst->fMat[kMSkewY]  = srcPt[0].fX - srcPt[1].fX;
    dst->fMat[kMPersp0] = 0;

    dst->fMat[kMSkewX]  = srcPt[1].fX - srcPt[0].fX;
    dst->fMat[kMScaleY] = srcPt[1].fY - srcPt[0].fY;
    dst->fMat[kMPersp1] = 0;

    dst->fMat[kMTransX] = srcPt[0].fX;
    dst->fMat[kMTransY] = srcPt[0].fY;
    dst->fMat[kMPersp2] = 1;
    dst->setTypeMask(kUnknown_Mask);
    return true;
}

bool Matrix::Poly3Proc(const Point srcPt[], Matrix* dst) {
    dst->fMat[kMScaleX] = srcPt[2].fX - srcPt[0].fX;
    dst->fMat[kMSkewY]  = srcPt[2].fY - srcPt[0].fY;
    dst->fMat[kMPersp0] = 0;

    dst->fMat[kMSkewX]  = srcPt[1].fX - srcPt[0].fX;
    dst->fMat[kMScaleY] = srcPt[1].fY - srcPt[0].fY;
    dst->fMat[kMPersp1] = 0;

    dst->fMat[kMTransX] = srcPt[0].fX;
    dst->fMat[kMTransY] = srcPt[0].fY;
    dst->fMat[kMPersp2] = 1;
    dst->setTypeMask(kUnknown_Mask);
    return true;
}
bool Matrix::Poly4Proc(const Point srcPt[], Matrix* dst) {
    float a1, a2;
    float x0, y0, x1, y1, x2, y2;

    x0 = srcPt[2].fX - srcPt[0].fX;
    y0 = srcPt[2].fY - srcPt[0].fY;
    x1 = srcPt[2].fX - srcPt[1].fX;
    y1 = srcPt[2].fY - srcPt[1].fY;
    x2 = srcPt[2].fX - srcPt[3].fX;
    y2 = srcPt[2].fY - srcPt[3].fY;

    /* check if abs(x2) > abs(y2) */
    if (x2 > 0 ? y2 > 0 ? x2 > y2 : x2 > -y2 : y2 > 0 ? -x2 > y2 : x2 < y2) {
        float denom = sk_ieee_float_divide(x1 * y2, x2) - y1;
        if (checkForZero(denom)) {
            return false;
        }
        a1 = (((x0 - x1) * y2 / x2) - y0 + y1) / denom;
    } else {
        float denom = x1 - sk_ieee_float_divide(y1 * x2, y2);
        if (checkForZero(denom)) {
            return false;
        }
        a1 = (x0 - x1 - sk_ieee_float_divide((y0 - y1) * x2, y2)) / denom;
    }

    /* check if abs(x1) > abs(y1) */
    if (x1 > 0 ? y1 > 0 ? x1 > y1 : x1 > -y1 : y1 > 0 ? -x1 > y1 : x1 < y1) {
        float denom = y2 - sk_ieee_float_divide(x2 * y1, x1);
        if (checkForZero(denom)) {
            return false;
        }
        a2 = (y0 - y2 - sk_ieee_float_divide((x0 - x2) * y1, x1)) / denom;
    } else {
        float denom = sk_ieee_float_divide(y2 * x1, y1) - x2;
        if (checkForZero(denom)) {
            return false;
        }
        a2 = (sk_ieee_float_divide((y0 - y2) * x1, y1) - x0 + x2) / denom;
    }

    dst->fMat[kMScaleX] = a2 * srcPt[3].fX + srcPt[3].fX - srcPt[0].fX;
    dst->fMat[kMSkewY]  = a2 * srcPt[3].fY + srcPt[3].fY - srcPt[0].fY;
    dst->fMat[kMPersp0] = a2;

    dst->fMat[kMSkewX]  = a1 * srcPt[1].fX + srcPt[1].fX - srcPt[0].fX;
    dst->fMat[kMScaleY] = a1 * srcPt[1].fY + srcPt[1].fY - srcPt[0].fY;
    dst->fMat[kMPersp1] = a1;

    dst->fMat[kMTransX] = srcPt[0].fX;
    dst->fMat[kMTransY] = srcPt[0].fY;
    dst->fMat[kMPersp2] = 1;
    dst->setTypeMask(kUnknown_Mask);
    return true;
}
bool Matrix::setPolyToPoly(const Point src[], const Point dst[], int count) {
    if ((unsigned)count > 4) {
        MNN_ERROR("---::setPolyToPoly count out of range %d\n", count);
        return false;
    }

    if (0 == count) {
        this->reset();
        return true;
    }
    if (1 == count) {
        this->setTranslate(dst[0].fX - src[0].fX, dst[0].fY - src[0].fY);
        return true;
    }

    typedef bool (*PolyMapProc)(const Point[], Matrix*);
    Matrix tempMap, result;
    const PolyMapProc gPolyMapProcs[] = {Matrix::Poly2Proc, Matrix::Poly3Proc, Matrix::Poly4Proc};
    auto proc                         = gPolyMapProcs[count - 2];

    if (!proc(src, &tempMap)) {
        return false;
    }
    if (!tempMap.invert(&result)) {
        return false;
    }
    if (!proc(dst, &tempMap)) {
        return false;
    }
    this->setConcat(tempMap, result);
    return true;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace CV
} // namespace MNN
