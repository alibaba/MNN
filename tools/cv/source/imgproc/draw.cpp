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
#include <MNN/expr/MathOp.hpp>
#include <cmath>
#include <limits>

namespace MNN {
namespace CV {

// help functions
#define MIN(a,b)  ((a) > (b) ? (b) : (a))
#define MAX(a,b)  ((a) < (b) ? (b) : (a))

struct Region {
public:
    Region(int _y, int _xl, int _xr) : y(_y), xl(_xl), xr(_xr) {}
    Region(int _y, int _xl) : y(_y), xl(_xl), xr(_xl) {}
    int y;
    int xl;
    int xr;
};

bool clipLine(Size2l img_size, Point2l& pt1, Point2l& pt2) {
    int c1, c2;
    int64_t right = img_size.width-1, bottom = img_size.height-1;
    if (img_size.width <= 0 || img_size.height <= 0) return false;

    int64_t &x1 = pt1.x, &y1 = pt1.y, &x2 = pt2.x, &y2 = pt2.y;
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;

    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
        int64_t a;
        if (c1 & 12) {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int64_t)((double)(a - y1) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if (c2 & 12) {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int64_t)((double)(a - y2) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if ((c1 & c2) == 0 && (c1 | c2) != 0) {
            if (c1) {
                a = c1 == 1 ? 0 : right;
                y1 += (int64_t)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if (c2) {
                a = c2 == 1 ? 0 : right;
                y2 += (int64_t)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }
        MNN_ASSERT((c1 & c2) != 0 || (x1 | y1 | x2 | y2) >= 0);
    }
    return (c1 | c2) == 0;
}
bool clipLine(Size img_size, Point2i& pt1, Point2i& pt2) {
    Point2l p1(pt1.x, pt1.y);
    Point2l p2(pt2.x, pt2.y);
    bool inside = clipLine(Size2l(img_size.width, img_size.height), p1, p2);
    pt1.x = (int)p1.x;
    pt1.y = (int)p1.y;
    pt2.x = (int)p2.x;
    pt2.y = (int)p2.y;
    return inside;
}

enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1<<12) - 256 };
static void Line(std::vector<Region>& regions, Size size, Point2i pt1_, Point2i pt2_, int connectivity = 8) {
    if (connectivity == 0) {
        connectivity = 8;
    } else if (connectivity == 1) {
        connectivity = 4;
    }
    int count = -1, err, minusDelta, plusDelta, minusStep, plusStep, minusShift, plusShift;
    Point2i p = Point2i(0, 0);
    Rect2i rect(0, 0, size.width, size.height);
    Point2i pt1 = pt1_ - rect.tl();
    Point2i pt2 = pt2_ - rect.tl();

    if ((unsigned)pt1.x >= (unsigned)(rect.width) || (unsigned)pt2.x >= (unsigned)(rect.width) ||
        (unsigned)pt1.y >= (unsigned)(rect.height) || (unsigned)pt2.y >= (unsigned)(rect.height)) {
        if (!clipLine(Size(rect.width, rect.height), pt1, pt2)) {
            err = plusDelta = minusDelta = plusStep = minusStep = plusShift = minusShift = count = 0;
        }
    }

    pt1 += rect.tl();
    pt2 += rect.tl();

    int delta_x = 1, delta_y = 1;
    int dx = pt2.x - pt1.x;
    int dy = pt2.y - pt1.y;

    if (dx < 0) {
        dx = -dx;
        dy = -dy;
        pt1 = pt2;
    }

    if (dy < 0) {
        dy = -dy;
        delta_y = -1;
    }

    bool vert = dy > dx;
    if (vert) {
        std::swap(dx, dy);
        std::swap(delta_x, delta_y);
    }

    MNN_ASSERT(dx >= 0 && dy >= 0);

    if (connectivity == 8) {
        err = dx - (dy + dy);
        plusDelta = dx + dx;
        minusDelta = -(dy + dy);
        minusShift = delta_x;
        plusShift = 0;
        minusStep = 0;
        plusStep = delta_y;
        count = dx + 1;
    } else /* connectivity == 4 */ {
        err = 0;
        plusDelta = (dx + dx) + (dy + dy);
        minusDelta = -(dy + dy);
        minusShift = delta_x;
        plusShift = -delta_x;
        minusStep = 0;
        plusStep = delta_y;
        count = dx + dy + 1;
    }

    if (vert) {
        std::swap(plusStep, plusShift);
        std::swap(minusStep, minusShift);
    }
    p = pt1;
    regions.emplace_back(Region{p.y, p.x});
    for(int i = 1; i < count; i++) {
        int mask = err < 0 ? -1 : 0;
        err += minusDelta + (plusDelta & mask);
        p.y += minusStep + (plusStep & mask);
        p.x += minusShift + (plusShift & mask);
        regions.emplace_back(Region{p.y, p.x});
    }
}

static void Line2(std::vector<Region>& regions, Size size, Point2l pt1, Point2l pt2) {
    int64_t dx, dy;
    int ecount;
    int64_t ax, ay;
    int64_t i, j;
    int x, y;
    int64_t x_step, y_step;
    Size2l sizeScaled(((int64_t)size.width) << XY_SHIFT, ((int64_t)size.height) << XY_SHIFT);
    if(!clipLine(sizeScaled, pt1, pt2)) {
        return;
    }
    dx = pt2.x - pt1.x;
    dy = pt2.y - pt1.y;
    j = dx < 0 ? -1 : 0;
    ax = (dx ^ j) - j;
    i = dy < 0 ? -1 : 0;
    ay = (dy ^ i) - i;

    if (ax > ay) {
        dy = (dy ^ j) - j;
        pt1.x ^= pt2.x & j;
        pt2.x ^= pt1.x & j;
        pt1.x ^= pt2.x & j;
        pt1.y ^= pt2.y & j;
        pt2.y ^= pt1.y & j;
        pt1.y ^= pt2.y & j;

        x_step = XY_ONE;
        y_step = (dy << XY_SHIFT) / (ax | 1);
        ecount = (int)((pt2.x - pt1.x) >> XY_SHIFT);
    } else {
        dx = (dx ^ i) - i;
        pt1.x ^= pt2.x & i;
        pt2.x ^= pt1.x & i;
        pt1.x ^= pt2.x & i;
        pt1.y ^= pt2.y & i;
        pt2.y ^= pt1.y & i;
        pt1.y ^= pt2.y & i;

        x_step = (dx << XY_SHIFT) / (ay | 1);
        y_step = XY_ONE;
        ecount = (int)((pt2.y - pt1.y) >> XY_SHIFT);
    }
    pt1.x += (XY_ONE >> 1);
    pt1.y += (XY_ONE >> 1);
    regions.emplace_back(Region{(int)((pt2.y + (XY_ONE >> 1)) >> XY_SHIFT), (int)((pt2.x + (XY_ONE >> 1)) >> XY_SHIFT)});
    if (ax > ay) {
        pt1.x >>= XY_SHIFT;
        while(ecount >= 0) {
            regions.emplace_back(Region{(int)(pt1.y >> XY_SHIFT), (int)(pt1.x)});
            pt1.x++;
            pt1.y += y_step;
            ecount--;
        }
    } else {
        pt1.y >>= XY_SHIFT;
        while(ecount >= 0) {
            regions.emplace_back(Region{(int)(pt1.y), (int)(pt1.x >> XY_SHIFT)});
            pt1.x += x_step;
            pt1.y++;
            ecount--;
        }
    }
}

static void FillConvexPoly(std::vector<Region>& regions, Size size, const Point2l* v, int npts, int line_type, int shift) {
    struct {
        int idx, di;
        int64_t x, dx;
        int ye;
    } edge[2];

    int delta = 1 << shift >> 1;
    int i, y, imin = 0;
    int edges = npts;
    int64_t xmin, xmax, ymin, ymax;
    Point2l p0;
    int delta1, delta2;

    delta1 = delta2 = XY_ONE >> 1;

    p0 = v[npts - 1];
    p0.x <<= XY_SHIFT - shift;
    p0.y <<= XY_SHIFT - shift;

    MNN_ASSERT(0 <= shift && shift <= XY_SHIFT);
    xmin = xmax = v[0].x;
    ymin = ymax = v[0].y;

    for (i = 0; i < npts; i++) {
        Point2l p = v[i];
        if (p.y < ymin) {
            ymin = p.y;
            imin = i;
        }

        ymax = std::max(ymax, p.y);
        xmax = std::max(xmax, p.x);
        xmin = MIN(xmin, p.x);

        p.x <<= XY_SHIFT - shift;
        p.y <<= XY_SHIFT - shift;

        if(!shift) {
            Point2i pt0, pt1;
            pt0.x = (int)(p0.x >> XY_SHIFT);
            pt0.y = (int)(p0.y >> XY_SHIFT);
            pt1.x = (int)(p.x >> XY_SHIFT);
            pt1.y = (int)(p.y >> XY_SHIFT);
            Line(regions, size, pt0, pt1, line_type);
        } else {
            Line2(regions, size, p0, p);
        }
        p0 = p;
    }

    xmin = (xmin + delta) >> shift;
    xmax = (xmax + delta) >> shift;
    ymin = (ymin + delta) >> shift;
    ymax = (ymax + delta) >> shift;

    if(npts < 3 || (int)xmax < 0 || (int)ymax < 0 || (int)xmin >= size.width || (int)ymin >= size.height) {
        return;
    }
    ymax = MIN(ymax, size.height - 1);
    edge[0].idx = edge[1].idx = imin;
    edge[0].ye = edge[1].ye = y = (int)ymin;
    edge[0].di = 1;
    edge[1].di = npts - 1;
    edge[0].x = edge[1].x = -XY_ONE;
    edge[0].dx = edge[1].dx = 0;
    int region_y = y;
    do {
        if (y < (int)ymax || y == (int)ymin) {
            for (i = 0; i < 2; i++) {
                if (y >= edge[i].ye) {
                    int idx0 = edge[i].idx, di = edge[i].di;
                    int idx = idx0 + di;
                    if (idx >= npts) idx -= npts;
                    int ty = 0;

                    for (; edges-- > 0; ) {
                        ty = (int)((v[idx].y + delta) >> shift);
                        if (ty > y) {
                            int64_t xs = v[idx0].x;
                            int64_t xe = v[idx].x;
                            if (shift != XY_SHIFT)
                            {
                                xs <<= XY_SHIFT - shift;
                                xe <<= XY_SHIFT - shift;
                            }

                            edge[i].ye = ty;
                            edge[i].dx = ((xe - xs)*2 + (ty - y)) / (2 * (ty - y));
                            edge[i].x = xs;
                            edge[i].idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx += di;
                        if (idx >= npts) idx -= npts;
                    }
                }
            }
        }

        if (edges < 0)
            break;

        if (y >= 0) {
            int left = 0, right = 1;
            if (edge[0].x > edge[1].x)
            {
                left = 1, right = 0;
            }

            int xx1 = (int)((edge[left].x + delta1) >> XY_SHIFT);
            int xx2 = (int)((edge[right].x + delta2) >> XY_SHIFT);

            if(xx2 >= 0 && xx1 < size.width)
            {
                if(xx1 < 0) {
                    xx1 = 0;
                }
                if(xx2 >= size.width) {
                    xx2 = size.width - 1;
                }
                if (xx2 - xx1 > 0) regions.emplace_back(Region{region_y, xx1, xx2});
            }
        }

        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        region_y++;
    } while(++y <= (int)ymax);
}

static void sincos(int angle, float& cosval, float& sinval) {
    angle += (angle < 0 ? 360 : 0);
    float radian = angle * MNN_PI / 180;
    sinval = sin(radian);
    cosval = cos(radian);
}

void ellipse2Poly(Point2d center, Size2d axes, int angle, int arc_start, int arc_end, int delta, std::vector<Point2d>& pts) {
    MNN_ASSERT(0 < delta && delta <= 180);

    float alpha, beta;
    int i;

    while(angle < 0) angle += 360;
    while(angle > 360) angle -= 360;

    if (arc_start > arc_end) {
        i = arc_start;
        arc_start = arc_end;
        arc_end = i;
    }
    while (arc_start < 0) {
        arc_start += 360;
        arc_end += 360;
    }
    while (arc_end > 360) {
        arc_end -= 360;
        arc_start -= 360;
    }
    if (arc_end - arc_start > 360) {
        arc_start = 0;
        arc_end = 360;
    }
    sincos(angle, alpha, beta);
    pts.resize(0);

    for (i = arc_start; i < arc_end + delta; i += delta) {
        double x, y;
        angle = i;
        if (angle > arc_end) angle = arc_end;
        float sinv, cosv;
        sincos(angle, sinv, cosv);
        x = axes.width * cosv;
        y = axes.height * sinv;
        Point2d pt;
        pt.x = center.x + x * alpha - y * beta;
        pt.y = center.y + x * beta + y * alpha;
        pts.push_back(pt);
    }

    // If there are no points, it's a zero-size polygon
    if( pts.size() == 1) {
        pts.assign(2,center);
    }
}
static void ThickLine(std::vector<Region>& regions, Size size, Point2l p0, Point2l p1, int thickness, int line_type, int flags, int shift);
static void PolyLine(std::vector<Region>& regions, Size size, const Point2l* v, int count, bool is_closed, int thickness, int line_type, int shift) {
    if (!v || count <= 0) {
        return;
    }

    int i = is_closed ? count - 1 : 0;
    int flags = 2 + !is_closed;
    Point2l p0;
    MNN_ASSERT(0 <= shift && shift <= XY_SHIFT && thickness >= 0);

    p0 = v[i];
    for (i = !is_closed; i < count; i++) {
        Point2l p = v[i];
        ThickLine(regions, size, p0, p, thickness, line_type, flags, shift );
        p0 = p;
        flags = 2;
    }
}

struct PolyEdge {
    PolyEdge() : y0(0), y1(0), x(0), dx(0), next(0) {}

    int y0, y1;
    int64_t x, dx;
    PolyEdge *next;
};

static void CollectPolyEdges(std::vector<Region>& regions, Size size, const Point2l* v, int count, std::vector<PolyEdge>& edges, int line_type, int shift, Point2i offset = Point2i()) {
    int delta = offset.y + ((1 << shift) >> 1);
    Point2l pt0 = v[count-1], pt1;
    pt0.x = (pt0.x + offset.x) << (XY_SHIFT - shift);
    pt0.y = (pt0.y + delta) >> shift;

    edges.reserve(edges.size() + count);

    for (int i = 0; i < count; i++, pt0 = pt1) {
        Point2l t0, t1;
        PolyEdge edge;

        pt1 = v[i];
        pt1.x = (pt1.x + offset.x) << (XY_SHIFT - shift);
        pt1.y = (pt1.y + delta) >> shift;

        t0.y = pt0.y; t1.y = pt1.y;
        t0.x = (pt0.x + (XY_ONE >> 1)) >> XY_SHIFT;
        t1.x = (pt1.x + (XY_ONE >> 1)) >> XY_SHIFT;
        Line(regions, size, t0, t1, line_type);

        if (pt0.y == pt1.y) continue;

        if (pt0.y < pt1.y) {
            edge.y0 = (int)(pt0.y);
            edge.y1 = (int)(pt1.y);
            edge.x = pt0.x;
        } else {
            edge.y0 = (int)(pt1.y);
            edge.y1 = (int)(pt0.y);
            edge.x = pt1.x;
        }
        edge.dx = (pt1.x - pt0.x) / (pt1.y - pt0.y);
        edges.push_back(edge);
    }
}

static void FillEdgeCollection(std::vector<Region>& regions, Size size, std::vector<PolyEdge>& edges) {
    PolyEdge tmp;
    int i, y, total = (int)edges.size();
    PolyEdge* e;
    int y_max = std::numeric_limits<int>::min(), y_min = std::numeric_limits<int>::max();
    int64_t x_max = 0xFFFFFFFFFFFFFFFF, x_min = 0x7FFFFFFFFFFFFFFF;

    if (total < 2) return;

    for (i = 0; i < total; i++) {
        PolyEdge& e1 = edges[i];
        MNN_ASSERT(e1.y0 < e1.y1);
        // Determine x-coordinate of the end of the edge.
        // (This is not necessary x-coordinate of any vertex in the array.)
        int64_t x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
        y_min = std::min( y_min, e1.y0 );
        y_max = std::max( y_max, e1.y1 );
        x_min = std::min( x_min, e1.x );
        x_max = std::max( x_max, e1.x );
        x_min = std::min( x_min, x1 );
        x_max = std::max( x_max, x1 );
    }

    if (y_max < 0 || y_min >= size.height || x_max < 0 || x_min >= ((int64_t)size.width<<XY_SHIFT)) return;

    std::sort( edges.begin(), edges.end(), [](const PolyEdge& e1, const PolyEdge& e2) {
        return e1.y0 - e2.y0 ? e1.y0 < e2.y0 : e1.x - e2.x ? e1.x < e2.x : e1.dx < e2.dx;
    });

    // start drawing
    tmp.y0 = std::numeric_limits<int>::max();
    edges.push_back(tmp); // after this point we do not add
                          // any elements to edges, thus we can use pointers
    i = 0;
    tmp.next = 0;
    e = &edges[i];
    y_max = MIN(y_max, size.height);

    for (y = e->y0; y < y_max; y++) {
        PolyEdge *last, *prelast, *keep_prelast;
        int sort_flag = 0;
        int draw = 0;
        int clipline = y < 0;

        prelast = &tmp;
        last = tmp.next;
        while (last || e->y0 == y) {
            if (last && last->y1 == y) {
                // exclude edge if y reaches its lower point
                prelast->next = last->next;
                last = last->next;
                continue;
            }
            keep_prelast = prelast;
            if (last && (e->y0 > y || last->x < e->x)) {
                // go to the next edge in active list
                prelast = last;
                last = last->next;
            } else if(i < total) {
                // insert new edge into active list if y reaches its upper point
                prelast->next = e;
                e->next = last;
                prelast = e;
                e = &edges[++i];
            } else {
                break;
            }

            if (draw) {
                if(!clipline) {
                    // convert x's from fixed-point to image coordinates
                    // uchar *timg = const_cast<uchar*>(img->readMap<uchar>()) + (y * pix_size * w);
                    int x1, x2;

                    if (keep_prelast->x > prelast->x) {
                        x1 = (int)((prelast->x + XY_ONE - 1) >> XY_SHIFT);
                        x2 = (int)(keep_prelast->x >> XY_SHIFT);
                    } else {
                        x1 = (int)((keep_prelast->x + XY_ONE - 1) >> XY_SHIFT);
                        x2 = (int)(prelast->x >> XY_SHIFT);
                    }

                    // clip and draw the line
                    if( x1 < size.width && x2 >= 0 )
                    {
                        if (x1 < 0) x1 = 0;
                        if (x2 >= size.width) x2 = size.width - 1;
                        regions.emplace_back(Region{y, x1, x2});
                    }
                }
                keep_prelast->x += keep_prelast->dx;
                prelast->x += prelast->dx;
            }
            draw ^= 1;
        }

        // sort edges (using bubble sort)
        keep_prelast = 0;
        do {
            prelast = &tmp;
            last = tmp.next;

            while (last != keep_prelast && last->next != 0) {
                PolyEdge *te = last->next;
                // swap edges
                if (last->x > te->x) {
                    prelast->next = te;
                    last->next = te->next;
                    te->next = last;
                    prelast = te;
                    sort_flag = 1;
                } else {
                    prelast = last;
                    last = te;
                }
            }
            keep_prelast = prelast;
        } while(sort_flag && keep_prelast != tmp.next && keep_prelast != &tmp);
    }
}

static void EllipseEx(std::vector<Region>& regions, Size size, Point2l center, Size2l axes, int angle, int arc_start, int arc_end, int thickness, int line_type) {
    axes.width = std::abs(axes.width), axes.height = std::abs(axes.height);
    int delta = (int)((std::max(axes.width,axes.height)+(XY_ONE>>1))>>XY_SHIFT);
    delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;

    std::vector<Point2d> _v;
    ellipse2Poly(Point2d((double)center.x, (double)center.y), Size2d((double)axes.width, (double)axes.height), angle, arc_start, arc_end, delta, _v);

    std::vector<Point2l> v;
    Point2l prevPt(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    v.resize(0);
    for (unsigned int i = 0; i < _v.size(); ++i)
    {
        Point2l pt;
        pt.x = (int64_t)std::round(_v[i].x / XY_ONE) << XY_SHIFT;
        pt.y = (int64_t)std::round(_v[i].y / XY_ONE) << XY_SHIFT;
        pt.x += std::round(_v[i].x - pt.x);
        pt.y += std::round(_v[i].y - pt.y);
        if (pt != prevPt) {
            v.push_back(pt);
            prevPt = pt;
        }
    }

    // If there are no points, it's a zero-size polygon
    if (v.size() == 1) {
        v.assign(2, center);
    }

    if (thickness >= 0) {
        PolyLine(regions, size, &v[0], (int)v.size(), false, thickness, line_type, XY_SHIFT);
    } else if( arc_end - arc_start >= 360 ) {
        FillConvexPoly(regions, size, &v[0], (int)v.size(), line_type, XY_SHIFT);
    } else {
        v.push_back(center);
        std::vector<PolyEdge> edges;
        CollectPolyEdges(regions, size, &v[0], (int)v.size(), edges, line_type, XY_SHIFT);
        FillEdgeCollection(regions, size, edges);
    }
}

static void Circle(std::vector<Region>& regions, Size size, Point2i center, int radius, int fill) {
    int err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1;
    int inside = center.x >= radius && center.x < size.width - radius &&
                 center.y >= radius && center.y < size.height - radius;

    while (dx >= dy) {
        int mask;
        int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
        int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

        if (inside) {
            if(!fill) {
                regions.emplace_back(Region{y11, x11});
                regions.emplace_back(Region{y12, x11});
                regions.emplace_back(Region{y11, x12});
                regions.emplace_back(Region{y12, x12});
                regions.emplace_back(Region{y21, x21});
                regions.emplace_back(Region{y22, x21});
                regions.emplace_back(Region{y21, x22});
                regions.emplace_back(Region{y22, x22});
            } else {
                regions.emplace_back(Region{y11, x11, x12});
                regions.emplace_back(Region{y12, x11, x12});
                regions.emplace_back(Region{y21, x21, x22});
                regions.emplace_back(Region{y22, x21, x22});
            }
        } else if (x11 < size.width && x12 >= 0 && y21 < size.height && y22 >= 0) {
            if (fill) {
                x11 = std::max(x11, 0);
                x12 = MIN(x12, size.width - 1);
            }
            if ((unsigned)y11 < (unsigned)size.height) {
                if (!fill) {
                    if(x11 >= 0) regions.emplace_back(Region{y11, x11});
                    if(x12 < size.width) regions.emplace_back(Region{y11, x12});
                } else {
                    regions.emplace_back(Region{y11, x11, x12});
                }
            }
            if ((unsigned)y12 < (unsigned)size.height) {
                if(!fill) {
                    if(x11 >= 0) regions.emplace_back(Region{y12, x11});
                    if(x12 < size.width) regions.emplace_back(Region{y12, x12});
                } else {
                   regions.emplace_back(Region{y12, x11, x12});
                }
            }

            if (x21 < size.width && x22 >= 0) {
                if (fill) {
                    x21 = std::max(x21, 0);
                    x22 = MIN(x22, size.width - 1);
                }
                if ((unsigned)y21 < (unsigned)size.height) {
                    if(!fill) {
                        if(x21 >= 0) regions.emplace_back(Region{y21, x21});
                        if(x22 < size.width) regions.emplace_back(Region{y21, x22});
                    } else {
                       regions.emplace_back(Region{y21, x21, x22});
                    }
                }
                if ((unsigned)y22 < (unsigned)size.height) {
                    if(!fill) {
                        if(x21 >= 0) regions.emplace_back(Region{y22, x21});
                        if(x22 < size.width) regions.emplace_back(Region{y22, x22});
                    } else {
                       regions.emplace_back(Region{y22, x21, x22});
                    }
                }
            }
        }
        dy++;
        err += plus;
        plus += 2;
        mask = (err <= 0) - 1;
        err -= minus & mask;
        dx += mask;
        minus -= mask & 2;
    }
}

static void ThickLine(std::vector<Region>& regions, Size size, Point2l p0, Point2l p1, int thickness, int line_type, int flags, int shift) {
    constexpr double INV_XY_ONE = 1./XY_ONE;
    p0.x <<= XY_SHIFT - shift;
    p0.y <<= XY_SHIFT - shift;
    p1.x <<= XY_SHIFT - shift;
    p1.y <<= XY_SHIFT - shift;

    if(thickness <= 1) {
        if (line_type == 1 || line_type == 4 || shift == 0) {
            p0.x = (p0.x + (XY_ONE>>1)) >> XY_SHIFT;
            p0.y = (p0.y + (XY_ONE>>1)) >> XY_SHIFT;
            p1.x = (p1.x + (XY_ONE>>1)) >> XY_SHIFT;
            p1.y = (p1.y + (XY_ONE>>1)) >> XY_SHIFT;
            Line(regions, size, p0, p1, line_type);
        } else {
            Line2(regions, size, p0, p1);
        }
    } else {
        Point2l pt[4], dp = Point2i(0,0);
        double dx = (p0.x - p1.x)*INV_XY_ONE, dy = (p1.y - p0.y)*INV_XY_ONE;
        double r = dx * dx + dy * dy;
        int i, oddThickness = thickness & 1;
        thickness <<= XY_SHIFT - 1;

        if( fabs(r) > 2.2e-16 ) {
            r = (thickness + oddThickness * XY_ONE * 0.5) / std::sqrt(r);
            dp.x = std::round( dy * r );
            dp.y = std::round( dx * r );

            pt[0].x = p0.x + dp.x;
            pt[0].y = p0.y + dp.y;
            pt[1].x = p0.x - dp.x;
            pt[1].y = p0.y - dp.y;
            pt[2].x = p1.x - dp.x;
            pt[2].y = p1.y - dp.y;
            pt[3].x = p1.x + dp.x;
            pt[3].y = p1.y + dp.y;
            FillConvexPoly(regions, size, pt, 4, line_type, XY_SHIFT);
        }

        for(i = 0; i < 2; i++) {
            if(flags & (i+1)) {
                Point2i center;
                center.x = (int)((p0.x + (XY_ONE>>1)) >> XY_SHIFT);
                center.y = (int)((p0.y + (XY_ONE>>1)) >> XY_SHIFT);
                Circle(regions, size, center, (thickness + (XY_ONE>>1)) >> XY_SHIFT, 1);
            }
            p0 = p1;
        }
    }
}

template <typename T> static inline
void scalarToRawData_(const Scalar& s, T * const buf, const int cn) {
    for(int i = 0; i < cn; i++) {
        buf[i] = static_cast<T>(s.val[i]);
    }
}

void scalarToRawData(const Scalar& s, void* buf, VARP img) {
    auto type = img->getInfo()->type;
    int cn = getVARPChannel(img);
    if (type == halide_type_of<uint8_t>()) {
        scalarToRawData_<uchar>(s, (uchar*)buf, cn);
    } else if (type == halide_type_of<float>()) {
        scalarToRawData_<float>(s, (float*)buf, cn);
    } else if (type == halide_type_of<double>()) {
        scalarToRawData_<double>(s, (double*)buf, cn);
    } else if (type == halide_type_of<int>()) {
        scalarToRawData_<int>(s, (int*)buf, cn);
    }
}

std::vector<Region> mergeRegions(std::vector<Region> regions) {
    std::vector<Region> res;
    // 1. get line's region
    std::map<int, std::vector<std::pair<int, int>>> lines;
    for (auto region : regions) {
        if (lines.find(region.y) != lines.end()) {
            lines[region.y].push_back({region.xl, region.xr});
        } else {
            lines[region.y] = std::vector<std::pair<int, int>>();
            lines[region.y].push_back({region.xl, region.xr});
        }
    }
    // 2. merge line's region
    for (auto line : lines) {
        auto liner = line.second;
        // sort line regions
        std::sort(liner.begin(), liner.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b){return a.first < b.first;});
        // merge
        res.emplace_back(Region{line.first, liner[0].first, liner[0].second});
        for (int i = 1; i < liner.size(); i++) {
            if (res.back().xr >= liner[i].second) {
                res.back().xr = MAX(res.back().xr, liner[i].second);
            } else {
                res.emplace_back(Region{line.first, liner[i].first, liner[i].second});
            }
        }
    }
    return res;
}

void doDraw(VARP& img, const std::vector<Region>& regions, const Scalar& color) {
    double buf[4];
    scalarToRawData(color, buf, img);
    auto mergeRegs = mergeRegions(regions);
    ImageProcess::Config config;
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    process->setDraw();
    int h, w, c; getVARPSize(img, &h, &w, &c);
    auto ptr = const_cast<uint8_t*>(img->readMap<uint8_t>());
    int num = (int)mergeRegs.size();
    process->draw(ptr, w, h, c, reinterpret_cast<const int*>(mergeRegs.data()), num, (uint8_t*)buf);
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

void circle(VARP& img, Point center, int radius, const Scalar& color, int thickness, int line_type, int shift) {
    Point2i center_(static_cast<int>(center.fX), static_cast<int>(center.fY));
    int h, w, c; getVARPSize(img, &h, &w, &c);
    Size size(w, h);
    std::vector<Region> regions;
    if( thickness > 1 || line_type != LINE_8 || shift > 0 ) {
        Point2l _center(center_);
        int64_t _radius(radius);
        _center.x <<= XY_SHIFT - shift;
        _center.y <<= XY_SHIFT - shift;
        _radius <<= XY_SHIFT - shift;
        EllipseEx(regions, size, _center, Size2l(_radius, _radius), 0, 0, 360, thickness, line_type);
    } else {
        Circle(regions, size, center_, radius, thickness < 0);
    }
    doDraw(img, regions, color);
}

void line(VARP& img, Point pt1, Point pt2, const Scalar& color,
          int thickness, int lineType, int shift) {
    int h, w, c; getVARPSize(img, &h, &w, &c);
    Point2i p1(static_cast<int>(pt1.fX), static_cast<int>(pt1.fY));
    Point2i p2(static_cast<int>(pt2.fX), static_cast<int>(pt2.fY));
    std::vector<Region> regions;
    ThickLine(regions, Size{w, h}, p1, p2, thickness, lineType, 3, shift);
    doDraw(img, regions, color);
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

void drawContours(VARP& img, std::vector<std::vector<Point>> _contours, int contourIdx, const Scalar& color, int thickness, int lineType) {
    size_t ncontours = _contours.size();
    if (!ncontours) return;
    int h, w, c; getVARPSize(img, &h, &w, &c);
    Size size(w, h);
    std::vector<Region> regions;
    size_t i = 0, first = 0, last = ncontours;
    if (contourIdx >= 0) {
        first = contourIdx;
        last = first + 1;
    }
    std::vector<PolyEdge> edges;
    for (i = first; i < last; i++) {
        const auto& contour = _contours[i];
        if (contour.empty()) continue;
        std::vector<Point2l> pts;
        for (int j = 0; j < contour.size(); j++) {
            int nextj = j + 1 == contour.size() ? 0 : j + 1;
            Point2l pt1(contour[j].fX, contour[j].fY), pt2(contour[nextj].fX, contour[nextj].fY);
            if(thickness >= 0) {
                ThickLine(regions, size, pt1, pt2, thickness, lineType, 2, 0);
            } else {
                if (!j) pts.push_back(pt1);
                pts.push_back(pt2);
            }
        }
        if (thickness < 0) {
            CollectPolyEdges(regions, size, &pts[0], (int)pts.size(), edges, lineType, 0);
        }
    }
    if (thickness < 0) {
        FillEdgeCollection(regions, size, edges);
    }
    doDraw(img, regions, color);
}

void fillPoly(VARP& img,  std::vector<std::vector<Point>> _pts, const Scalar& color, int line_type, int shift, Point _offset) {
    int ncontours = _pts.size();
    if (!ncontours) return;
    int h, w, c;
    getVARPSize(img, &h, &w, &c);
    Size size(w, h);
    std::vector<Region> regions;
    std::vector<std::vector<Point2i>> pts(ncontours);
    std::vector<Point2i*> _ptsptr(ncontours);
    std::vector<int> _npts(ncontours);
    Point2i** ptsptr = _ptsptr.data();
    int *npts = _npts.data(), total = 0;
    for(int i = 0; i < ncontours; i++ ) {
        int num = _pts[i].size();
        pts[i].resize(num);
        for (int j = 0; j < num; j++) {
            pts[i][j].x = _pts[i][j].fX;
            pts[i][j].y = _pts[i][j].fY;
        }
        ptsptr[i] = pts[i].data();
        npts[i] = num;
        total += num;
    }
    if(line_type == LINE_AA && img->getInfo()->type == halide_type_of<uint8_t>()) line_type = 8;
    MNN_ASSERT(ptsptr && npts && ncontours >= 0 && 0 <= shift && shift <= XY_SHIFT);
    std::vector<PolyEdge> edges;
    Point2i offset(_offset.fX, _offset.fY);
    edges.reserve( total + 1 );
    for (int i = 0; i < ncontours; i++) {
        std::vector<Point2l> _pts(ptsptr[i], ptsptr[i] + npts[i]);
        CollectPolyEdges(regions, size, _pts.data(), npts[i], edges, line_type, shift, offset);
    }
    FillEdgeCollection(regions, size, edges);
    doDraw(img, regions, color);
}

} // CV
} // MNN
