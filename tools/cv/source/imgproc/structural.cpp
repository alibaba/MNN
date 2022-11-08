//
//  structural.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ImageProcess.hpp>
#include "cv/imgproc/structural.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <cmath>
#include <set>
#include <limits>

namespace MNN {
namespace CV {

/*Copy From OpenCV*/
#define  CV_CMP(a,b)    (((a) > (b)) - ((a) < (b)))
#define  CV_SIGN(a)     CV_CMP((a),0)
struct CvPoint {
    int x;
    int y;

    template<typename _Tp> CvPoint(const std::initializer_list<_Tp> list)
    {
        MNN_ASSERT(list.size() == 0 || list.size() == 2);
        x = y = 0;
        if (list.size() == 2)
        {
            x = list.begin()[0]; y = list.begin()[1];
        }
    };
    CvPoint(int _x = 0, int _y = 0): x(_x), y(_y) {}
};

static const CvPoint icvCodeDeltas[8] =
{ CvPoint(1, 0), CvPoint(1, -1), CvPoint(0, -1), CvPoint(-1, -1), CvPoint(-1, 0), CvPoint(-1, 1), CvPoint(0, 1), CvPoint(1, 1) };

/* initializes 8-element array for fast access to 3x3 neighborhood of a pixel */
#define  CV_INIT_3X3_DELTAS( deltas, step, nch )            \
    ((deltas)[0] =  (nch),  (deltas)[1] = -(step) + (nch),  \
     (deltas)[2] = -(step), (deltas)[3] = -(step) - (nch),  \
     (deltas)[4] = -(nch),  (deltas)[5] =  (step) - (nch),  \
     (deltas)[6] =  (step), (deltas)[7] =  (step) + (nch))

struct _CvContourScanner {
public:
    _CvContourScanner() {}
    schar *img0;                /* image origin */
    schar *img;                 /* current image row */
    int img_step;               /* image step */
    int img_w;
    int img_h;
    CvPoint offset;             /* ROI offset: coordinates, added to each contour point */
    CvPoint pt;                 /* current scanner position */
    CvPoint lnbd;               /* position of the last met contour */
    int nbd;                    /* current mark val */
    int mode;                   /* contour scanning mode:
                                  0 - external only
                                  1 - all the contours w/o any hierarchy
                                  2 - connected components (i.e. two-level structure -
                                  external contours and holes),
                                  3 - full hierarchy;
                                  4 - connected components of a multi-level image
                                 */
    int method;
};

typedef _CvContourScanner* CvContourScanner;

static CvContourScanner cvStartFindContours( VARP _img, CvPoint offset, int mode, int method )
{
    int img_w, img_h, img_c;
    getVARPSize(_img, &img_h, &img_w, &img_c);
    int step = img_w;
    MNN_ASSERT(img_c == 1);
    uchar* img = (uchar*)(_img->readMap<uchar>());
    for (int i=0; i<img_h*img_w; ++i)
    {
        if (img[i] > 0)
        {
            img[i] = 1;
        }
    }

    CvContourScanner scanner = new _CvContourScanner;
    memset( scanner, 0, sizeof(*scanner) );

    scanner->img0 = (schar *) img;
    scanner->img = (schar *) (img + step);
    scanner->img_step = step;
    scanner->img_h = img_h - 1;
    scanner->img_w = img_w -1;
    scanner->mode = mode;
    scanner->method = method;
    scanner->offset = offset;
    scanner->pt.x = scanner->pt.y = 1;
    scanner->lnbd.x = 0;
    scanner->lnbd.y = 1;
    scanner->nbd = 2;

    memset( img, 0, img_w*sizeof(uchar) );
    memset( img + step * (img_h - 1), 0, img_w*sizeof(uchar) );

    img += step;
    for( int y = 1; y < img_h - 1; y++, img += step )
    {
        img[0] = img[img_w-1] = (schar)0;
    }
    return scanner;
}

static void icvFetchContour(schar* ptr, int step, CvPoint pt, bool is_hole, int method, std::vector<CvPoint>& points)
{
    const char     nbd = 2;
    int             deltas[16];
    int             prev_s = -1, s, s_end;
    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1);
    ::memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));
    schar  *i0 = (ptr), *i1, *i3, *i4 = 0;
    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
        if( *i1 != 0 )
            break;
    }
    while( s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | -128);
        if( method >= 0 )
        {
            points.push_back(pt);
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | -128);
            }
            else if( *i3 == 1 )
            {
                *i3 = nbd;
            }

            {
                if( s != prev_s || method == 1 )
                {
                    points.push_back(pt);
                    prev_s = s;
                }

                pt.x += icvCodeDeltas[s].x;
                pt.y += icvCodeDeltas[s].y;

            }

            if( i4 == i0 && i3 == i1 )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
}

static bool cvFindNextContour(CvContourScanner scanner, std::vector<CvPoint>& points)
{
    /* initialize local state */
    schar* img0 = scanner->img0;
    schar* img = scanner->img;
    int step = scanner->img_step;
    int x = scanner->pt.x;
    int y = scanner->pt.y;
    int width = scanner->img_w;
    int height = scanner->img_h;
    int mode = scanner->mode;
    int method = scanner->method;
    CvPoint lnbd = scanner->lnbd;
    int nbd = scanner->nbd;
    int prev = img[x - 1];
    int new_mask = -2;

    for( ; y < height; y++, img += step )
    {
        int* img0_i = 0;
        int* img_i = 0;
        int p = 0;

        for( ; x < width; x++ )
        {
            if( img_i )
            {
                for( ; x < width && ((p = img_i[x]) == prev || (p & ~new_mask) == (prev & ~new_mask)); x++ )
                    prev = p;
            }
            else
            {
                for( ; x < width && (p = img[x]) == prev; x++ )
                    ;
            }

            if( x >= width )
                break;

            {
                int is_hole = 0;
                CvPoint origin;

                /* if not external contour */
                if( (!img_i && !(prev == 0 && p == 1)) ||
                    (img_i && !(((prev & new_mask) != 0 || prev == 0) && (p & new_mask) == 0)) )
                {
                    /* check hole */
                    if( (!img_i && (p != 0 || prev < 1)) ||
                        (img_i && ((prev & new_mask) != 0 || (p & new_mask) != 0)))
                        goto resume_scan;

                    if( prev & new_mask )
                    {
                        lnbd.x = x - 1;
                    }
                    is_hole = 1;
                }
                
                if( mode == 0 && (is_hole || img0[lnbd.y * static_cast<size_t>(step) + lnbd.x] > 0) )
                    goto resume_scan;


                origin.y = y;
                origin.x = x - is_hole;

                lnbd.x = x - is_hole;
                
                /* initialize header */
                CvPoint searchStart(origin.x + scanner->offset.x, origin.y + scanner->offset.y);
                icvFetchContour( img + x - is_hole, step, searchStart, is_hole, method, points);
                
                p = img[x];
                prev = p;
                /* update lnbd */
                if( prev & -2 )
                {
                    lnbd.x = x;
                }
                if (!points.empty()) {
                    return true;
                }
            }                   /* end of prev != p */
        resume_scan:
            {
                prev = p;
                /* update lnbd */
                if( prev & -2 )
                {
                    lnbd.x = x;
                }
            }
        }                       /* end of loop on x */

        lnbd.x = 0;
        lnbd.y = y + 1;
        x = 1;
        prev = 0;
    }                           /* end of loop on y */

    return false;
}
template<typename _Tp, typename _DotTp>
static int Sklansky_( Point_<_Tp>** array, int start, int end, int* stack, int nsign, int sign2 )
{
    int incr = end > start ? 1 : -1;
    // prepare first triangle
    int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
    int stacksize = 3;

    if( start == end ||
       (array[start]->x == array[end]->x &&
        array[start]->y == array[end]->y) )
    {
        stack[0] = start;
        return 1;
    }

    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;

    end += incr; // make end = afterend

    while( pnext != end )
    {
        // check the angle p1,p2,p3
        _Tp cury = array[pcur]->y;
        _Tp nexty = array[pnext]->y;
        _Tp by = nexty - cury;

        if( CV_SIGN( by ) != nsign )
        {
            _Tp ax = array[pcur]->x - array[pprev]->x;
            _Tp bx = array[pnext]->x - array[pcur]->x;
            _Tp ay = cury - array[pprev]->y;
            _DotTp convexity = (_DotTp)ay*bx - (_DotTp)ax*by; // if >0 then convex angle

            if( CV_SIGN( convexity ) == sign2 && (ax != 0 || ay != 0) )
            {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize++;
            }
            else
            {
                if( pprev == start )
                {
                    pcur = pnext;
                    stack[1] = pcur;
                    pnext += incr;
                    stack[2] = pnext;
                }
                else
                {
                    stack[stacksize-2] = pnext;
                    pcur = pprev;
                    pprev = stack[stacksize-4];
                    stacksize--;
                }
            }
        }
        else
        {
            pnext += incr;
            stack[stacksize-1] = pnext;
        }
    }

    return --stacksize;
}
enum { CALIPERS_MAXHEIGHT=0, CALIPERS_MINAREARECT=1, CALIPERS_MAXDIST=2 };
static void rotatingCalipers( const Point2f* points, int n, int mode, float* out )
{
#ifdef _MSC_VER
    float minarea = FLT_MAX;
#else
    float minarea = __FLT_MAX__;
#endif
    float max_dist = 0;
    char buffer[32] = {};
    int i, k;
    std::vector<float> abuf(n*3);
    float* inv_vect_length = abuf.data();
    Point2f* vect = (Point2f*)(inv_vect_length + n);
    int left = 0, bottom = 0, right = 0, top = 0;
    int seq[4] = { -1, -1, -1, -1 };

    /* rotating calipers sides will always have coordinates
     (a,b) (-b,a) (-a,-b) (b, -a)
     */
    /* this is a first base vector (a,b) initialized by (1,0) */
    float orientation = 0;
    float base_a;
    float base_b = 0;

    float left_x, right_x, top_y, bottom_y;
    Point2f pt0 = points[0];

    left_x = right_x = pt0.x;
    top_y = bottom_y = pt0.y;

    for( i = 0; i < n; i++ )
    {
        double dx, dy;

        if( pt0.x < left_x )
            left_x = pt0.x, left = i;

        if( pt0.x > right_x )
            right_x = pt0.x, right = i;

        if( pt0.y > top_y )
            top_y = pt0.y, top = i;

        if( pt0.y < bottom_y )
            bottom_y = pt0.y, bottom = i;

        Point2f pt = points[(i+1) & (i+1 < n ? -1 : 0)];

        dx = pt.x - pt0.x;
        dy = pt.y - pt0.y;

        vect[i].x = (float)dx;
        vect[i].y = (float)dy;
        inv_vect_length[i] = (float)(1./std::sqrt(dx*dx + dy*dy));

        pt0 = pt;
    }

    // find convex hull orientation
    {
        double ax = vect[n-1].x;
        double ay = vect[n-1].y;

        for( i = 0; i < n; i++ )
        {
            double bx = vect[i].x;
            double by = vect[i].y;

            double convexity = ax * by - ay * bx;

            if( convexity != 0 )
            {
                orientation = (convexity > 0) ? 1.f : (-1.f);
                break;
            }
            ax = bx;
            ay = by;
        }
        MNN_ASSERT( orientation != 0 );
    }
    base_a = orientation;

    /*****************************************************************************************/
    /*                         init calipers position                                        */
    seq[0] = bottom;
    seq[1] = right;
    seq[2] = top;
    seq[3] = left;
    /*****************************************************************************************/
    /*                         Main loop - evaluate angles and rotate calipers               */

    /* all of edges will be checked while rotating calipers by 90 degrees */
    for( k = 0; k < n; k++ )
    {
        /* sinus of minimal angle */
        /*float sinus;*/

        /* compute cosine of angle between calipers side and polygon edge */
        /* dp - dot product */
        float dp[4] = {
            +base_a * vect[seq[0]].x + base_b * vect[seq[0]].y,
            -base_b * vect[seq[1]].x + base_a * vect[seq[1]].y,
            -base_a * vect[seq[2]].x - base_b * vect[seq[2]].y,
            +base_b * vect[seq[3]].x - base_a * vect[seq[3]].y,
        };

        float maxcos = dp[0] * inv_vect_length[seq[0]];

        /* number of calipers edges, that has minimal angle with edge */
        int main_element = 0;

        /* choose minimal angle */
        for ( i = 1; i < 4; ++i )
        {
            float cosalpha = dp[i] * inv_vect_length[seq[i]];
            if (cosalpha > maxcos)
            {
                main_element = i;
                maxcos = cosalpha;
            }
        }

        /*rotate calipers*/
        {
            //get next base
            int pindex = seq[main_element];
            float lead_x = vect[pindex].x*inv_vect_length[pindex];
            float lead_y = vect[pindex].y*inv_vect_length[pindex];
            switch( main_element )
            {
            case 0:
                base_a = lead_x;
                base_b = lead_y;
                break;
            case 1:
                base_a = lead_y;
                base_b = -lead_x;
                break;
            case 2:
                base_a = -lead_x;
                base_b = -lead_y;
                break;
            case 3:
                base_a = -lead_y;
                base_b = lead_x;
                break;
            default:
                MNN_ERROR("main_element should be 0, 1, 2 or 3");
            }
        }
        /* change base point of main edge */
        seq[main_element] += 1;
        seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];

        switch (mode)
        {
        case CALIPERS_MAXHEIGHT:
            {
            /* now main element lies on edge aligned to calipers side */

            /* find opposite element i.e. transform  */
            /* 0->2, 1->3, 2->0, 3->1                */
            int opposite_el = main_element ^ 2;

            float dx = points[seq[opposite_el]].x - points[seq[main_element]].x;
            float dy = points[seq[opposite_el]].y - points[seq[main_element]].y;
            float dist;

            if( main_element & 1 )
                dist = (float)fabs(dx * base_a + dy * base_b);
            else
                dist = (float)fabs(dx * (-base_b) + dy * base_a);

            if( dist > max_dist )
                max_dist = dist;
            }
            break;
        case CALIPERS_MINAREARECT:
            /* find area of rectangle */
            {
            float height;
            float area;

            /* find vector left-right */
            float dx = points[seq[1]].x - points[seq[3]].x;
            float dy = points[seq[1]].y - points[seq[3]].y;

            /* dotproduct */
            float width = dx * base_a + dy * base_b;

            /* find vector left-right */
            dx = points[seq[2]].x - points[seq[0]].x;
            dy = points[seq[2]].y - points[seq[0]].y;

            /* dotproduct */
            height = -dx * base_b + dy * base_a;

            area = width * height;
            if( area <= minarea )
            {
                float *buf = (float *) buffer;

                minarea = area;
                /* leftist point */
                ((int *) buf)[0] = seq[3];
                buf[1] = base_a;
                buf[2] = width;
                buf[3] = base_b;
                buf[4] = height;
                /* bottom point */
                ((int *) buf)[5] = seq[0];
                buf[6] = area;
            }
            }
            break;
        }                       /*switch */
    }                           /* for */

    switch (mode)
    {
    case CALIPERS_MINAREARECT:
        {
        float *buf = (float *) buffer;

        float A1 = buf[1];
        float B1 = buf[3];

        float A2 = -buf[3];
        float B2 = buf[1];

        float C1 = A1 * points[((int *) buf)[0]].x + points[((int *) buf)[0]].y * B1;
        float C2 = A2 * points[((int *) buf)[5]].x + points[((int *) buf)[5]].y * B2;

        float idet = 1.f / (A1 * B2 - A2 * B1);

        float px = (C1 * B2 - C2 * B1) * idet;
        float py = (A1 * C2 - A2 * C1) * idet;

        out[0] = px;
        out[1] = py;

        out[2] = A1 * buf[2];
        out[3] = B1 * buf[2];

        out[4] = A2 * buf[4];
        out[5] = B2 * buf[4];
        }
        break;
    case CALIPERS_MAXHEIGHT:
        {
        out[0] = max_dist;
        }
        break;
    }
}
enum ConnectedComponentsTypes {
    CC_STAT_LEFT   = 0, //!< The leftmost (x) coordinate which is the inclusive start of the bounding
                        //!< box in the horizontal direction.
    CC_STAT_TOP    = 1, //!< The topmost (y) coordinate which is the inclusive start of the bounding
                        //!< box in the vertical direction.
    CC_STAT_WIDTH  = 2, //!< The horizontal size of the bounding box
    CC_STAT_HEIGHT = 3, //!< The vertical size of the bounding box
    CC_STAT_AREA   = 4, //!< The total area (in pixels) of the connected component
    CC_STAT_MAX    = 5 //!< Max enumeration value. Used internally only for memory allocation
};
struct Point2ui64{
     uint64_t x, y;
     Point2ui64(uint64_t _x, uint64_t _y) :x(_x), y(_y){}
};
struct CCStatsOp{
    VARP& statsv;
    VARP& centroidsv;
    std::vector<Point2ui64> integrals;
    int _nextLoc;
    int _nlabels;

    CCStatsOp(VARP& _statsv, VARP& _centroidsv) : statsv(_statsv), centroidsv(_centroidsv), _nextLoc(0){}

    inline
    void init(int nlabels){
        _nlabels = nlabels;
        std::vector<int> statsv_data(nlabels * CC_STAT_MAX);
        std::vector<float> centroidsv_data(nlabels * 2, 0);
        for (int l = 0; l < nlabels; ++l){
            int *row = statsv_data.data() + l * CC_STAT_MAX;
            row[CC_STAT_LEFT] = std::numeric_limits<int>::max();
            row[CC_STAT_TOP] = std::numeric_limits<int>::max();
            row[CC_STAT_WIDTH] = std::numeric_limits<int>::min();
            row[CC_STAT_HEIGHT] = std::numeric_limits<int>::min();
            row[CC_STAT_AREA] = 0;
        }
        statsv = _Const(statsv_data.data(), {nlabels, CC_STAT_MAX}, NHWC, halide_type_of<int>());
        centroidsv = _Const(centroidsv_data.data(), {nlabels, 2}, NHWC, halide_type_of<float>());
        integrals.resize(nlabels, Point2ui64(0, 0));
    }

    void operator()(int r, int c, int l){
        #define MIN(a,b)  ((a) > (b) ? (b) : (a))
        #define MAX(a,b)  ((a) < (b) ? (b) : (a))
        int *row = statsv->writeMap<int>() + l * CC_STAT_MAX;
        row[CC_STAT_LEFT] = MIN(row[CC_STAT_LEFT], c);
        row[CC_STAT_WIDTH] = MAX(row[CC_STAT_WIDTH], c);
        row[CC_STAT_TOP] = MIN(row[CC_STAT_TOP], r);
        row[CC_STAT_HEIGHT] = MAX(row[CC_STAT_HEIGHT], r);
        row[CC_STAT_AREA]++;
        Point2ui64& integral = integrals[l];
        integral.x += c;
        integral.y += r;
    }

    void finish(){
        int* stats_ptr = statsv->writeMap<int>();
        float* centroid_ptr = centroidsv->writeMap<float>();
        for (int l = 0; l < _nlabels; ++l){
            int *row = stats_ptr + l * CC_STAT_MAX;
            float area = ((unsigned*)row)[CC_STAT_AREA];
            float *centroid = centroid_ptr + l * 2;
            if (area > 0){
                row[CC_STAT_WIDTH] = row[CC_STAT_WIDTH] - row[CC_STAT_LEFT] + 1;
                row[CC_STAT_HEIGHT] = row[CC_STAT_HEIGHT] - row[CC_STAT_TOP] + 1;
                Point2ui64& integral = integrals[l];
                centroid[0] = float(integral.x) / area;
                centroid[1] = float(integral.y) / area;
            } else {
                row[CC_STAT_WIDTH] = 0;
                row[CC_STAT_HEIGHT] = 0;
                row[CC_STAT_LEFT] = -1;
                centroid[0] = std::numeric_limits<float>::quiet_NaN();
                centroid[1] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
};
template<typename LabelT>
inline static
LabelT findRoot(const LabelT *P, LabelT i){
    LabelT root = i;
    while (P[root] < root){
        root = P[root];
    }
    return root;
}
template<typename LabelT>
inline static
void setRoot(LabelT *P, LabelT i, LabelT root){
    while (P[i] < i){
        LabelT j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}
template<typename LabelT>
inline static
LabelT find(LabelT *P, LabelT i){
    LabelT root = findRoot(P, i);
    setRoot(P, i, root);
    return root;
}
template<typename LabelT>
inline static
LabelT set_union(LabelT *P, LabelT i, LabelT j){
    LabelT root = findRoot(P, i);
    if (i != j){
        LabelT rootj = findRoot(P, j);
        if (root > rootj){
            root = rootj;
        }
        setRoot(P, j, root);
    }
    setRoot(P, i, root);
    return root;
}
template<typename LabelT>
inline static
LabelT flattenL(LabelT *P, LabelT length){
    LabelT k = 1;
    for (LabelT i = 1; i < length; ++i){
        if (P[i] < i){
            P[i] = P[P[i]];
        }
        else{
            P[i] = k; k = k + 1;
        }
    }
    return k;
}
template<typename LabelT>
inline static
void flattenL(LabelT *P, const int start, const int nElem, LabelT& k){
    for (int i = start; i < start + nElem; ++i){
        if (P[i] < i){//node that point to root
            P[i] = P[P[i]];
        }
        else{ //for root node
            P[i] = k;
            k = k + 1;
        }
    }
}
template<typename LabelT, typename PixelT>
LabelT LabelingGrana(VARP img, VARP& imgLabels, int connectivity, CCStatsOp& sop) {
    int h, w, c;
    getVARPSize(img, &h, &w, &c);
    std::vector<int> label(h * w, 0);
    imgLabels = _Const(label.data(), {1, h, w, 1}, NHWC, halide_type_of<int>());
    int img_step = w * c, label_step = w;
    //A quick and dirty upper bound for the maximum number of labels.
    //Following formula comes from the fact that a 2x2 block in 8-connectivity case
    //can never have more than 1 new label and 1 label for background.
    //Worst case image example pattern:
    //1 0 1 0 1...
    //0 0 0 0 0...
    //1 0 1 0 1...
    //............
    const size_t Plength = size_t(((h + 1) / 2) * size_t((w + 1) / 2)) + 1;
    std::vector<LabelT> P_(Plength, 0);
    LabelT *P = P_.data();
    LabelT lunique = 1;
    // First scan
    for (int r = 0; r < h; r += 2) {
        // Get rows pointer
        const PixelT * const img_row = img->readMap<PixelT>() + r * img_step;
        const PixelT * const img_row_prev = img_row - img_step;
        const PixelT * const img_row_prev_prev = img_row_prev - img_step;
        const PixelT * const img_row_fol = img_row + img_step;
        LabelT * const imgLabels_row = imgLabels->writeMap<LabelT>() + r * label_step;
        LabelT * const imgLabels_row_prev_prev = imgLabels_row - label_step - label_step;
        for (int c = 0; c < w; c += 2) {
            #define condition_b c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
            #define condition_c r-2>=0 && img_row_prev_prev[c]>0
            #define condition_d c+1<w&& r-2>=0 && img_row_prev_prev[c+1]>0
            #define condition_e c+2<w  && r-1>=0 && img_row_prev[c-1]>0

            #define condition_g c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
            #define condition_h c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
            #define condition_i r-1>=0 && img_row_prev[c]>0
            #define condition_j c+1<w && r-1>=0 && img_row_prev[c+1]>0
            #define condition_k c+2<w && r-1>=0 && img_row_prev[c+2]>0

            #define condition_m c-2>=0 && img_row[c-2]>0
            #define condition_n c-1>=0 && img_row[c-1]>0
            #define condition_o img_row[c]>0
            #define condition_p c+1<w && img_row[c+1]>0

            #define condition_r c-1>=0 && r+1<h && img_row_fol[c-1]>0
            #define condition_s r+1<h && img_row_fol[c]>0
            #define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0
            if (condition_o) {
                if (condition_n) {
                    if (condition_j) {
                        if (condition_i) {
                            //Action_6: Assign label of block S
                            imgLabels_row[c] = imgLabels_row[c - 2];
                            continue;
                        }
                        else {
                            if (condition_c) {
                                if (condition_h) {
                                    //Action_6: Assign label of block S
                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                    continue;
                                }
                                else {
                                    if (condition_g) {
                                        if (condition_b) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_11: Merge labels of block Q and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                            }
                            else {
                                //Action_11: Merge labels of block Q and S
                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                continue;
                            }
                        }
                    }
                    else {
                        if (condition_p) {
                            if (condition_k) {
                                if (condition_d) {
                                    if (condition_i) {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_c) {
                                            if (condition_h) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_12: Merge labels of block R and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    //Action_12: Merge labels of block R and S
                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                    continue;
                                }
                            }
                            else {
                                //Action_6: Assign label of block S
                                imgLabels_row[c] = imgLabels_row[c - 2];
                                continue;
                            }
                        }
                        else {
                            //Action_6: Assign label of block S
                            imgLabels_row[c] = imgLabels_row[c - 2];
                            continue;
                        }
                    }
                }
                else {
                    if (condition_r) {
                        if (condition_j) {
                            if (condition_m) {
                                if (condition_h) {
                                    if (condition_i) {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        if (condition_c) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_g) {
                                        if (condition_b) {
                                            if (condition_i) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_c) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_11: Merge labels of block Q and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_i) {
                                    //Action_11: Merge labels of block Q and S
                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                    continue;
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_c) {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                        else {
                                            //Action_14: Merge labels of block P, Q and S
                                            imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_11: Merge labels of block Q and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                            }
                        }
                        else {
                            if (condition_p) {
                                if (condition_k) {
                                    if (condition_m) {
                                        if (condition_h) {
                                            if (condition_d) {
                                                if (condition_i) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_c) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_d) {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        if (condition_i) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            if (condition_c) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                if (condition_i) {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_16: labels of block Q, R and S
                                                            imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_16: labels of block Q, R and S
                                                        imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_12: Merge labels of block R and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_d) {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                            else {
                                                //Action_16: labels of block Q, R and S
                                                imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_h) {
                                                if (condition_d) {
                                                    if (condition_c) {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_15: Merge labels of block P, R and S
                                                        imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_15: Merge labels of block P, R and S
                                                    imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                }
                                else {
                                    if (condition_h) {
                                        if (condition_m) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_9 Merge labels of block P and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_m) {
                                                if (condition_g) {
                                                    if (condition_b) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_h) {
                                    if (condition_m) {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                    else {
                                        // ACTION_9 Merge labels of block P and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        if (condition_m) {
                                            if (condition_g) {
                                                if (condition_b) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_11: Merge labels of block Q and S
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        //Action_6: Assign label of block S
                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    else {
                        if (condition_j) {
                            if (condition_i) {
                                //Action_4: Assign label of block Q
                                imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                continue;
                            }
                            else {
                                if (condition_h) {
                                    if (condition_c) {
                                        //Action_4: Assign label of block Q
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        //Action_7: Merge labels of block P and Q
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
                                        continue;
                                    }
                                }
                                else {
                                    //Action_4: Assign label of block Q
                                    imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                    continue;
                                }
                            }
                        }
                        else {
                            if (condition_p) {
                                if (condition_k) {
                                    if (condition_i) {
                                        if (condition_d) {
                                            //Action_5: Assign label of block R
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                            continue;
                                        }
                                        else {
                                            // ACTION_10 Merge labels of block Q and R
                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_h) {
                                            if (condition_d) {
                                                if (condition_c) {
                                                    //Action_5: Assign label of block R
                                                    imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                    continue;
                                                }
                                                else {
                                                    //Action_8: Merge labels of block P and R
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_8: Merge labels of block P and R
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_5: Assign label of block R
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                            continue;
                                        }
                                    }
                                }
                                else {
                                    if (condition_i) {
                                        //Action_4: Assign label of block Q
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                        continue;
                                    }
                                    else {
                                        if (condition_h) {
                                            //Action_3: Assign label of block P
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
                                            continue;
                                        }
                                        else {
                                            //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                            imgLabels_row[c] = lunique;
                                            P[lunique] = lunique;
                                            lunique = lunique + 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_i) {
                                    //Action_4: Assign label of block Q
                                    imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                    continue;
                                }
                                else {
                                    if (condition_h) {
                                        //Action_3: Assign label of block P
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
                                        continue;
                                    }
                                    else {
                                        //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                        imgLabels_row[c] = lunique;
                                        P[lunique] = lunique;
                                        lunique = lunique + 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else {
                if (condition_s) {
                    if (condition_p) {
                        if (condition_n) {
                            if (condition_j) {
                                if (condition_i) {
                                    //Action_6: Assign label of block S
                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                    continue;
                                }
                                else {
                                    if (condition_c) {
                                        if (condition_h) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_g) {
                                                if (condition_b) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        //Action_11: Merge labels of block Q and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                            }
                            else {
                                if (condition_k) {
                                    if (condition_d) {
                                        if (condition_i) {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                        else {
                                            if (condition_c) {
                                                if (condition_h) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        //Action_12: Merge labels of block R and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                                else {
                                    //Action_6: Assign label of block S
                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                    continue;
                                }
                            }
                        }
                        else {
                            if (condition_r) {
                                if (condition_j) {
                                    if (condition_m) {
                                        if (condition_h) {
                                            if (condition_i) {
                                                //Action_6: Assign label of block S
                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                continue;
                                            }
                                            else {
                                                if (condition_c) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                        }
                                        else {
                                            if (condition_g) {
                                                if (condition_b) {
                                                    if (condition_i) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_c) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_11: Merge labels of block Q and S
                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        //Action_11: Merge labels of block Q and S
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                        continue;
                                    }
                                }
                                else {
                                    if (condition_k) {
                                        if (condition_d) {
                                            if (condition_m) {
                                                if (condition_h) {
                                                    if (condition_i) {
                                                        //Action_6: Assign label of block S
                                                        imgLabels_row[c] = imgLabels_row[c - 2];
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_c) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            if (condition_i) {
                                                                //Action_6: Assign label of block S
                                                                imgLabels_row[c] = imgLabels_row[c - 2];
                                                                continue;
                                                            }
                                                            else {
                                                                if (condition_c) {
                                                                    //Action_6: Assign label of block S
                                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                                    continue;
                                                                }
                                                                else {
                                                                    //Action_12: Merge labels of block R and S
                                                                    imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                    continue;
                                                                }
                                                            }
                                                        }
                                                        else {
                                                            //Action_12: Merge labels of block R and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            if (condition_i) {
                                                if (condition_m) {
                                                    if (condition_h) {
                                                        //Action_12: Merge labels of block R and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                    else {
                                                        if (condition_g) {
                                                            if (condition_b) {
                                                                //Action_12: Merge labels of block R and S
                                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                            else {
                                                                //Action_16: labels of block Q, R and S
                                                                imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                                continue;
                                                            }
                                                        }
                                                        else {
                                                            //Action_16: labels of block Q, R and S
                                                            imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                }
                                                else {
                                                    //Action_16: labels of block Q, R and S
                                                    imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
                                                    continue;
                                                }
                                            }
                                            else {
                                                //Action_12: Merge labels of block R and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            if (condition_m) {
                                                if (condition_h) {
                                                    //Action_6: Assign label of block S
                                                    imgLabels_row[c] = imgLabels_row[c - 2];
                                                    continue;
                                                }
                                                else {
                                                    if (condition_g) {
                                                        if (condition_b) {
                                                            //Action_6: Assign label of block S
                                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                                            continue;
                                                        }
                                                        else {
                                                            //Action_11: Merge labels of block Q and S
                                                            imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                            continue;
                                                        }
                                                    }
                                                    else {
                                                        //Action_11: Merge labels of block Q and S
                                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                        continue;
                                                    }
                                                }
                                            }
                                            else {
                                                //Action_11: Merge labels of block Q and S
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_6: Assign label of block S
                                            imgLabels_row[c] = imgLabels_row[c - 2];
                                            continue;
                                        }
                                    }
                                }
                            }
                            else {
                                if (condition_j) {
                                    //Action_4: Assign label of block Q
                                    imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                    continue;
                                }
                                else {
                                    if (condition_k) {
                                        if (condition_i) {
                                            if (condition_d) {
                                                //Action_5: Assign label of block R
                                                imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                                continue;
                                            }
                                            else {
                                                // ACTION_10 Merge labels of block Q and R
                                                imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
                                                continue;
                                            }
                                        }
                                        else {
                                            //Action_5: Assign label of block R
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                            continue;
                                        }
                                    }
                                    else {
                                        if (condition_i) {
                                            //Action_4: Assign label of block Q
                                            imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                            continue;
                                        }
                                        else {
                                            //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                            imgLabels_row[c] = lunique;
                                            P[lunique] = lunique;
                                            lunique = lunique + 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else {
                        if (condition_r) {
                            //Action_6: Assign label of block S
                            imgLabels_row[c] = imgLabels_row[c - 2];
                            continue;
                        }
                        else {
                            if (condition_n) {
                                //Action_6: Assign label of block S
                                imgLabels_row[c] = imgLabels_row[c - 2];
                                continue;
                            }
                            else {
                                //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                imgLabels_row[c] = lunique;
                                P[lunique] = lunique;
                                lunique = lunique + 1;
                                continue;
                            }
                        }
                    }
                }
                else {
                    if (condition_p) {
                        if (condition_j) {
                            //Action_4: Assign label of block Q
                            imgLabels_row[c] = imgLabels_row_prev_prev[c];
                            continue;
                        }
                        else {
                            if (condition_k) {
                                if (condition_i) {
                                    if (condition_d) {
                                        //Action_5: Assign label of block R
                                        imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                        continue;
                                    }
                                    else {
                                        // ACTION_10 Merge labels of block Q and R
                                        imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
                                        continue;
                                    }
                                }
                                else {
                                    //Action_5: Assign label of block R
                                    imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
                                    continue;
                                }
                            }
                            else {
                                if (condition_i) {
                                    //Action_4: Assign label of block Q
                                    imgLabels_row[c] = imgLabels_row_prev_prev[c];
                                    continue;
                                }
                                else {
                                    //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                                    imgLabels_row[c] = lunique;
                                    P[lunique] = lunique;
                                    lunique = lunique + 1;
                                    continue;
                                }
                            }
                        }
                    }
                    else {
                        if (condition_t) {
                            //Action_2: New label (the block has foreground pixels and is not connected to anything else)
                            imgLabels_row[c] = lunique;
                            P[lunique] = lunique;
                            lunique = lunique + 1;
                            continue;
                        }
                        else {
                            // Action_1: No action (the block has no foreground pixels)
                            imgLabels_row[c] = 0;
                            continue;
                        }
                    }
                }
            }
        }
    }

    // Second scan + analysis
    LabelT nLabels = flattenL(P, lunique);
    sop.init(nLabels);
    if (h & 1){
        if (w & 1){
            // Case 1: both rows and cols odd
            for (int r = 0; r < h; r += 2) {
                // Get rows pointer
                const PixelT * const img_row = img->readMap<PixelT>() + r * img_step;
                const PixelT * const img_row_fol = img_row + img_step;
                LabelT * const imgLabels_row = imgLabels->writeMap<LabelT>() + r * label_step;
                LabelT * const imgLabels_row_fol = imgLabels_row + label_step;

                for (int c = 0; c < w; c += 2) {
                    LabelT iLabel = imgLabels_row[c];
                    if (iLabel > 0) {
                        iLabel = P[iLabel];
                        if (img_row[c] > 0){
                            imgLabels_row[c] = iLabel;
                            sop(r, c, iLabel);
                        }
                        else{
                            imgLabels_row[c] = 0;
                            sop(r, c, 0);
                        }
                        if (c + 1 < w) {
                            if (img_row[c + 1] > 0){
                                imgLabels_row[c + 1] = iLabel;
                                sop(r, c + 1, iLabel);
                            }
                            else{
                                imgLabels_row[c + 1] = 0;
                                sop(r, c + 1, 0);
                            }
                            if (r + 1 < h) {
                                if (img_row_fol[c] > 0){
                                    imgLabels_row_fol[c] = iLabel;
                                    sop(r + 1, c, iLabel);
                                }
                                else{
                                    imgLabels_row_fol[c] = 0;
                                    sop(r + 1, c, 0);
                                }
                                if (img_row_fol[c + 1] > 0){
                                    imgLabels_row_fol[c + 1] = iLabel;
                                    sop(r + 1, c + 1, iLabel);
                                }
                                else{
                                    imgLabels_row_fol[c + 1] = 0;
                                    sop(r + 1, c + 1, 0);
                                }
                            }
                        }
                        else if (r + 1 < h) {
                            if (img_row_fol[c] > 0){
                                imgLabels_row_fol[c] = iLabel;
                                sop(r + 1, c, iLabel);
                            }
                            else{
                                imgLabels_row_fol[c] = 0;
                                sop(r + 1, c, 0);
                            }
                        }
                    }
                    else {
                        imgLabels_row[c] = 0;
                        sop(r, c, 0);
                        if (c + 1 < w) {
                            imgLabels_row[c + 1] = 0;
                            sop(r, c + 1, 0);
                            if (r + 1 < h) {
                                imgLabels_row_fol[c] = 0;
                                imgLabels_row_fol[c + 1] = 0;
                                sop(r + 1, c, 0);
                                sop(r + 1, c + 1, 0);
                            }
                        }
                        else if (r + 1 < h) {
                            imgLabels_row_fol[c] = 0;
                            sop(r + 1, c, 0);
                        }
                    }
                }
            }
        }//END Case 1
        else{
            //Case 2: only rows odd
            for (int r = 0; r < h; r += 2) {
                // Get rows pointer
                const PixelT * const img_row = img->readMap<PixelT>() + r * img_step;
                const PixelT * const img_row_fol = img_row + img_step;
                LabelT * const imgLabels_row = imgLabels->writeMap<LabelT>() + r * label_step;
                LabelT * const imgLabels_row_fol = imgLabels_row + label_step;

                for (int c = 0; c < w; c += 2) {
                    LabelT iLabel = imgLabels_row[c];
                    if (iLabel > 0) {
                        iLabel = P[iLabel];
                        if (img_row[c] > 0){
                            imgLabels_row[c] = iLabel;
                            sop(r, c, iLabel);
                        }
                        else{
                            imgLabels_row[c] = 0;
                            sop(r, c, 0);
                        }
                        if (img_row[c + 1] > 0){
                            imgLabels_row[c + 1] = iLabel;
                            sop(r, c + 1, iLabel);
                        }
                        else{
                            imgLabels_row[c + 1] = 0;
                            sop(r, c + 1, 0);
                        }
                        if (r + 1 < h) {
                            if (img_row_fol[c] > 0){
                                imgLabels_row_fol[c] = iLabel;
                                sop(r + 1, c, iLabel);
                            }
                            else{
                                imgLabels_row_fol[c] = 0;
                                sop(r + 1, c, 0);
                            }
                            if (img_row_fol[c + 1] > 0){
                                imgLabels_row_fol[c + 1] = iLabel;
                                sop(r + 1, c + 1, iLabel);
                            }
                            else{
                                imgLabels_row_fol[c + 1] = 0;
                                sop(r + 1, c + 1, 0);
                            }
                        }
                    }
                    else {
                        imgLabels_row[c] = 0;
                        imgLabels_row[c + 1] = 0;
                        sop(r, c, 0);
                        sop(r, c + 1, 0);
                        if (r + 1 < h) {
                            imgLabels_row_fol[c] = 0;
                            imgLabels_row_fol[c + 1] = 0;
                            sop(r + 1, c, 0);
                            sop(r + 1, c + 1, 0);
                        }
                    }
                }
            }
        }// END Case 2
    }
    else{
        if (w & 1){
            //Case 3: only cols odd
            for (int r = 0; r < h; r += 2) {
                // Get rows pointer
                const PixelT * const img_row = img->readMap<PixelT>() + r * img_step;
                const PixelT * const img_row_fol = img_row + img_step;
                LabelT * const imgLabels_row = imgLabels->writeMap<LabelT>() + r * label_step;
                LabelT * const imgLabels_row_fol = imgLabels_row + label_step;

                for (int c = 0; c < w; c += 2) {
                    LabelT iLabel = imgLabels_row[c];
                    if (iLabel > 0) {
                        iLabel = P[iLabel];
                        if (img_row[c] > 0){
                            imgLabels_row[c] = iLabel;
                            sop(r, c, iLabel);
                        }
                        else{
                            imgLabels_row[c] = 0;
                            sop(r, c, 0);
                        }
                        if (img_row_fol[c] > 0){
                            imgLabels_row_fol[c] = iLabel;
                            sop(r + 1, c, iLabel);
                        }
                        else{
                            imgLabels_row_fol[c] = 0;
                            sop(r + 1, c, 0);
                        }
                        if (c + 1 < w) {
                            if (img_row[c + 1] > 0){
                                imgLabels_row[c + 1] = iLabel;
                                sop(r, c + 1, iLabel);
                            }
                            else{
                                imgLabels_row[c + 1] = 0;
                                sop(r, c + 1, 0);
                            }
                            if (img_row_fol[c + 1] > 0){
                                imgLabels_row_fol[c + 1] = iLabel;
                                sop(r + 1, c + 1, iLabel);
                            }
                            else{
                                imgLabels_row_fol[c + 1] = 0;
                                sop(r + 1, c + 1, 0);
                            }
                        }
                    }
                    else{
                        imgLabels_row[c] = 0;
                        imgLabels_row_fol[c] = 0;
                        sop(r, c, 0);
                        sop(r + 1, c, 0);
                        if (c + 1 < w) {
                            imgLabels_row[c + 1] = 0;
                            imgLabels_row_fol[c + 1] = 0;
                            sop(r, c + 1, 0);
                            sop(r + 1, c + 1, 0);
                        }
                    }
                }
            }
        }// END case 3
        else{
            //Case 4: nothing odd
            for (int r = 0; r < h; r += 2) {
                // Get rows pointer
                const PixelT * const img_row = img->readMap<PixelT>() + r * img_step;
                const PixelT * const img_row_fol = img_row + img_step;
                LabelT * const imgLabels_row = imgLabels->writeMap<LabelT>() + r * label_step;
                LabelT * const imgLabels_row_fol = imgLabels_row + label_step;

                for (int c = 0; c < w; c += 2) {
                    LabelT iLabel = imgLabels_row[c];
                    if (iLabel > 0) {
                        iLabel = P[iLabel];
                        if (img_row[c] > 0){
                            imgLabels_row[c] = iLabel;
                            sop(r, c, iLabel);
                        }
                        else{
                            imgLabels_row[c] = 0;
                            sop(r, c, 0);
                        }
                        if (img_row[c + 1] > 0){
                            imgLabels_row[c + 1] = iLabel;
                            sop(r, c + 1, iLabel);
                        }
                        else{
                            imgLabels_row[c + 1] = 0;
                            sop(r, c + 1, 0);
                        }
                        if (img_row_fol[c] > 0){
                            imgLabels_row_fol[c] = iLabel;
                            sop(r + 1, c, iLabel);
                        }
                        else{
                            imgLabels_row_fol[c] = 0;
                            sop(r + 1, c, 0);
                        }
                        if (img_row_fol[c + 1] > 0){
                            imgLabels_row_fol[c + 1] = iLabel;
                            sop(r + 1, c + 1, iLabel);
                        }
                        else{
                            imgLabels_row_fol[c + 1] = 0;
                            sop(r + 1, c + 1, 0);
                        }
                    }
                    else {
                        imgLabels_row[c] = 0;
                        imgLabels_row[c + 1] = 0;
                        imgLabels_row_fol[c] = 0;
                        imgLabels_row_fol[c + 1] = 0;
                        sop(r, c, 0);
                        sop(r, c + 1, 0);
                        sop(r + 1, c, 0);
                        sop(r + 1, c + 1, 0);
                    }
                }
            }
        }//END case 4
    }
    sop.finish();
    return nLabels;
}
/*Copy From OpenCV End*/

std::vector<VARP> findContours(VARP image, int mode, int method, Point offset) {
    if (method > CHAIN_APPROX_SIMPLE) {
        MNN_ERROR("findContours: just support method = [CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE].");
    }
    auto img = _Clone(image, true);
    CvPoint off((int)offset.fX, (int)offset.fY);
    auto info = cvStartFindContours(img, off, mode, method);
    std::vector<CvPoint> points;
    std::vector<VARP> contours;
    while (cvFindNextContour(info, points)) {
        auto ptr = reinterpret_cast<int*>(points.data());
        contours.push_back(_Const(ptr, {static_cast<int>(points.size()), 1, 2}, NHWC, halide_type_of<int>()));
        points.clear();
    }
    // same to opencv
    std::reverse(contours.begin(), contours.end());
    delete info;
    return contours;
}
double contourArea(VARP _contour, bool oriented) {
    auto info = _contour->getInfo();
    int npoints = info->size / 2;
    if (!npoints) return 0;
    bool is_float = info->type == halide_type_of<float>();
    bool is_int   = info->type == halide_type_of<int>();
    MNN_ASSERT(is_float || is_int);
    double a00 = 0;
    float prevx, prevy;
    if (is_float) {
        auto ptr = _contour->readMap<float>();
        prevx = ptr[npoints * 2 - 2], prevy = ptr[npoints * 2 - 1];
        for(int i = 0; i < npoints; i++) {
            auto x = ptr[i * 2], y = ptr[i * 2 + 1];
            a00 += (double)prevx * y - (double)prevy * x;
            prevx = x, prevy = y;
        }
    } else {
        auto ptr = _contour->readMap<int>();
        prevx = ptr[npoints * 2 - 2], prevy = ptr[npoints * 2 - 1];
        for(int i = 0; i < npoints; i++) {
            float x = ptr[i * 2], y = ptr[i * 2 + 1];
            a00 += (double)prevx * y - (double)prevy * x;
            prevx = x, prevy = y;
        }
    }

    a00 *= 0.5;
    if(!oriented) a00 = fabs(a00);
    return a00;
}

std::vector<int> convexHull(VARP points, bool clockwise, bool returnPoints) {
    auto info = points->getInfo();
    auto pointPtr = points->readMap<int>();
    int i, total = info->size / 2, nout = 0;
    int miny_ind = 0, maxy_ind = 0;
    std::vector<int> _hull;
    if( total == 0 )
    {
        return _hull;
    }
    std::vector<Point2i> _points(total);
    std::vector<Point2i*> _pointer(total);
    Point2i** pointer = _pointer.data();
    std::vector<int> _stack(total + 2), _hullbuf(total);
    int* stack = _stack.data();
    int* hullbuf = _hullbuf.data();
    for( i = 0; i < total; i++ ) {
        _points[i].x = pointPtr[i * 2 + 0];
        _points[i].y = pointPtr[i * 2 + 1];
        pointer[i] = reinterpret_cast<Point2i*>(&_points[i]);
    }
    Point2i* data0 = pointer[0];

    // sort the point set by x-coordinate, find min and max y
    std::sort(pointer, pointer + total, [](Point2i* p1, Point2i* p2) {
        if( p1->x != p2->x )
            return p1->x < p2->x;
        if( p1->y != p2->y )
            return p1->y < p2->y;
        return p1 < p2;
    });
    for( i = 1; i < total; i++ )
    {
        int y = pointer[i]->y;
        if( pointer[miny_ind]->y > y )
            miny_ind = i;
        if( pointer[maxy_ind]->y < y )
            maxy_ind = i;
    }
    if( pointer[0]->x == pointer[total-1]->x &&
        pointer[0]->y == pointer[total-1]->y )
    {
        hullbuf[nout++] = 0;
    }
    else
    {
        // upper half
        int *tl_stack = stack;
        int tl_count = Sklansky_<int, int64_t>( pointer, 0, maxy_ind, tl_stack, -1, 1);
        int *tr_stack = stack + tl_count;
        int tr_count = Sklansky_<int, int64_t>( pointer, total-1, maxy_ind, tr_stack, -1, -1);

        // gather upper part of convex hull to output
        if( !clockwise )
        {
            std::swap( tl_stack, tr_stack );
            std::swap( tl_count, tr_count );
        }

        for( i = 0; i < tl_count-1; i++ )
            hullbuf[nout++] = int(pointer[tl_stack[i]] - data0);
        for( i = tr_count - 1; i > 0; i-- )
            hullbuf[nout++] = int(pointer[tr_stack[i]] - data0);
        int stop_idx = tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;

        // lower half
        int *bl_stack = stack;
        int bl_count = Sklansky_<int, int64_t>( pointer, 0, miny_ind, bl_stack, 1, -1);
        int *br_stack = stack + bl_count;
        int br_count = Sklansky_<int, int64_t>( pointer, total-1, miny_ind, br_stack, 1, 1);

        if( clockwise )
        {
            std::swap( bl_stack, br_stack );
            std::swap( bl_count, br_count );
        }

        if( stop_idx >= 0 )
        {
            int check_idx = bl_count > 2 ? bl_stack[1] :
            bl_count + br_count > 2 ? br_stack[2-bl_count] : -1;
            if( check_idx == stop_idx || (check_idx >= 0 &&
                                          pointer[check_idx]->x == pointer[stop_idx]->x &&
                                          pointer[check_idx]->y == pointer[stop_idx]->y) )
            {
                // if all the points lie on the same line, then
                // the bottom part of the convex hull is the mirrored top part
                // (except the exteme points).
                bl_count = std::min( bl_count, 2 );
                br_count = std::min( br_count, 2 );
            }
        }

        for( i = 0; i < bl_count-1; i++ )
            hullbuf[nout++] = int(pointer[bl_stack[i]] - data0);
        for( i = br_count-1; i > 0; i-- )
            hullbuf[nout++] = int(pointer[br_stack[i]] - data0);

        // try to make the convex hull indices form
        // an ascending or descending sequence by the cyclic
        // shift of the output sequence.
        if( nout >= 3 )
        {
            int min_idx = 0, max_idx = 0, lt = 0;
            for( i = 1; i < nout; i++ )
            {
                int idx = hullbuf[i];
                lt += hullbuf[i-1] < idx;
                if( lt > 1 && lt <= i-2 )
                    break;
                if( idx < hullbuf[min_idx] )
                    min_idx = i;
                if( idx > hullbuf[max_idx] )
                    max_idx = i;
            }
            int mmdist = std::abs(max_idx - min_idx);
            if( (mmdist == 1 || mmdist == nout-1) && (lt <= 1 || lt >= nout-2) )
            {
                int ascending = (max_idx + 1) % nout == min_idx;
                int i0 = ascending ? min_idx : max_idx, j = i0;
                if( i0 > 0 )
                {
                    for( i = 0; i < nout; i++ )
                    {
                        int curr_idx = stack[i] = hullbuf[j];
                        int next_j = j+1 < nout ? j+1 : 0;
                        int next_idx = hullbuf[next_j];
                        if( i < nout-1 && (ascending != (curr_idx < next_idx)) )
                            break;
                        j = next_j;
                    }
                    if( i == nout )
                        memcpy(hullbuf, stack, nout*sizeof(hullbuf[0]));
                }
            }
        }
    }
    if( returnPoints ) {
        _hull.resize(nout * 2);
        for (int i = 0; i < nout; i++) {
            _hull[2 * i] = pointPtr[_hullbuf[i] * 2];
            _hull[2 * i + 1] = pointPtr[_hullbuf[i] * 2 + 1];
        }
    } else {
        _hull.resize(nout);
        for (int i = 0; i < nout; i++) {
            _hull[i] = _hullbuf[i];
        }
    }
    return _hull;
}
RotatedRect minAreaRect(VARP _points) {
    auto hull = convexHull(_points);
    int n = hull.size() / 2;
    Point2f out[3];
    RotatedRect box;
    std::vector<Point2f> points(n);
    for (int i = 0; i < n; i++) {
        points[i].x = hull[i * 2];
        points[i].y = hull[i * 2 + 1];
    }
    auto hpoints = points.data();
    if (n > 2) {
        rotatingCalipers( hpoints, n, CALIPERS_MINAREARECT, (float*)out );
        box.center.x = out[0].x + (out[1].x + out[2].x)*0.5f;
        box.center.y = out[0].y + (out[1].y + out[2].y)*0.5f;
        box.size.width = (float)std::sqrt((double)out[1].x*out[1].x + (double)out[1].y*out[1].y);
        box.size.height = (float)std::sqrt((double)out[2].x*out[2].x + (double)out[2].y*out[2].y);
        box.angle = (float)atan2( (double)out[1].y, (double)out[1].x );
    } else if (n == 2) {
        box.center.x = (hpoints[0].x + hpoints[1].x)*0.5f;
        box.center.y = (hpoints[0].y + hpoints[1].y)*0.5f;
        double dx = hpoints[1].x - hpoints[0].x;
        double dy = hpoints[1].y - hpoints[0].y;
        box.size.width = (float)std::sqrt(dx*dx + dy*dy);
        box.size.height = 0;
        box.angle = (float)atan2( dy, dx );
    } else if (n == 1) {
        box.center = hpoints[0];
    }
    box.angle = (float)(box.angle*180/MNN_PI);
    return box;
}
Rect2i boundingRect(VARP points) {
    auto info = points->getInfo();
    int npoints = info->size / 2;
    if( npoints == 0 )
        return Rect2i();
    bool is_float = info->type == halide_type_of<float>();
    bool is_int   = info->type == halide_type_of<int>();
    MNN_ASSERT(is_float || is_int);
    int xmin = 0, ymin = 0, xmax = -1, ymax = -1;
    auto iptr = points->readMap<int>();
    auto fptr = points->readMap<float>();
    xmin = xmax = is_float ? fptr[0] : iptr[0];
    ymin = ymax = is_float ? fptr[1] : iptr[1];
    for(int i = 1; i < npoints; i++) {
        int x = is_float ? fptr[2 * i] : iptr[2 * i];
        int y = is_float ? fptr[2 * i + 1] : iptr[2 * i + 1];

        if( xmin > x )
            xmin = x;

        if( xmax < x )
            xmax = x;

        if( ymin > y )
            ymin = y;

        if( ymax < y )
            ymax = y;
    }
    return Rect2i(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}
int connectedComponentsWithStats(VARP image, VARP& labels, VARP& statsv, VARP& centroids, int connectivity) {
    MNN_ASSERT(connectivity == 8 || connectivity == 4);
    CCStatsOp sop(statsv, centroids);
    return LabelingGrana<int, uchar>(image, labels, connectivity, sop);
}

VARP boxPoints(RotatedRect box) {
    std::vector<Point> pt(4);
    double _angle = box.angle*MNN_PI/180.;
    float b = (float)cos(_angle)*0.5f;
    float a = (float)sin(_angle)*0.5f;
    pt[0].fX = box.center.x - a*box.size.height - b*box.size.width;
    pt[0].fY = box.center.y + b*box.size.height - a*box.size.width;
    pt[1].fX = box.center.x + a*box.size.height - b*box.size.width;
    pt[1].fY = box.center.y - b*box.size.height - a*box.size.width;
    pt[2].fX = 2*box.center.x - pt[0].fX;
    pt[2].fY = 2*box.center.y - pt[0].fY;
    pt[3].fX = 2*box.center.x - pt[1].fX;
    pt[3].fY = 2*box.center.y - pt[1].fY;
    return _Const(pt.data(), {4, 2});
}

} // CV
} // MNN
