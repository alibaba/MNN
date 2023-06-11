// MNN CV
// helper functions
bool isSize(PyObject* obj) {
    return (isInts(obj) && toInts(obj).size() == 2);
}
CV::Size toSize(PyObject* obj) {
    auto vals = toInts(obj);
    MNN_ASSERT(vals.size() == 2);
    return CV::Size(vals[0], vals[1]);
}
bool isPoint(PyObject* obj);
CV::Point toPoint(PyObject* obj);
bool isPoints(PyObject* obj);
std::vector<CV::Point> toPoints(PyObject* obj);
PyObject* toPyObj(std::vector<CV::Point> _points);
bool isMatrix(PyObject* obj);
CV::Matrix toMatrix(PyObject* obj);
PyObject* toPyObj(CV::Matrix m);
#if defined(PYMNN_IMGCODECS) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_haveImageReader(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    if (PyArg_ParseTuple(args, "s", &filename) && filename) {
        return toPyObj(CV::haveImageReader(filename));
    }
    PyMNN_ERROR("haveImageReader require args: (string)");
}
static PyObject* PyMNNCV_haveImageWriter(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    if (PyArg_ParseTuple(args, "s", &filename) && filename) {
        return toPyObj(CV::haveImageWriter(filename));
    }
    PyMNN_ERROR("haveImageWriter require args: (string)");
}
static PyObject* PyMNNCV_imdecode(PyObject *self, PyObject *args) {
    PyObject *buf;
    int flags = CV::IMREAD_COLOR;
    if (PyArg_ParseTuple(args, "Oi", &buf, &flags)) {
        int64_t length = -1;
        auto data = static_cast<uint8_t*>(toPtr(buf, DType_UINT8, length));
        std::vector<uint8_t> buf(data, data + length);
        return toPyObj(CV::imdecode(buf, flags));
    }
    PyMNN_ERROR("imdecode require args: (ptr, ImreadModes)");
}
static PyObject* PyMNNCV_imencode(PyObject *self, PyObject *args) {
    const char *ext = NULL;
    INTS default_param = {};
    PyObject *img, *params = nullptr /* default_param */;
    if (PyArg_ParseTuple(args, "sO|O", &ext, &img, &params) && isVar(img) && (params == nullptr || isInts(params))) {
        return toPyObj<bool, toPyObj, std::vector<uint8_t>, toPyObj>(CV::imencode(ext, toVar(img), PARSE(params, default_param, toInts)));
    }
    PyMNN_ERROR("imencode require args: (string, Var, |[int])");
}
static PyObject* PyMNNCV_imread(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    int flags = CV::IMREAD_COLOR;
    if (PyArg_ParseTuple(args, "s|i", &filename, &flags) && filename) {
        return toPyObj(CV::imread(filename, flags));
    }
    PyMNN_ERROR("imread require args: (string, ImreadModes)");
}
static PyObject* PyMNNCV_imwrite(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    INTS default_param = {};
    PyObject *img, *params = nullptr /* default_param */;
    if (PyArg_ParseTuple(args, "sO|O", &filename, &img, &params) &&
        filename && isVar(img) && (params == nullptr || isInts(params))) {
        return toPyObj(CV::imwrite(filename, toVar(img), PARSE(params, default_param, toInts)));
    }
    PyMNN_ERROR("imwrite require args: (string, Var, |[int])");
}
#endif
#if defined(PYMNN_CALIB3D) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_Rodrigues(PyObject *self, PyObject *args) {
    PyObject *src;
    if (PyArg_ParseTuple(args, "O", &src) && isVar(src)) {
        return toPyObj(CV::Rodrigues(toVar(src)));
    }
    PyMNN_ERROR("Rodrigues require args: (Var)");
}
static PyObject* PyMNNCV_solvePnP(PyObject *self, PyObject *args) {
    int useExtrinsicGuess = 0;
    PyObject *objPoints, *imgPoints, *cameraMatrix, *distCoeffs;
    if (PyArg_ParseTuple(args, "OOOO|i", &objPoints, &imgPoints, &cameraMatrix, &distCoeffs, &useExtrinsicGuess) &&
        isVar(objPoints) && isVar(imgPoints) && isVar(cameraMatrix) && isVar(distCoeffs)) {
        return toPyObj<VARP, toPyObj, VARP, toPyObj>(CV::solvePnP(toVar(objPoints), toVar(imgPoints), toVar(cameraMatrix),
                                    toVar(distCoeffs), useExtrinsicGuess));
    }
    PyMNN_ERROR("solvePnP require args: (Var, Var, Var, Var, |bool)");
}
#endif
// core
#if defined(PYMNN_CVCORE) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_solve(PyObject *self, PyObject *args) {
    PyObject *src1, *src2;
    int method = 0;
    if (PyArg_ParseTuple(args, "OO|i", &src1, &src2, &method) && isVar(src1) && isVar(src2)) {
        return toPyObj<bool, toPyObj, VARP, toPyObj>(CV::solve(toVar(src1), toVar(src2), method));
    }
    PyMNN_ERROR("solve require args: (Var, Var, |int)");
}
#endif
// color
#if defined(PYMNN_IMGPROC_COLOR) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_cvtColor(PyObject *self, PyObject *args) {
    PyObject *src;
    int code, dstCn = 0;
    if (PyArg_ParseTuple(args, "Oi|i", &src, &code, &dstCn) && isVar(src)) {
        return toPyObj(CV::cvtColor(toVar(src), code, dstCn));
    }
    PyMNN_ERROR("cvtColor require args: (Var, ColorConversionCodes, |int)");
}
static PyObject* PyMNNCV_cvtColorTwoPlane(PyObject *self, PyObject *args) {
    PyObject *src1, *src2;
    int code;
    if (PyArg_ParseTuple(args, "OOi", &src1, &src2, &code) &&
        isVar(src1) && isVar(src2)) {
        return toPyObj(CV::cvtColorTwoPlane(toVar(src1), toVar(src2), code));
    }
    PyMNN_ERROR("cvtColorTwoPlane require args: (Var, Var, ColorConversionCodes)");
}
#endif
// filter
#if defined(PYMNN_IMGPROC_FILTER) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_bilateralFilter(PyObject *self, PyObject *args) {
    PyObject *src;
    int d, borderType = 1;
    float sigmaColor, sigmaSpace;
    if (PyArg_ParseTuple(args, "Oiff|i", &src, &d, &sigmaColor, &sigmaSpace, &borderType) && isVar(src)) {
        return toPyObj(CV::bilateralFilter(toVar(src), d, sigmaColor, sigmaSpace, borderType));
    }
    PyMNN_ERROR("bilateralFilter require args: (Var, int, float, float, |BorderTypes)");
}
static PyObject* PyMNNCV_blur(PyObject *self, PyObject *args) {
    PyObject *src, *ksize;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OO|i", &src, &ksize, &borderType) &&
        isVar(src) && isSize(ksize)) {
        return toPyObj(CV::blur(toVar(src), toSize(ksize), borderType));
    }
    PyMNN_ERROR("blur require args: (Var, [int], |BorderTypes)");
}
static PyObject* PyMNNCV_boxFilter(PyObject *self, PyObject *args) {
    PyObject *src, *ksize;
    int ddepth;
    int normalize = 1;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OiO|ii", &src, &ddepth, &ksize, &normalize, &borderType) &&
        isVar(src) && isSize(ksize)) {
        return toPyObj(CV::boxFilter(toVar(src), ddepth, toSize(ksize), normalize, borderType));
    }
    PyMNN_ERROR("boxFilter require args: (Var, int, [int], |bool, BorderTypes)");
}
static PyObject* PyMNNCV_dilate(PyObject *self, PyObject *args) {
    PyObject *src, *kernel;
    int iterations = 1;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OO|ii", &src, &kernel, &iterations, &borderType) &&
        isVar(src) && isVar(kernel)) {
        return toPyObj(CV::dilate(toVar(src), toVar(kernel), iterations, borderType));
    }
    PyMNN_ERROR("dilate require args: (Var, Var, |int, BorderTypes)");
}
static PyObject* PyMNNCV_erode(PyObject *self, PyObject *args) {
    PyObject *src, *kernel;
    int iterations = 1;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OO|ii", &src, &kernel, &iterations, &borderType) &&
        isVar(src) && isVar(kernel)) {
        return toPyObj(CV::erode(toVar(src), toVar(kernel), iterations, borderType));
    }
    PyMNN_ERROR("erode require args: (Var, Var, |int, BorderTypes)");
}
static PyObject* PyMNNCV_filter2D(PyObject *self, PyObject *args) {
    PyObject *src, *kernel;
    int ddepth;
    float delta = 0;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OiO|fO", &src, &ddepth, &kernel, &delta, &borderType) &&
        isVar(src) && isVar(kernel)) {
        return toPyObj(CV::filter2D(toVar(src), ddepth, toVar(kernel), delta, borderType));
    }
    PyMNN_ERROR("filter2D require args: (Var, int, Var, |float, BorderTypes)");
}
static PyObject* PyMNNCV_GaussianBlur(PyObject *self, PyObject *args) {
    PyObject *src, *ksize;
    float sigmaX, sigmaY = 0;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OOf|fO", &src, &ksize, &sigmaX, &sigmaY, &borderType) &&
        isVar(src) && isSize(ksize)) {
        return toPyObj(CV::GaussianBlur(toVar(src), toSize(ksize), sigmaX, sigmaY, borderType));
    }
    PyMNN_ERROR("GaussianBlur require args: (Var, [int], float, |float, BorderTypes)");
}
static PyObject* PyMNNCV_getDerivKernels(PyObject *self, PyObject *args) {
    int dx, dy, ksize;
    int normalize = 0;
    if (PyArg_ParseTuple(args, "iii|i", &dx, &dy, &ksize, &normalize)) {
        return toPyObj<VARP, toPyObj, VARP, toPyObj>(CV::getDerivKernels(dx, dy, ksize, normalize));
    }
    PyMNN_ERROR("getDerivKernels require args: (int, int, int, |bool)");
}
static PyObject* PyMNNCV_getGaborKernel(PyObject *self, PyObject *args) {
    PyObject *ksize;
    float sigma, theta, lambd, gamma, psi = MNN_PI * 0.5;
    if (PyArg_ParseTuple(args, "Offff|f", &ksize, &sigma, &theta, &lambd, &gamma, &psi) && isSize(ksize)) {
        return toPyObj(CV::getGaborKernel(toSize(ksize), sigma, theta, lambd, gamma, psi));
    }
    PyMNN_ERROR("getGaborKernel require args: ([int], float, float, float, float, |float)");
}
static PyObject* PyMNNCV_getGaussianKernel(PyObject *self, PyObject *args) {
    int n;
    float sigma;
    if (PyArg_ParseTuple(args, "if", &n, &sigma)) {
        return toPyObj(CV::getGaussianKernel(n, sigma));
    }
    PyMNN_ERROR("getGaussianKernel require args: (int, float)");
}
static PyObject* PyMNNCV_getStructuringElement(PyObject *self, PyObject *args) {
    int shape;
    PyObject *ksize;
    if (PyArg_ParseTuple(args, "iO", &shape, &ksize) && isSize(ksize)) {
        return toPyObj(CV::getStructuringElement(shape, toSize(ksize)));
    }
    PyMNN_ERROR("getStructuringElement require args: (int, [int])");
}
static PyObject* PyMNNCV_Laplacian(PyObject *self, PyObject *args) {
    PyObject *src;
    int ddepth, ksize = 1;
    float scale = 1, delta = 0;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "Oi|iffi", &src, &ddepth, &ksize, &scale, &delta, &borderType)
        && isVar(src)) {
        return toPyObj(CV::Laplacian(toVar(src), ddepth, ksize, scale, delta, borderType));
    }
    PyMNN_ERROR("Laplacian require args: (Var, int, |int, float, float, BorderTypes)");
}
static PyObject* PyMNNCV_pyrDown(PyObject *self, PyObject *args) {
    INTS default_size = {0, 0};
    PyObject *src, *dstsize = nullptr /* default_size */;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "O|Oi", &src, &dstsize, &borderType) &&
        isVar(src) && (dstsize == nullptr || isSize(dstsize)))  {
        return toPyObj(CV::pyrDown(toVar(src),
                PARSE(dstsize, CV::Size(default_size[0], default_size[1]), toSize),
                borderType));
    }
    PyMNN_ERROR("pyrDown require args: (Var, |[int], BorderTypes)");
}
static PyObject* PyMNNCV_pyrUp(PyObject *self, PyObject *args) {
    INTS default_size = {0, 0};
    PyObject *src, *dstsize = nullptr /* default_size */;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "O|Oi", &src, &dstsize, &borderType) &&
        isVar(src) && (dstsize == nullptr || isSize(dstsize)))  {
        return toPyObj(CV::pyrUp(toVar(src),
                PARSE(dstsize, CV::Size(default_size[0], default_size[1]), toSize),
                borderType));
    }
    PyMNN_ERROR("pyrUp require args: (Var, |[int], BorderTypes)");
}
static PyObject* PyMNNCV_Scharr(PyObject *self, PyObject *args) {
    PyObject *src;
    int ddepth, dx, dy;
    float scale = 1, delta = 0;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "Oiii|ffi", &src, &ddepth, &dx, &dy, &scale, &delta, &borderType)
        && isVar(src))  {
        return toPyObj(CV::Scharr(toVar(src), ddepth, dx, dy, scale, delta, borderType));
    }
    PyMNN_ERROR("Scharr require args: (Var, int, int, int, |float, float, BorderTypes)");
}
static PyObject* PyMNNCV_sepFilter2D(PyObject *self, PyObject *args) {
    PyObject *src, *kernelX, *kernelY;
    int ddepth;
    float delta = 0;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OiOO|fi", &src, &ddepth, &kernelX, &kernelY, &delta, &borderType)
        && isVar(src) && isVar(kernelX) && isVar(kernelY))  {
        auto kx = toVar(kernelX), ky = toVar(kernelY);
        return toPyObj(CV::sepFilter2D(toVar(src), ddepth, kx, ky, delta, borderType));
    }
    PyMNN_ERROR("sepFilter2D require args: (Var, int, Var, Var, |float, BorderTypes)");
}
static PyObject* PyMNNCV_Sobel(PyObject *self, PyObject *args) {
    PyObject *src;
    int ddepth, dx, dy, ksize = 3;
    float scale = 1, delta = 0;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "Oiii|iffi", &src, &ddepth, &dx, &dy, &ksize, &scale, &delta, &borderType)
        && isVar(src))  {
        return toPyObj(CV::Sobel(toVar(src), ddepth, dx, dy, ksize, scale, delta, borderType));
    }
    PyMNN_ERROR("Sobel require args: (Var, int, int, int, |int, float, float, BorderTypes)");
}
static PyObject* PyMNNCV_spatialGradient(PyObject *self, PyObject *args) {
    PyObject *src;
    int ksize = 3;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "O|ii", &src, &ksize, &borderType)) {
        return toPyObj<VARP, toPyObj, VARP, toPyObj>(CV::spatialGradient(toVar(src), ksize, borderType));
    }
    PyMNN_ERROR("spatialGradient require args: (Var, |int, BorderTypes)");
}
static PyObject* PyMNNCV_sqrBoxFilter(PyObject *self, PyObject *args) {
    PyObject *src, *ksize;
    int ddepth;
    int normalize = 1;
    int borderType = 1;
    if (PyArg_ParseTuple(args, "OiO|ii", &src, &ddepth, &ksize, &normalize, &borderType) &&
        isVar(src) && isSize(ksize)) {
        return toPyObj(CV::sqrBoxFilter(toVar(src), ddepth, toSize(ksize), normalize, borderType));
    }
    PyMNN_ERROR("sqrBoxFilter require args: (Var, int, [int], |bool, BorderTypes)");
}
#endif
// geometric
#if defined(PYMNN_IMGPROC_GEOMETRIC) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_convertMaps(PyObject *self, PyObject *args) {
    PyObject *map1, *map2;
    int dstmap1type;
    bool nninterpolation = false;
    if (PyArg_ParseTuple(args, "OOi|i", &map1, &map2, &dstmap1type, &nninterpolation) && isVar(map1) && isVar(map2)) {
        return toPyObj<VARP, toPyObj, VARP, toPyObj>(CV::convertMaps(toVar(map1), toVar(map2), dstmap1type, nninterpolation));
    }
    PyMNN_ERROR("convertMaps require args: (Var, Var, int, |bool)");
}
static PyObject* PyMNNCV_getAffineTransform(PyObject *self, PyObject *args) {
    PyObject *src, *dst;
    if (PyArg_ParseTuple(args, "OO", &src, &dst) && isPoints(src) && isPoints(dst)) {
        return toPyObj(CV::getAffineTransform(toPoints(src).data(), toPoints(dst).data()));
    }
    PyMNN_ERROR("getAffineTransform require args: ([(float, float)], [(float, float)])");
}
static PyObject* PyMNNCV_getPerspectiveTransform(PyObject *self, PyObject *args) {
    PyObject *src, *dst;
    if (PyArg_ParseTuple(args, "OO", &src, &dst) && isPoints(src) && isPoints(dst)) {
        return toPyObj(CV::getPerspectiveTransform(toPoints(src).data(), toPoints(dst).data()));
    }
    PyMNN_ERROR("getPerspectiveTransform require args: ([(float, float)], [(float, float)])");
}
static PyObject* PyMNNCV_getRectSubPix(PyObject *self, PyObject *args) {
    PyObject *image, *patchSize, *center;
    if (PyArg_ParseTuple(args, "OOO", &image, &patchSize, &center) &&
        isVar(image) && isSize(patchSize) && isPoint(center)) {
        return toPyObj(CV::getRectSubPix(toVar(image), toSize(patchSize), toPoint(center)));
    }
    PyMNN_ERROR("getRectSubPix require args: (Var, [int], [float])");
}
static PyObject* PyMNNCV_getRotationMatrix2D(PyObject *self, PyObject *args) {
    PyObject *center;
    float angle, scale;
    if (PyArg_ParseTuple(args, "Off", &center, &angle, &scale) && isPoint(center)) {
        return toPyObj(CV::getRotationMatrix2D(toPoint(center), angle, scale));
    }
    PyMNN_ERROR("getRotationMatrix2D require args: ([float], float, float)");
}
static PyObject* PyMNNCV_invertAffineTransform(PyObject *self, PyObject *args) {
    PyObject *M;
    if (PyArg_ParseTuple(args, "O", &M) && isMatrix(M)) {
        return toPyObj(CV::invertAffineTransform(toMatrix(M)));
    }
    PyMNN_ERROR("invertAffineTransform require args: (Matrix)");
}
static PyObject* PyMNNCV_remap(PyObject *self, PyObject *args) {
    PyObject *src, *map1, *map2;
    int interpolation, borderMode = 0, borderValue = 0;
    if (PyArg_ParseTuple(args, "OOOi|ii", &src, &map1, &map2, &interpolation, &borderMode, &borderValue) &&
        isVar(src) && isVar(map1) && isVar(map2)) {
        return toPyObj(CV::remap(toVar(src), toVar(map1), toVar(map2), interpolation, borderMode, borderValue));
    }
    PyMNN_ERROR("remap require args: (Var, Var, Var, int, |int, int)");
}
static PyObject* PyMNNCV_resize(PyObject *self, PyObject *args) {
    std::vector<float> default_floats = {};
    PyObject *src, *dsize, *mean = nullptr /* default_floats */, *norm = nullptr /* default_floats */;
    float fx = 0, fy = 0;
    int code = -1, interpolation = CV::INTER_LINEAR;
    if (PyArg_ParseTuple(args, "OO|ffiiOO", &src, &dsize, &fx, &fy, &interpolation, &code, &mean, &norm) &&
        isVar(src) && isSize(dsize)
        && (mean == nullptr || isFloats(mean))
        && (norm == nullptr || isFloats(norm))) {
        return toPyObj(CV::resize(toVar(src), toSize(dsize), fx, fy, interpolation, code,
                    PARSE(mean, default_floats, toFloats),
                    PARSE(norm, default_floats, toFloats)));
    }
    PyMNN_ERROR("resize require args: (Var, [int], |float, float, InterpolationFlags, int, [float], [float])");
}
static PyObject* PyMNNCV_warpAffine(PyObject *self, PyObject *args) {
    std::vector<float> default_floats = {};
    PyObject *src, *M, *dsize, *mean = nullptr /* default_floats */, *norm = nullptr /* default_floats */;
    int borderValue = 0, code = -1, flag = CV::INTER_LINEAR, borderMode = CV::BORDER_CONSTANT;
    if (PyArg_ParseTuple(args, "OOO|iiiiOO", &src, &M, &dsize, &flag, &borderMode, &borderValue, &code, &mean, &norm) &&
        isVar(src) && isMatrix(M) && isSize(dsize)
        && (mean == nullptr || isFloats(mean))
        && (norm == nullptr || isFloats(norm))) {
        return toPyObj(CV::warpAffine(toVar(src), toMatrix(M), toSize(dsize),
                       flag, borderMode, borderValue, code,
                                PARSE(mean, default_floats, toFloats),
                                PARSE(norm, default_floats, toFloats)));
    }
    PyMNN_ERROR("warpAffine require args: (Var, Matrix, [int], |InterpolationFlags, BorderTypes, int, int, [float], [float])");
}
static PyObject* PyMNNCV_warpPerspective(PyObject *self, PyObject *args) {
    PyObject *src, *M, *dsize;
    int borderValue = 0, flag = CV::INTER_LINEAR, borderMode = CV::BORDER_CONSTANT;
    if (PyArg_ParseTuple(args, "OOO|iii", &src, &M, &dsize, &flag, &borderMode, &borderValue) &&
        isVar(src) && isMatrix(M) && isSize(dsize)) {
        return toPyObj(CV::warpPerspective(toVar(src), toMatrix(M), toSize(dsize),
                       flag, borderMode, borderValue));
    }
    PyMNN_ERROR("warpPerspective require args: (Var, Matrix, [int], |InterpolationFlags, BorderTypes, int)");
}
#endif
// miscellaneous
#if defined(PYMNN_IMGPROC_MISCELLANEOUS) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_adaptiveThreshold(PyObject *self, PyObject *args) {
    PyObject *src;
    float maxValue, C;
    int adaptiveMethod, thresholdType, blockSize;
    if (PyArg_ParseTuple(args, "Ofiiif", &src, &maxValue, &adaptiveMethod, &thresholdType, &blockSize, &C) && isVar(src)) {
        return toPyObj(CV::adaptiveThreshold(toVar(src), maxValue, adaptiveMethod, thresholdType, blockSize, C));
    }
    PyMNN_ERROR("adaptiveThreshold require args: (Var, float, int, int, int, float)");
}
static PyObject* PyMNNCV_blendLinear(PyObject *self, PyObject *args) {
    PyObject *src1, *src2, *weight1, *weight2;
    if (PyArg_ParseTuple(args, "OOOO", &src1, &src2, &weight1, &weight2) &&
        isVar(src1) && isVar(src2) && isVar(weight1) && isVar(weight2)) {
        return toPyObj(CV::blendLinear(toVar(src1), toVar(src2), toVar(weight1), toVar(weight2)));
    }
    PyMNN_ERROR("blendLinear require args: (Var, Var, Var, Var)");
}
static PyObject* PyMNNCV_threshold(PyObject *self, PyObject *args) {
    PyObject *src;
    float thresh, maxval, type;
    if (PyArg_ParseTuple(args, "Offi", &src, &thresh, &maxval, &type) && isVar(src)) {
        return toPyObj(CV::threshold(toVar(src), thresh, maxval, type));
    }
    PyMNN_ERROR("threshold require args: (Var, float, float, ThresholdTypes)");
}
#endif
// structural
#if defined(PYMNN_IMGPROC_STRUCTURAL) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_findContours(PyObject *self, PyObject *args) {
    PyObject *image, *offset = nullptr /* {0, 0} */;
    int mode, method;
    if (PyArg_ParseTuple(args, "Oii|O", &image, &mode, &method, &offset) &&
        isVar(image)
        && (offset == nullptr || isPoint(offset))) {
        CV::Point point;
        if (offset == nullptr) {
            point.set(0.f, 0.f);
        } else {
            point = toPoint(offset);
        }
        auto contours = CV::findContours(toVar(image), mode, method, point);
        PyObject* obj = PyTuple_New(2);
        PyTuple_SetItem(obj, 0, toPyObj<VARP, toPyObj>(contours));
        PyTuple_SetItem(obj, 1, toPyObj("no hierarchy"));
        return obj;
    }
    PyMNN_ERROR("findContours require args: (Var, RetrievalModes, ContourApproximationModes, [float])");
}
static PyObject* PyMNNCV_contourArea(PyObject *self, PyObject *args) {
    PyObject *points;
    int oriented = 0;
    if (PyArg_ParseTuple(args, "O|i", &points, &oriented) && isVar(points)) {
        float res = CV::contourArea(toVar(points), oriented);
        return toPyObj(res);
    }
    PyMNN_ERROR("contourArea require args: (Var, |bool)");
}
static PyObject* PyMNNCV_convexHull(PyObject *self, PyObject *args) {
    PyObject *points;
    int clockwise = 0, returnPoints = 1;
    if (PyArg_ParseTuple(args, "O|ii", &points, &clockwise, &returnPoints) && isVar(points)) {
        auto res = CV::convexHull(toVar(points), clockwise, returnPoints);
        if (returnPoints) {
            int npoints = res.size() / 2;
            return toPyObj(Express::_Const(res.data(), { npoints, 1, 2 }, NHWC, halide_type_of<int>()));
        }
        return toPyObj(res);
    }
    PyMNN_ERROR("convexHull require args: (Var, |bool, bool)");
}
static PyObject* PyMNNCV_minAreaRect(PyObject *self, PyObject *args) {
    PyObject *points;
    if (PyArg_ParseTuple(args, "O", &points) && isVar(points)) {
        auto rect = CV::minAreaRect(toVar(points));
        PyObject* center = PyTuple_New(2);
        PyTuple_SetItem(center, 0, toPyObj(rect.center.x));
        PyTuple_SetItem(center, 1, toPyObj(rect.center.y));
        PyObject* size = PyTuple_New(2);
        PyTuple_SetItem(size, 0, toPyObj(rect.size.width));
        PyTuple_SetItem(size, 1, toPyObj(rect.size.height));
        PyObject* obj = PyTuple_New(3);
        PyTuple_SetItem(obj, 0, center);
        PyTuple_SetItem(obj, 1, size);
        PyTuple_SetItem(obj, 2, toPyObj(rect.angle));
        return obj;
    }
    PyMNN_ERROR("minAreaRect require args: (Var)");
}
static PyObject* PyMNNCV_boundingRect(PyObject *self, PyObject *args) {
    PyObject *points;
    if (PyArg_ParseTuple(args, "O", &points) && isVar(points)) {
        auto rect = CV::boundingRect(toVar(points));
        std::vector<int> res { rect.x, rect.y, rect.width, rect.height };
        return toPyObj(res);
    }
    PyMNN_ERROR("boundingRect require args: (Var)");
}
static PyObject* PyMNNCV_connectedComponentsWithStats(PyObject *self, PyObject *args) {
    PyObject *image;
    int connectivity = 8;
    if (PyArg_ParseTuple(args, "O|i", &image, &connectivity) && isVar(image)) {
        VARP labels, statsv, centroids;
        int nlabels = CV::connectedComponentsWithStats(toVar(image), labels, statsv, centroids, connectivity);
        PyObject* obj = PyTuple_New(4);
        PyTuple_SetItem(obj, 0, toPyObj(nlabels));
        PyTuple_SetItem(obj, 1, toPyObj(labels));
        PyTuple_SetItem(obj, 2, toPyObj(statsv));
        PyTuple_SetItem(obj, 3, toPyObj(centroids));
        return obj;
    }
    PyMNN_ERROR("connectedComponentsWithStats require args: (Var, int)");
}
static PyObject* PyMNNCV_boxPoints(PyObject *self, PyObject *args) {
    PyObject *_box;
    if (PyArg_ParseTuple(args, "O", &_box) && PyTuple_Check(_box) && PyTuple_Size(_box) == 3) {
        auto _center = PyTuple_GetItem(_box, 0);
        auto _size = PyTuple_GetItem(_box, 1);
        auto _angle = PyTuple_GetItem(_box, 2);
        if (!(PyTuple_Check(_center) && PyTuple_Size(_center) == 2) ||
            !(PyTuple_Check(_size) && PyTuple_Size(_size) == 2) ||
            !isFloat(_angle)) goto error_;
        CV::RotatedRect box;
        box.center.x = toFloat(PyTuple_GetItem(_center, 0));
        box.center.y = toFloat(PyTuple_GetItem(_center, 1));
        box.size.width = toFloat(PyTuple_GetItem(_size, 0));
        box.size.height = toFloat(PyTuple_GetItem(_size, 1));
        box.angle = toFloat(_angle);
        return toPyObj(CV::boxPoints(box));
    }
error_:
    PyMNN_ERROR("boxPoints require args: [(float, (float, float), (float, float))])");
}
#endif
// draw
#if defined(PYMNN_IMGPROC_DRAW) || (!defined(PYMNN_USE_ALINNPYTHON))
static bool isColor(PyObject* obj) {
    return isInts(obj) || isFloats(obj);
}
CV::Scalar toColor(PyObject* obj) {
    if (isInts(obj)) {
        auto vals = toInts(obj);
        switch (vals.size()) {
            case 1:
                return CV::Scalar(vals[0], 255, 255);
            case 2:
                return CV::Scalar(vals[0], vals[1], 255);
            case 3:
                return CV::Scalar(vals[0], vals[1], vals[2]);
            case 4:
                return CV::Scalar(vals[0], vals[1], vals[2], vals[3]);
            default:
                return CV::Scalar(255, 255, 255);
        }
    } else {
        auto vals = toFloats(obj);
        switch (vals.size()) {
            case 1:
                return CV::Scalar(vals[0], 255, 255);
            case 2:
                return CV::Scalar(vals[0], vals[1], 255);
            case 3:
                return CV::Scalar(vals[0], vals[1], vals[2]);
            case 4:
                return CV::Scalar(vals[0], vals[1], vals[2], vals[3]);
            default:
                return CV::Scalar(255, 255, 255);
        }
    }
}
static PyObject* PyMNNCV_line(PyObject *self, PyObject *args) {
    PyObject *img, *pt1, *pt2, *color;
    int thickness = 1, shift = 0, linetype = CV::LINE_8;
    if (PyArg_ParseTuple(args, "OOOO|iOi", &img, &pt1, &pt2, &color, &thickness, &linetype, &shift)
        && isVar(img) && isPoint(pt1) && isPoint(pt2) && isColor(color)) {
        auto image = toVar(img);
        CV::line(image, toPoint(pt1), toPoint(pt2), toColor(color), thickness, linetype, shift);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("line require args: (Var, Point, Point, Color, |int, LineType, int)");
}
static PyObject* PyMNNCV_arrowedLine(PyObject *self, PyObject *args) {
    PyObject *img, *pt1, *pt2, *color;
    int thickness = 1, shift = 0, linetype = CV::LINE_8;
    float tipLength = 0.1;
    if (PyArg_ParseTuple(args, "OOOO|iOif", &img, &pt1, &pt2, &color, &thickness, &linetype, &shift, &tipLength)
        && isVar(img) && isPoint(pt1) && isPoint(pt2) && isColor(color)) {
        auto image = toVar(img);
        CV::arrowedLine(image, toPoint(pt1), toPoint(pt2), toColor(color),
                        thickness, linetype, shift, tipLength);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("arrowedLine require args: (Var, Point, Point, Color, |int, LineType, int, float)");
}
static PyObject* PyMNNCV_circle(PyObject *self, PyObject *args) {
    PyObject *img, *center, *color;
    int radius, thickness = 1, shift = 0, linetype = CV::LINE_8;
    if (PyArg_ParseTuple(args, "OOiO|iOi", &img, &center, &radius, &color, &thickness, &linetype, &shift)
        && isVar(img) && isPoint(center) && isColor(color)) {
        auto image = toVar(img);
        CV::circle(image, toPoint(center), radius, toColor(color), thickness, linetype, shift);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("circle require args: (Var, Point, int, Color, |int, LineType, int)");
}
static PyObject* PyMNNCV_rectangle(PyObject *self, PyObject *args) {
    PyObject *img, *pt1, *pt2, *color;
    int thickness = 1, shift = 0, linetype = CV::LINE_8;
    if (PyArg_ParseTuple(args, "OOOO|iOi", &img, &pt1, &pt2, &color, &thickness, &linetype, &shift)
        && isVar(img) && isPoint(pt1) && isPoint(pt2) && isColor(color)) {
        auto image = toVar(img);
        CV::rectangle(image, toPoint(pt1), toPoint(pt2), toColor(color), thickness, linetype, shift);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("rectangle require args: (Var, Point, Point, Color, |int, LineType, int)");
}
static PyObject* PyMNNCV_drawContours(PyObject *self, PyObject *args) {
    PyObject *img, *contours, *color;
    int contourIdx, thickness = 1, linetype = CV::LINE_8;
    if (PyArg_ParseTuple(args, "OOiO|iO", &img, &contours, &contourIdx, &color, &thickness, &linetype)
        && isVar(img) && isVec<isPoints>(contours) && isColor(color)) {
        auto image = toVar(img);
        CV::drawContours(image, toVec<std::vector<CV::Point>, toPoints>(contours), contourIdx, toColor(color), thickness, linetype);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("drawContours require args: (Var, [Points], int, Color, |int, LineType)");
}
static PyObject* PyMNNCV_fillPoly(PyObject *self, PyObject *args) {
    PyObject *img, *contours, *color, *offset = nullptr /* {0, 0} */;
    int shift = 0, linetype = CV::LINE_8;
    if (PyArg_ParseTuple(args, "OOO|OiO", &img, &contours, &color, &linetype, &shift, &offset)
        && isVar(img) && (isVec<isPoints>(contours) || isPoints(contours)) && isColor(color)
        && (offset == nullptr || isPoint(offset))) {
        auto image = toVar(img);
        CV::Point point;
        if (offset == nullptr) {
            point.set(0.f, 0.f);
        } else {
            point = toPoint(offset);
        }
        CV::fillPoly(image, toVec<std::vector<CV::Point>, toPoints>(contours), toColor(color), linetype, shift, point);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("fillPoly require args: (Var, [Points], Color, |LineType, int, Point)");
}
#endif
#if defined(PYMNN_IMGPROC_HISTOGRAMS) || (!defined(PYMNN_USE_ALINNPYTHON))
static PyObject* PyMNNCV_calcHist(PyObject *self, PyObject *args) {
    PyObject *imgs, *channels, *mask, *histSize, *ranges;
    int accumulate = 0;
    if (PyArg_ParseTuple(args, "OOOOO|i", &imgs, &channels, &mask, &histSize, &ranges, &accumulate)
        && isVars(imgs) && isInts(channels) && (isVar(mask) || isNone(mask)) && isInts(histSize) && isFloats(ranges)) {
        VARP maskVar;
        if (!isNone(mask)) { maskVar = toVar(mask); }
        return toPyObj(CV::calcHist(toVars(imgs), toInts(channels), maskVar, toInts(histSize), toFloats(ranges), accumulate));
    }
    PyMNN_ERROR("calcHist require args: ([Var], [int], (Var|None), [int], [float], |bool)");
}
#endif
static PyMethodDef PyMNNCV_methods[] = {
#if defined(PYMNN_IMGCODECS) || (!defined(PYMNN_USE_ALINNPYTHON))
    // imgcodecs
    register_methods(CV,
        haveImageReader, "haveImageReader",
        haveImageWriter, "haveImageWriter",
        imdecode, "imdecode",
        imencode, "imencode",
        imread, "imread",
        imwrite, "imwrite"
    )
#endif
#if defined(PYMNN_CVCORE) || (!defined(PYMNN_USE_ALINNPYTHON))
    // core
    register_methods(CV,
        solve, "solve"
    )
#endif
#if defined(PYMNN_CALIB3D) || (!defined(PYMNN_USE_ALINNPYTHON))
    // calib3d
    register_methods(CV,
        Rodrigues, "Rodrigues",
        solvePnP, "solvePnP"
    )
#endif
#if defined(PYMNN_IMGPROC_COLOR) || (!defined(PYMNN_USE_ALINNPYTHON))
    // color
    register_methods(CV,
        cvtColor, "cvtColor.",
        cvtColorTwoPlane, "cvtColorTwoPlane."
    )
#endif
#if defined(PYMNN_IMGPROC_FILTER) || (!defined(PYMNN_USE_ALINNPYTHON))
    // filter
    register_methods(CV,
        bilateralFilter, "bilateralFilter",
        blur, "blur",
        boxFilter, "boxFilter",
        dilate, "dilate",
        erode, "erode",
        filter2D, "filter2D",
        GaussianBlur, "GaussianBlur",
        getDerivKernels, "getDerivKernels",
        getGaborKernel, "getGaborKernel",
        getGaussianKernel, "getGaussianKernel",
        getStructuringElement, "getStructuringElement",
        Laplacian, "Laplacian",
        pyrDown, "pyrDown",
        pyrUp, "pyrUp",
        Scharr, "Scharr",
        sepFilter2D, "sepFilter2D",
        Sobel, "Sobel",
        spatialGradient, "spatialGradient",
        sqrBoxFilter, "sqrBoxFilter"
    )
#endif
#if defined(PYMNN_IMGPROC_GEOMETRIC) || (!defined(PYMNN_USE_ALINNPYTHON))
    // geometric
    register_methods(CV,
        convertMaps, "convertMaps",
        getAffineTransform, "getAffineTransform",
        getPerspectiveTransform, "getPerspectiveTransform",
        getRectSubPix, "getRectSubPix",
        getRotationMatrix2D, "getRotationMatrix2D",
        invertAffineTransform, "invertAffineTransform",
        remap, "remap",
        resize, "resize",
        warpAffine, "warpAffine",
        warpPerspective, "warpPerspective"
    )
#endif
#if defined(PYMNN_IMGPROC_MISCELLANEOUS) || (!defined(PYMNN_USE_ALINNPYTHON))
    // miscellaneous
    register_methods(CV,
        adaptiveThreshold, "adaptiveThreshold",
        blendLinear, "blendLinear",
        threshold, "threshold"
    )
#endif
#if defined(PYMNN_IMGPROC_STRUCTURAL) || (!defined(PYMNN_USE_ALINNPYTHON))
    // structural
    register_methods(CV,
        findContours, "findContours",
        contourArea, "contourArea",
        convexHull, "convexHull",
        minAreaRect, "minAreaRect",
        boundingRect, "boundingRect",
        connectedComponentsWithStats, "connectedComponentsWithStats",
        boxPoints, "boxPoints"
    )
#endif
#if defined(PYMNN_IMGPROC_DRAW) || (!defined(PYMNN_USE_ALINNPYTHON))
    // draw
    register_methods(CV,
        line, "line",
        arrowedLine, "arrowedLine",
        circle, "circle",
        rectangle, "rectangle",
        drawContours, "drawContours",
        fillPoly, "fillPoly"
    )
#endif
#if defined(PYMNN_IMGPROC_HISTOGRAMS) || (!defined(PYMNN_USE_ALINNPYTHON))
    register_methods(CV,
        calcHist, "calcHist"
    )
#endif
};
