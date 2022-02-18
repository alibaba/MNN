// MNN CV
def_enum(Format, CV::ImageFormat,
        CV::RGBA, "RGBA",
        CV::RGB, "RGB",
        CV::GRAY, "GRAY",
        CV::BGR, "BGR",
        CV::YUV_NV21, "YUV_NV21",
        CV::YUV_NV12, "YUV_NV12"
        )
def_enum(ColorConversionCodes, CV::ColorConversionCodes,
        CV::COLOR_BGR2BGRA, "COLOR_BGR2BGRA",
        CV::COLOR_RGB2RGBA, "COLOR_RGB2RGBA",
        CV::COLOR_BGRA2BGR, "COLOR_BGRA2BGR",
        CV::COLOR_RGBA2RGB, "COLOR_RGBA2RGB",
        CV::COLOR_BGR2RGBA, "COLOR_BGR2RGBA",
        CV::COLOR_RGB2BGRA, "COLOR_RGB2BGRA",
        CV::COLOR_RGBA2BGR, "COLOR_RGBA2BGR",
        CV::COLOR_BGRA2RGB, "COLOR_BGRA2RGB",
        CV::COLOR_BGR2RGB, "COLOR_BGR2RGB",
        CV::COLOR_RGB2BGR, "COLOR_RGB2BGR",
        CV::COLOR_BGRA2RGBA, "COLOR_BGRA2RGBA",
        CV::COLOR_RGBA2BGRA, "COLOR_RGBA2BGRA",
        CV::COLOR_BGR2GRAY, "COLOR_BGR2GRAY",
        CV::COLOR_RGB2GRAY, "COLOR_RGB2GRAY",
        CV::COLOR_GRAY2BGR, "COLOR_GRAY2BGR",
        CV::COLOR_GRAY2RGB, "COLOR_GRAY2RGB",
        CV::COLOR_GRAY2BGRA, "COLOR_GRAY2BGRA",
        CV::COLOR_GRAY2RGBA, "COLOR_GRAY2RGBA",
        CV::COLOR_BGRA2GRAY, "COLOR_BGRA2GRAY",
        CV::COLOR_RGBA2GRAY, "COLOR_RGBA2GRAY",
        CV::COLOR_BGR2BGR565, "COLOR_BGR2BGR565",
        CV::COLOR_RGB2BGR565, "COLOR_RGB2BGR565",
        CV::COLOR_BGR2BGR555, "COLOR_BGR2BGR555",
        CV::COLOR_RGB2BGR555, "COLOR_RGB2BGR555",
        CV::COLOR_BGR2XYZ, "COLOR_BGR2XYZ",
        CV::COLOR_RGB2XYZ, "COLOR_RGB2XYZ",
        CV::COLOR_BGR2YCrCb, "COLOR_BGR2YCrCb",
        CV::COLOR_RGB2YCrCb, "COLOR_RGB2YCrCb",
        CV::COLOR_BGR2HSV, "COLOR_BGR2HSV",
        CV::COLOR_RGB2HSV, "COLOR_RGB2HSV",
        CV::COLOR_BGR2YUV, "COLOR_BGR2YUV",
        CV::COLOR_RGB2YUV, "COLOR_RGB2YUV",
        CV::COLOR_YUV2RGB_NV12, "COLOR_YUV2RGB_NV12",
        CV::COLOR_YUV2BGR_NV12, "COLOR_YUV2BGR_NV12",
        CV::COLOR_YUV2RGB_NV21, "COLOR_YUV2RGB_NV21",
        CV::COLOR_YUV2BGR_NV21, "COLOR_YUV2BGR_NV21",
        CV::COLOR_YUV2RGBA_NV12, "COLOR_YUV2RGBA_NV12",
        CV::COLOR_YUV2BGRA_NV12, "COLOR_YUV2BGRA_NV12",
        CV::COLOR_YUV2RGBA_NV21, "COLOR_YUV2RGBA_NV21",
        CV::COLOR_YUV2BGRA_NV21, "COLOR_YUV2BGRA_NV21",
        CV::COLOR_YUV2RGB_I420, "COLOR_YUV2RGB_I420",
        CV::COLOR_YUV2BGR_I420, "COLOR_YUV2BGR_I420",
        CV::COLOR_YUV2RGBA_I420, "COLOR_YUV2RGBA_I420",
        CV::COLOR_YUV2BGRA_I420, "COLOR_YUV2BGRA_I420"
        )
def_enum(InterpolationFlags, CV::InterpolationFlags,
        CV::INTER_NEAREST, "INTER_NEAREST",
        CV::INTER_LINEAR, "INTER_LINEAR",
        CV::INTER_CUBIC, "INTER_CUBIC",
        CV::INTER_AREA, "INTER_AREA",
        CV::INTER_LANCZOS4, "INTER_LANCZOS4",
        CV::INTER_LINEAR_EXACT, "INTER_LINEAR_EXACT",
        CV::INTER_NEAREST_EXACT, "INTER_NEAREST_EXACT",
        CV::INTER_MAX, "INTER_MAX",
        CV::WARP_FILL_OUTLIERS, "WARP_FILL_OUTLIERS",
        CV::WARP_INVERSE_MAP, "WARP_INVERSE_MAP"
        )
def_enum(BorderTypes, CV::BorderTypes,
        CV::BORDER_CONSTANT, "BORDER_CONSTANT",
        CV::BORDER_REPLICATE, "BORDER_REPLICATE",
        CV::BORDER_REFLECT, "BORDER_REFLECT",
        CV::BORDER_WRAP, "BORDER_WRAP",
        CV::BORDER_REFLECT_101, "BORDER_REFLECT_101",
        CV::BORDER_TRANSPARENT, "BORDER_TRANSPARENT",
        CV::BORDER_REFLECT101, "BORDER_REFLECT101",
        CV::BORDER_DEFAULT, "BORDER_DEFAULT",
        CV::BORDER_ISOLATED, "BORDER_ISOLATED"
        )
def_enum(ThresholdTypes, CV::ThresholdTypes,
        CV::THRESH_BINARY, "THRESH_BINARY",
        CV::THRESH_BINARY_INV, "THRESH_BINARY_INV",
        CV::THRESH_TRUNC, "THRESH_TRUNC",
        CV::THRESH_TOZERO, "THRESH_TOZERO",
        CV::THRESH_TOZERO_INV, "THRESH_TOZERO_INV",
        CV::THRESH_MASK, "THRESH_MASK",
        CV::THRESH_OTSU, "THRESH_OTSU",
        CV::THRESH_TRIANGLE, "THRESH_TRIANGLE"
        )
def_enum(RetrievalModes, CV::RetrievalModes,
        CV::RETR_EXTERNAL, "RETR_EXTERNAL",
        CV::RETR_LIST, "RETR_LIST",
        CV::RETR_CCOMP, "RETR_CCOMP",
        CV::RETR_TREE, "RETR_TREE",
        CV::RETR_FLOODFILL, "RETR_FLOODFILL"
        )
def_enum(ContourApproximationModes, CV::ContourApproximationModes,
        CV::CHAIN_APPROX_NONE, "CHAIN_APPROX_NONE",
        CV::CHAIN_APPROX_SIMPLE, "CHAIN_APPROX_SIMPLE",
        CV::CHAIN_APPROX_TC89_L1, "CHAIN_APPROX_TC89_L1",
        CV::CHAIN_APPROX_TC89_KCOS, "CHAIN_APPROX_TC89_KCOS"
        )
def_enum(LineTypes, CV::LineTypes,
        CV::FILLED, "FILLED",
        CV::LINE_4, "LINE_4",
        CV::LINE_8, "LINE_8",
        CV::LINE_AA, "LINE_AA"
        )
// helper functions
INTS default_size = {0, 0}, default_param = {};
bool isSize(PyObject* obj) {
    return (isInts(obj) && toInts(obj).size() == 2);
}
CV::Size toSize(PyObject* obj) {
    auto vals = toInts(obj);
    MNN_ASSERT(val.size() == 2);
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
#ifdef PYMNN_IMGCODECS
def_enum(ImreadModes, CV::ImreadModes,
        CV::IMREAD_GRAYSCALE, "IMREAD_GRAYSCALE",
        CV::IMREAD_COLOR, "IMREAD_COLOR",
        CV::IMREAD_ANYDEPTH, "IMREAD_ANYDEPTH"
        )
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
    PyObject *buf, *flags = toPyObj(CV::IMREAD_COLOR);
    if (PyArg_ParseTuple(args, "OO", &buf, &flags) && isImreadModes(flags)) {
        int64_t length = -1;
        auto data = static_cast<uint8_t*>(toPtr(buf, DType_UINT8, length));
        std::vector<uint8_t> buf(data, data + length);
        return toPyObj(CV::imdecode(buf, toEnum<CV::ImreadModes>(flags)));
    }
    PyMNN_ERROR("imdecode require args: (ptr, int)");
}
static PyObject* PyMNNCV_imencode(PyObject *self, PyObject *args) {
    const char *ext = NULL;
    PyObject *img, *params = toPyObj(default_param);
    if (PyArg_ParseTuple(args, "sO|O", &ext, &img, &params) && isVar(img) && isInts(params)) {
        return toPyObj<bool, toPyObj, std::vector<uint8_t>, toPyObj>(CV::imencode(ext, toVar(img), toInts(params)));
    }
    PyMNN_ERROR("imencode require args: (string, Var, |[int])");
}
static PyObject* PyMNNCV_imread(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    PyObject* flags = toPyObj(CV::IMREAD_COLOR);
    if (PyArg_ParseTuple(args, "s|O", &filename, &flags) && filename && isImreadModes(flags)) {
        return toPyObj(CV::imread(filename, toEnum<CV::ImreadModes>(flags)));
    }
    PyMNN_ERROR("imread require args: (string, ImreadModes)");
}
static PyObject* PyMNNCV_imwrite(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    PyObject *img, *params = toPyObj(default_param);
    if (PyArg_ParseTuple(args, "sO|O", &filename, &img, &params) &&
        filename && isVar(img) && isInts(params)) {
        return toPyObj(CV::imwrite(filename, toVar(img), toInts(params)));
    }
    PyMNN_ERROR("imwrite require args: (string, Var, |[int])");
}
#endif
// color
static PyObject* PyMNNCV_cvtColor(PyObject *self, PyObject *args) {
    PyObject *src, *code;
    int dstCn = 0;
    if (PyArg_ParseTuple(args, "OO|i", &src, &code, &dstCn) &&
        isVar(src) && isColorConversionCodes(code)) {
        return toPyObj(CV::cvtColor(toVar(src), toEnum<CV::ColorConversionCodes>(code), dstCn));
    }
    PyMNN_ERROR("cvtColor require args: (Var, ColorConversionCodes, |int)");
}
static PyObject* PyMNNCV_cvtColorTwoPlane(PyObject *self, PyObject *args) {
    PyObject *src1, *src2, *code;
    if (PyArg_ParseTuple(args, "OOO", &src1, &src2, &code) &&
        isVar(src1) && isVar(src2) && isColorConversionCodes(code)) {
        return toPyObj(CV::cvtColorTwoPlane(toVar(src1), toVar(src2), toEnum<CV::ColorConversionCodes>(code)));
    }
    PyMNN_ERROR("cvtColorTwoPlane require args: (Var, Var, ColorConversionCodes)");
}
// filter
static PyObject* PyMNNCV_blur(PyObject *self, PyObject *args) {
    PyObject *src, *ksize, *borderType = toPyObj(REFLECT);
    if (PyArg_ParseTuple(args, "OO|O", &src, &ksize, &borderType) &&
        isVar(src) && isSize(ksize) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::blur(toVar(src), toSize(ksize), toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("blur require args: (Var, [int], |PadValue_Mode)");
}
static PyObject* PyMNNCV_boxFilter(PyObject *self, PyObject *args) {
    PyObject *src, *ksize, *borderType = toPyObj(REFLECT);
    int ddepth;
    int normalize = 1;
    if (PyArg_ParseTuple(args, "OiO|iO", &src, &ddepth, &ksize, &normalize, &borderType) &&
        isVar(src) && isSize(ksize) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::boxFilter(toVar(src), ddepth, toSize(ksize), normalize, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("boxFilter require args: (Var, int, [int], |bool, PadValue_Mode)");
}
static PyObject* PyMNNCV_dilate(PyObject *self, PyObject *args) {
    PyObject *src, *kernel, *borderType = toPyObj(REFLECT);
    int iterations = 1;
    if (PyArg_ParseTuple(args, "OO|iO", &src, &kernel, &iterations, &borderType) &&
        isVar(src) && isVar(kernel) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::dilate(toVar(src), toVar(kernel), iterations, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("dilate require args: (Var, Var, |int, PadValue_Mode)");
}
static PyObject* PyMNNCV_filter2D(PyObject *self, PyObject *args) {
    PyObject *src, *kernel, *borderType = toPyObj(REFLECT);
    int ddepth;
    float delta = 0;
    if (PyArg_ParseTuple(args, "OiO|fO", &src, &ddepth, &kernel, &delta, &borderType) &&
        isVar(src) && isVar(kernel) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::filter2D(toVar(src), ddepth, toVar(kernel), delta, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("filter2D require args: (Var, int, Var, |float, PadValue_Mode)");
}
static PyObject* PyMNNCV_GaussianBlur(PyObject *self, PyObject *args) {
    PyObject *src, *ksize, *borderType = toPyObj(REFLECT);
    float sigmaX, sigmaY = 0;
    if (PyArg_ParseTuple(args, "OOf|fO", &src, &ksize, &sigmaX, &sigmaY, &borderType) &&
        isVar(src) && isSize(ksize) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::GaussianBlur(toVar(src), toSize(ksize), sigmaX, sigmaY, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("GaussianBlur require args: (Var, [int], float, |float, PadValue_Mode)");
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
    PyObject *src, *borderType = toPyObj(REFLECT);
    int ddepth, ksize = 1;
    float scale = 1, delta = 0;
    if (PyArg_ParseTuple(args, "Oi|iffO", &src, &ddepth, &ksize, &scale, &delta, &borderType)
        && isVar(src) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::Laplacian(toVar(src), ddepth, ksize, scale, delta, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("Laplacian require args: (Var, int, |int, float, float, PadValue_Mode)");
}
static PyObject* PyMNNCV_pyrDown(PyObject *self, PyObject *args) {
    PyObject *src, *dstsize = toPyObj(default_size), *borderType = toPyObj(REFLECT);
    if (PyArg_ParseTuple(args, "O|OO", &src, &dstsize, &borderType) &&
        isVar(src) && isSize(dstsize) && isPadValue_Mode(borderType))  {
        return toPyObj(CV::pyrDown(toVar(src), toSize(dstsize), toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("pyrDown require args: (Var, |[int], PadValue_Mode)");
}
static PyObject* PyMNNCV_pyrUp(PyObject *self, PyObject *args) {
    PyObject *src, *dstsize = toPyObj(default_size), *borderType = toPyObj(REFLECT);
    if (PyArg_ParseTuple(args, "O|OO", &src, &dstsize, &borderType) &&
        isVar(src) && isSize(dstsize) && isPadValue_Mode(borderType))  {
        return toPyObj(CV::pyrUp(toVar(src), toSize(dstsize), toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("pyrUp require args: (Var, |[int], PadValue_Mode)");
}
static PyObject* PyMNNCV_Scharr(PyObject *self, PyObject *args) {
    PyObject *src, *borderType = toPyObj(REFLECT);
    int ddepth, dx, dy;
    float scale = 1, delta = 0;
    if (PyArg_ParseTuple(args, "Oiii|ffO", &src, &ddepth, &dx, &dy, &scale, &delta, &borderType)
        && isVar(src) && isPadValue_Mode(borderType))  {
        return toPyObj(CV::Scharr(toVar(src), ddepth, dx, dy, scale, delta, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("Scharr require args: (Var, int, int, int, |float, float, PadValue_Mode)");
}
static PyObject* PyMNNCV_sepFilter2D(PyObject *self, PyObject *args) {
    PyObject *src, *kernelX, *kernelY, *borderType = toPyObj(REFLECT);
    int ddepth;
    float delta = 0;
    if (PyArg_ParseTuple(args, "OiOO|fO", &src, &ddepth, &kernelX, &kernelY, &delta, &borderType)
        && isVar(src) && isVar(kernelX) && isVar(kernelY) && isPadValue_Mode(borderType))  {
        auto kx = toVar(kernelX), ky = toVar(kernelY);
        return toPyObj(CV::sepFilter2D(toVar(src), ddepth, kx, ky, delta,
                       toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("sepFilter2D require args: (Var, int, Var, Var, |float, PadValue_Mode)");
}
static PyObject* PyMNNCV_Sobel(PyObject *self, PyObject *args) {
    PyObject *src, *borderType = toPyObj(REFLECT);
    int ddepth, dx, dy, ksize = 3;
    float scale = 1, delta = 0;
    if (PyArg_ParseTuple(args, "Oiii|iffO", &src, &ddepth, &dx, &dy, &ksize, &scale, &delta, &borderType)
        && isVar(src) && isPadValue_Mode(borderType))  {
        return toPyObj(CV::Sobel(toVar(src), ddepth, dx, dy, ksize, scale, delta, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("Sobel require args: (Var, int, int, int, |int, float, float, PadValue_Mode)");
}
static PyObject* PyMNNCV_spatialGradient(PyObject *self, PyObject *args) {
    PyObject *src, *borderType = toPyObj(REFLECT);
    int ksize = 3;
    if (PyArg_ParseTuple(args, "O|iO", &src, &ksize, &borderType)) {
        return toPyObj<VARP, toPyObj, VARP, toPyObj>(CV::spatialGradient(toVar(src), ksize, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("spatialGradient require args: (Var, |int, PadValue_Mode)");
}
static PyObject* PyMNNCV_sqrBoxFilter(PyObject *self, PyObject *args) {
    PyObject *src, *ksize, *borderType = toPyObj(REFLECT);
    int ddepth;
    int normalize = 1;
    if (PyArg_ParseTuple(args, "OiO|iO", &src, &ddepth, &ksize, &normalize, &borderType) &&
        isVar(src) && isSize(ksize) && isPadValue_Mode(borderType)) {
        return toPyObj(CV::sqrBoxFilter(toVar(src), ddepth, toSize(ksize), normalize, toEnum<MNN::Express::PadValueMode>(borderType)));
    }
    PyMNN_ERROR("sqrBoxFilter require args: (Var, int, [int], |bool, PadValue_Mode)");
}
// geometric
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
std::vector<float> default_floats = {};
static PyObject* PyMNNCV_resize(PyObject *self, PyObject *args) {
    PyObject *src, *dsize, *interpolation = toPyObj(CV::INTER_LINEAR),
             *mean = toPyObj(default_floats), *norm = toPyObj(default_floats);
    float fx = 0, fy = 0;
    int code = -1;
    if (PyArg_ParseTuple(args, "OO|ffOiOO", &src, &dsize, &fx, &fy, &interpolation, &code, &mean, &norm) &&
        isVar(src) && isSize(dsize) && isInterpolationFlags(interpolation) && isFloats(mean) && isFloats(norm)) {
        return toPyObj(CV::resize(toVar(src), toSize(dsize), fx, fy, toEnum<CV::InterpolationFlags>(interpolation), code, toFloats(mean), toFloats(norm)));
    }
    PyMNN_ERROR("resize require args: (Var, [int], |float, float, InterpolationFlags, int, [float], [float])");
}
static PyObject* PyMNNCV_warpAffine(PyObject *self, PyObject *args) {
    PyObject *src, *M, *dsize, *flag = toPyObj(CV::INTER_LINEAR), *borderMode = toPyObj(CV::BORDER_CONSTANT),
             *mean = toPyObj(default_floats), *norm = toPyObj(default_floats);
    int borderValue = 0, code = -1;
    if (PyArg_ParseTuple(args, "OOO|OOiiOO", &src, &M, &dsize, &flag, &borderMode, &borderValue, &code, &mean, &norm) &&
        isVar(src) && isMatrix(M) && isSize(dsize) && isInterpolationFlags(flag) && isBorderTypes(borderMode) && isFloats(mean) && isFloats(norm)) {
        return toPyObj(CV::warpAffine(toVar(src), toMatrix(M), toSize(dsize),
                       toEnum<CV::InterpolationFlags>(flag), toEnum<CV::BorderTypes>(borderMode), borderValue, code, toFloats(mean), toFloats(norm)));
    }
    PyMNN_ERROR("warpAffine require args: (Var, Matrix, [int], |InterpolationFlags, BorderTypes, int, int, [float], [float])");
}
static PyObject* PyMNNCV_warpPerspective(PyObject *self, PyObject *args) {
    PyObject *src, *M, *dsize, *flag = toPyObj(CV::INTER_LINEAR), *borderMode = toPyObj(CV::BORDER_CONSTANT);
    int borderValue = 0;
    if (PyArg_ParseTuple(args, "OOO|OOi", &src, &M, &dsize, &flag, &borderMode, &borderValue) &&
        isVar(src) && isMatrix(M) && isSize(dsize) && isInterpolationFlags(flag) && isBorderTypes(borderMode)) {
        return toPyObj(CV::warpPerspective(toVar(src), toMatrix(M), toSize(dsize),
                       toEnum<CV::InterpolationFlags>(flag), toEnum<CV::BorderTypes>(borderMode), borderValue));
    }
    PyMNN_ERROR("warpPerspective require args: (Var, Matrix, [int], |InterpolationFlags, BorderTypes, int)");
}
// miscellaneous
static PyObject* PyMNNCV_blendLinear(PyObject *self, PyObject *args) {
    PyObject *src1, *src2, *weight1, *weight2;
    if (PyArg_ParseTuple(args, "OOOO", &src1, &src2, &weight1, &weight2) &&
        isVar(src1) && isVar(src2) && isVar(weight1) && isVar(weight2)) {
        return toPyObj(CV::blendLinear(toVar(src1), toVar(src2), toVar(weight1), toVar(weight2)));
    }
    PyMNN_ERROR("blendLinear require args: (Var, Var, Var, Var)");
}
static PyObject* PyMNNCV_threshold(PyObject *self, PyObject *args) {
    PyObject *src, *type;
    float thresh, maxval;
    if (PyArg_ParseTuple(args, "OffO", &src, &thresh, &maxval, &type) &&
        isVar(src) && isThresholdTypes(type)) {
        return toPyObj(CV::threshold(toVar(src), thresh, maxval, toEnum<CV::ThresholdTypes>(type)));
    }
    PyMNN_ERROR("threshold require args: (Var, float, float, ThresholdTypes)");
}
// structural
static PyObject* PyMNNCV_findContours(PyObject *self, PyObject *args) {
    PyObject *image, *mode, *method, *offset = toPyObj(std::vector<float>{0, 0});
    if (PyArg_ParseTuple(args, "OOO|O", &image, &mode, &method, &offset) &&
        isVar(image) && isRetrievalModes(mode) && isContourApproximationModes(method) && isPoint(offset)) {
        auto contours = CV::findContours(toVar(image), toEnum<CV::RetrievalModes>(mode),
                                         toEnum<CV::ContourApproximationModes>(method), toPoint(offset));
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
// draw
static bool isColor(PyObject* obj) {
    return (isInts(obj) && (toInts(obj).size() == 3 || toInts(obj).size() == 4));
}
CV::Scalar toColor(PyObject* obj) {
    auto vals = toInts(obj);
    if (vals.size() == 3) {
        return CV::Scalar(vals[0], vals[1], vals[2]);
    }
    if (vals.size() == 4) {
        return CV::Scalar(vals[0], vals[1], vals[2], vals[3]);
    }
    return CV::Scalar(255, 255, 255);
}
static PyObject* PyMNNCV_line(PyObject *self, PyObject *args) {
    PyObject *img, *pt1, *pt2, *color, *linetype = toPyObj(CV::LINE_8);
    int thickness = 1, shift = 0;
    if (PyArg_ParseTuple(args, "OOOO|iOi", &img, &pt1, &pt2, &color, &thickness, &linetype, &shift)
        && isVar(img) && isPoint(pt1) && isPoint(pt2) && isColor(color) && isLineTypes(linetype)) {
        auto image = toVar(img);
        CV::line(image, toPoint(pt1), toPoint(pt2), toColor(color),
                 thickness, toEnum<CV::LineTypes>(linetype), shift);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("line require args: (Var, Point, Point, Color, |int, LineType, int)");
}
static PyObject* PyMNNCV_arrowedLine(PyObject *self, PyObject *args) {
    PyObject *img, *pt1, *pt2, *color, *linetype = toPyObj(CV::LINE_8);
    int thickness = 1, shift = 0;
    float tipLength = 0.1;
    if (PyArg_ParseTuple(args, "OOOO|iOif", &img, &pt1, &pt2, &color, &thickness, &linetype, &shift, &tipLength)
        && isVar(img) && isPoint(pt1) && isPoint(pt2) && isColor(color) && isLineTypes(linetype)) {
        auto image = toVar(img);
        CV::arrowedLine(image, toPoint(pt1), toPoint(pt2), toColor(color),
                        thickness, toEnum<CV::LineTypes>(linetype), shift, tipLength);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("arrowedLine require args: (Var, Point, Point, Color, |int, LineType, int, float)");
}
static PyObject* PyMNNCV_circle(PyObject *self, PyObject *args) {
    PyObject *img, *center, *color, *linetype = toPyObj(CV::LINE_8);
    int radius, thickness = 1, shift = 0;
    if (PyArg_ParseTuple(args, "OOiO|iOi", &img, &center, &radius, &color, &thickness, &linetype, &shift)
        && isVar(img) && isPoint(center) && isColor(color) && isLineTypes(linetype)) {
        auto image = toVar(img);
        CV::circle(image, toPoint(center), radius, toColor(color),
                 thickness, toEnum<CV::LineTypes>(linetype), shift);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("circle require args: (Var, Point, int, Color, |int, LineType, int)");
}
static PyObject* PyMNNCV_rectangle(PyObject *self, PyObject *args) {
    PyObject *img, *pt1, *pt2, *color, *linetype = toPyObj(CV::LINE_8);
    int thickness = 1, shift = 0;
    if (PyArg_ParseTuple(args, "OOOO|iOi", &img, &pt1, &pt2, &color, &thickness, &linetype, &shift)
        && isVar(img) && isPoint(pt1) && isPoint(pt2) && isColor(color) && isLineTypes(linetype)) {
        auto image = toVar(img);
        CV::rectangle(image, toPoint(pt1), toPoint(pt2), toColor(color),
                      thickness, toEnum<CV::LineTypes>(linetype), shift);
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("rectangle require args: (Var, Point, Point, Color, |int, LineType, int)");
}
static PyObject* PyMNNCV_drawContours(PyObject *self, PyObject *args) {
    PyObject *img, *contours, *color, *linetype = toPyObj(CV::LINE_8);
    int contourIdx, thickness = 1;
    if (PyArg_ParseTuple(args, "OOiO|iO", &img, &contours, &contourIdx, &color, &thickness, &linetype)
        && isVar(img) && isVec<isPoints>(contours) && isColor(color) && isLineTypes(linetype)) {
        auto image = toVar(img);
        CV::drawContours(image, toVec<std::vector<CV::Point>, toPoints>(contours), contourIdx, toColor(color),
                         thickness, toEnum<CV::LineTypes>(linetype));
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("drawContours require args: (Var, [Points], int, Color, |int, LineType)");
}
static PyObject* PyMNNCV_fillPoly(PyObject *self, PyObject *args) {
    PyObject *img, *contours, *color, *linetype = toPyObj(CV::LINE_8), *offset = toPyObj(std::vector<float>{0, 0});
    int shift = 0;
    if (PyArg_ParseTuple(args, "OOO|OiO", &img, &contours, &color, &linetype, &shift, &offset)
        && isVar(img) && isVec<isPoints>(contours) && isColor(color) && isLineTypes(linetype) && isPoint(offset)) {
        auto image = toVar(img);
        CV::fillPoly(image, toVec<std::vector<CV::Point>, toPoints>(contours), toColor(color),
                     toEnum<CV::LineTypes>(linetype), shift, toPoint(offset));
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("fillPoly require args: (Var, [Points], Color, |LineType, int, Point)");
}
static PyMethodDef PyMNNCV_methods[] = {
#ifdef PYMNN_IMGCODECS
    register_methods(CV,
        // imgcodecs
        haveImageReader, "haveImageReader",
        haveImageWriter, "haveImageWriter",
        imdecode, "imdecode",
        imencode, "imencode",
        imread, "imread",
        imwrite, "imwrite"
    )
#endif
    register_methods(CV,
        // color
        cvtColor, "cvtColor.",
        cvtColorTwoPlane, "cvtColorTwoPlane.",
        // filter
        blur, "blur",
        boxFilter, "boxFilter",
        dilate, "dilate",
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
        sqrBoxFilter, "sqrBoxFilter",
        // geometric
        getAffineTransform, "getAffineTransform",
        getPerspectiveTransform, "getPerspectiveTransform",
        getRectSubPix, "getRectSubPix",
        getRotationMatrix2D, "getRotationMatrix2D",
        invertAffineTransform, "invertAffineTransform",
        resize, "resize",
        warpAffine, "warpAffine",
        warpPerspective, "warpPerspective",
        // miscellaneous
        blendLinear, "blendLinear",
        threshold, "threshold",
        // structural
        findContours, "findContours",
        contourArea, "contourArea",
        convexHull, "convexHull",
        minAreaRect, "minAreaRect",
        boundingRect, "boundingRect",
        connectedComponentsWithStats, "connectedComponentsWithStats",
        boxPoints, "boxPoints",
        // draw
        line, "line",
        arrowedLine, "arrowedLine",
        circle, "circle",
        rectangle, "rectangle",
        drawContours, "drawContours",
        fillPoly, "fillPoly"
    )
};
