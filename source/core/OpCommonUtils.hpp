#ifndef OpCommonUtils_hpp
#define OpCommonUtils_hpp
#include <MNN/Tensor.hpp>
namespace MNN {
class MNN_PUBLIC OpCommonUtils {
public:
    static void broastCastComputeDim(int* dims, int* stride, int* iStride0, int* iStride1, const Tensor* input0, const Tensor* input1, const Tensor* output);
};
}

#endif
