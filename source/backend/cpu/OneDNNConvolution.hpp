#include "core/Execution.hpp"
#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
namespace OneDNN {
Execution* createConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize);
};
};
