#ifndef HexagonQuantization_hpp
#define HexagonQuantization_hpp

#include "../htp-ops-lib/include/dsp/int8_cnn.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace MNN {

class Op;
class Tensor;

DSPRequantParam makeRequantMultiplier(double realScale, int32_t outputZero = 0,
                                      int32_t clampMin = -127,
                                      int32_t clampMax = 127);
int32_t requantizeReference(int32_t value, const DSPRequantParam& param);

size_t w8a8Conv1x1PackedWeightBytes(int inputChannels, int outputChannels);
bool packW8A8Conv1x1Weight(std::vector<int8_t>& packed, const int8_t* weight,
                           size_t weightCount, int inputChannels,
                           int outputChannels);
bool buildW8A8Conv1x1PostParams(std::vector<int32_t>& biasInt32,
                               std::vector<DSPRequantParam>& requant,
                               const float* bias, int biasCount,
                               const float* weightScale, int weightScaleCount,
                               int outputChannels, float inputScale,
                               float outputScale, bool relu, bool relu6,
                               int32_t outputMin = -127,
                               int32_t outputMax = 127);
bool supportsW8A8Conv1x1(const Op* op, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs);

bool packFloatToInt8Nc(const float* src, int8_t* dst, int batch, int channels,
                       int height, int width, int devicePack, int hostPack,
                       float scale, int32_t zero, int32_t clampMin,
                       int32_t clampMax, bool hostNhwc = false);
bool unpackInt8NcToFloat(const int8_t* src, float* dst, int batch, int channels,
                         int height, int width, int devicePack, int hostPack,
                         float scale, int32_t zero, bool hostNhwc = false);

} // namespace MNN

#endif
