#ifndef SafetensorUtils_hpp
#define SafetensorUtils_hpp

#include <utility>

#include <MNN/MNNDefine.h>
#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
namespace Express {
namespace SafeTensorUtils {

struct LayerNormInfo {
    VARP inputLayerNormWeight;
    VARP inputLayerNormBias;
    float ln_eps = 0.0f;
    bool useRMSNorm = false;
    int hiddenSize = 0;
    bool useC4 = false;

    LayerNormInfo() = default;
    LayerNormInfo(VARP weight, VARP bias, float eps, bool rms, int hidden = 0, bool c4 = false)
        : inputLayerNormWeight(weight), inputLayerNormBias(bias), ln_eps(eps), useRMSNorm(rms), hiddenSize(hidden), useC4(c4) {
    }
};

struct RopeInfo {
    LayerNormInfo qNorm;
    LayerNormInfo kNorm;
    int cutHeadDim = 0;
};

MNN_PUBLIC VARP _QConvolution1x1(int inputCount, VARP input, VARP inputScale, VARP inputZero, VARP weight,
                                 VARP wscale, VARP wzeropoint, VARP bias, int outputcount = 0,
                                 bool scaleInputCount = false, int weightBits = 0);
MNN_PUBLIC VARP _TransformerLayerNorm(VARP hiddenState, const LayerNormInfo& info);
MNN_PUBLIC std::pair<VARP, VARP> _BinaryLayerNorm(VARP r0, VARP r1, const LayerNormInfo& info);
MNN_PUBLIC VARP _GPT2Attention(int numHead, int headDim, VARP q, VARP k, VARP v, VARP qk_scale_q, VARP qk_scale_k,
                               VARP sv_scale_s, VARP sv_scale_v, VARP mask, bool supportC4Opt = false,
                               float attnScale = 0.0f);
MNN_PUBLIC VARP _PrecomputePosEmbedding(int dim, int end, float theta = 1000000.0f, bool interleaved = false);
MNN_PUBLIC VARPS _TransformerRoPE(VARP q, VARP k, VARP cosEven, VARP cosOdd, VARP sinEven, VARP sinOdd,
                                  const RopeInfo& info);
MNN_PUBLIC VARP _MakeLastHiddenStateOutput(VARP hiddenState, int hiddenSize);

} // namespace SafeTensorUtils
} // namespace Express
} // namespace MNN

#endif
