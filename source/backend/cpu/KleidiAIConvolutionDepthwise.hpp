#ifndef KleidiAIConvolutionDepthwise_hpp
#define KleidiAIConvolutionDepthwise_hpp

#ifdef MNN_KLEIDIAI_ENABLED

#include "core/AutoStorage.h"
#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionIntFactory.hpp"
#include "kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"
#include "kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "backend/cpu/CPUTensorConvert.hpp"

namespace MNN {
class KleidiAIConvolutionDepthwise {
    public:
    class KleidiAIDepthwiseExecution : public CPUConvolution {
        public:
            KleidiAIDepthwiseExecution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                            size_t originWeightSize, const float *bias, size_t biasSize);
            virtual ~KleidiAIDepthwiseExecution() = default;
            virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
            virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

        private:
            int mNumber = 1;
            std::shared_ptr<Tensor> mPackedRhs;
            std::shared_ptr<Tensor> mWeightTemp;
            Tensor mOutputNHWC;
            Tensor mInputNHWC;
    };
};

} // namespace MNN

#endif // defined(MNN_KLEIDIAI_ENABLED)

#endif /* KleidiAIConvolutionDepthwise_hpp */
