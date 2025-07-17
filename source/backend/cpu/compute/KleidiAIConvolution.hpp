//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KleidiAIConvolution_hpp
#define KleidiAIConvolution_hpp
#ifdef MNN_KLEIDIAI_ENABLED
#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
namespace MNN {
#ifndef MNN_REDUCE_SIZE

class KleidiAIConvolution : public CPUConvolution{
    public:
        KleidiAIConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize);
        KleidiAIConvolution(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b);
        virtual ~KleidiAIConvolution();
    
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    private:
        std::shared_ptr<Tensor> mInputResource;
        std::shared_ptr<Tensor> mInputConvertBuffer;
        std::shared_ptr<Tensor> mOutputConvertBuffer;
        std::shared_ptr<CPUConvolution::Resource> mResource;
        KleidiAI::AccelType mAccelType = KleidiAI::AccelType::ACC_TYPE_NUMBER;

};
#endif //MNN_KLEIDIAI_ENABLED

} // namespace MNN
#endif
#endif /* KleidiAIConvolution_hpp */
