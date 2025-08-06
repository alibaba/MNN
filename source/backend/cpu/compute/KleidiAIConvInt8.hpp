//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KleidiAIConvInt8_hpp
#define KleidiAIConvInt8_hpp
#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/arm/mnn_kleidiai.h"

namespace MNN {
class KleidiAIConvInt8 : public CPUConvolution {
public:
    KleidiAIConvInt8(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant, KleidiAI &kai, KleidiAI::AccelType accelType, int32_t blockNum);
    virtual ~KleidiAIConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe);
    std::shared_ptr<Tensor> mWeightInt8;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    std::shared_ptr<Tensor> mInputConvertBuffer;
    std::shared_ptr<Tensor> mOutputConvertBuffer;
    KleidiAI &kai;
    KleidiAI::AccelType mAccelType = KleidiAI::AccelType::ACC_TYPE_NUMBER;
    int32_t mBlockNum = 1;
};

} // namespace MNN
#endif /* KleidiAIConvInt8_hpp */
