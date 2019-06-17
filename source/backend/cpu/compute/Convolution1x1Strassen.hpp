//
//  Convolution1x1Strassen.hpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Convolution1x1Strassen_hpp
#define Convolution1x1Strassen_hpp

#include <functional>
#include "../CPUConvolution.hpp"
#include "StrassenMatmulComputor.hpp"
namespace MNN {
class Convolution1x1Strassen : public CPUConvolution {
public:
    Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                           size_t originWeightSize, const float *bias, size_t biasSize);
    virtual ~Convolution1x1Strassen();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onReleaseCache() override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;

    struct Unit {
        bool mValid = true;
        std::shared_ptr<Tensor> mTempInput;
        std::shared_ptr<Tensor> mTempWeight;
        std::shared_ptr<Tensor> mTempOutput;
        std::vector<Tensor *> mTempInputVector;
        std::vector<Tensor *> mTempOutputVector;
        std::shared_ptr<StrassenMatrixComputor> mStracssenComputor;

        std::function<void()> mPostExecutor;
    };

    std::vector<Unit> mUnits;
    CPUConvolution::POSTFUNCTION mPostFunction;
    std::shared_ptr<Tensor> mTempInputBatch;
    std::shared_ptr<Tensor> mTempOutputBatch;
    bool mNeedPretreat = false;
    std::function<void(const float *srcBatch, float *dstBatch)> mPretreatFunction;
};
} // namespace MNN

#endif /* Convolution1x1Strassen_hpp */
