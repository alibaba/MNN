//
//  CPUConvolution3D.hpp
//  MNN
//
//  Created by MNN on 2019/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvolution3D_hpp
#define CPUConvolution3D_hpp

#include <vector>
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
    class CPUConvolution3D : public Execution {
    public:
        CPUConvolution3D(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *b);
        virtual ~CPUConvolution3D();
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        
        typedef void (*POSTFUNCTION)(float *dst, const float *bias, size_t planeNumber, size_t biasNumber);
        static POSTFUNCTION getPostFunction(const Convolution3DCommon* common);
        
    private:
        void convertToDepthMajor(float* dst, const float* src, uint32_t planeNumber, uint32_t depth, uint32_t outsideNumber);
        void convertDNC4HW4toNC4DHW4(float* dst, const float* src, uint32_t planeNumber, uint32_t depth, uint32_t outsideNumber, bool add);

        const Convolution3DCommon* mCommon;
        std::vector<int> mDilates;
        std::vector<int> mStrides;
        std::vector<int> mKernels;
        std::vector<int> mPads;
        int mInputCount;
        int mOutputCount;
        PadMode mPadMode;
        POSTFUNCTION mPostFunction;
        std::shared_ptr<Tensor> mBias;
        std::shared_ptr<Tensor> mWeights;
        std::shared_ptr<Tensor> mInputStorage;
        std::shared_ptr<Tensor> mSubOutputTensor;
        std::vector<std::shared_ptr<Tensor>> mSubInputTensors;
        std::vector<std::shared_ptr<Execution>> mSubExecution;
        bool mBreakDown;
        bool mCrossDepth;
    };
    
} // namespace MNN

#endif /* CPUConvolution3D_hpp */
