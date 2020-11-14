//
//  ConvInt83x3.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvInt83x3_hpp
#define ConvInt83x3_hpp

#include "backend/cpu/CPUConvolution.hpp"
#define Int5Enough

namespace MNN {
class ConvInt83x3 : public CPUConvolution {
public:
    ConvInt83x3(Backend *backend, const MNN::Convolution2D *convOp, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~ConvInt83x3();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    struct ComputeStrategy {
        // winograd unit type, only 2D have many meaningless compute.
        enum UnitType { D2, D2_D1 } unitType;
        /*
         Online: do 2D and 1D transform on runtime
         ExtraOnline: do 2D transform before run, 1D transform on runtime
         Offline: do 2D and 1D transform before run
         The option influence runtime memory usage.
         */
        enum TransPhase { Online, ExtraOnline, Offline } transPhase;
                
        bool operator == (ComputeStrategy& another) {
            return unitType == another.unitType && transPhase == another.transPhase;
        }
    };
    ComputeStrategy getComputeStrategy(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) const;
    ErrorCode tensorMemoryOnStrategyChange(ComputeStrategy* oldStrategy, ComputeStrategy* newStrategy,
                                           const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                           std::vector<Tensor*> *dynamicAllocTensors);
    void weightContent(bool trans2d, bool trans1d);
    
    ComputeStrategy mStrategy;
    // strategy is completed if all entry is defined.
    bool mStrategyCompleted = false;
    // {mulType, D2, Offline} strategy will always be use unless program ending, which reduce memory usage on runtime.
    // the attribute is False when MemoryMode is High.
    bool mFixedSimpleStrategy = true;
    
    // relu or relu6
    bool mRelu;
        
    int mActBits;

    // untransformed reordered weight (ocUnit, icUnit, 3*3, 4*unitI)
    std::shared_ptr<Tensor> mWeightInt8;
    // winograd 2d-transformed weight (BLOCK_UNIT2, ocUnit, icUnit, 4*unitI)
    std::shared_ptr<Tensor> mWeight;
    // winograd 1d-transformed weight (2 [horizontal and vertical], kernel * BLOCK_UNIT, ocUnit, icUnit, 4*unitI)
    std::shared_ptr<Tensor> mWeightExtra;
    // leftover elements (read needed) on bottom-right using 2d1d mix unit partition
    // leftover num (for 3x3 kernel): 2*2 when feature map size is odd, otherwise 0
    std::shared_ptr<Tensor> mWeightLeftOver;
    
    std::shared_ptr<Tensor> mBiasFloat;
    std::shared_ptr<Tensor> mScaleFloat;

    std::shared_ptr<Tensor> mTempInput;
    std::shared_ptr<Tensor> mTempSrcBuffer;
    std::shared_ptr<Tensor> mTempDstBuffer;
    std::shared_ptr<Tensor> mTempOutBuffer;
    std::shared_ptr<Tensor> mTempTransformBuffer;
};
}

#endif //ConvInt83x3_hpp
