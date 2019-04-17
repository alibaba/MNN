//
//  CPULSTM.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPULSTM_hpp
#define CPULSTM_hpp

#include "AutoStorage.h"
#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {

class CPULSTM : public Execution {
public:
    CPULSTM(Backend *backend, const LSTM *LSTM);
    virtual ~CPULSTM();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const LSTM *mLSTM;

    bool mInit = false;
    std::shared_ptr<Tensor> mWeightI;
    std::shared_ptr<Tensor> mWeightH;
    std::shared_ptr<Tensor> mBiasC;

    Tensor mInput;
    Tensor mCont;
    Tensor mGates;
    Tensor mCell;
    Tensor mOutput;
};

} // namespace MNN

#endif /* CPULSTM_hpp */
