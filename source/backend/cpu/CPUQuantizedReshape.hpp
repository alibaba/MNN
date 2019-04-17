//
//  CPUQuantizedReshape.hpp
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedReshape_hpp
#define CPUQuantizedReshape_hpp

#include "CPUBackend.hpp"

namespace MNN {
class CPUQuantizedReshape : public Execution {
public:
    CPUQuantizedReshape(const MNN::Op *op, Backend *b);
    virtual ~CPUQuantizedReshape() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool mIstflite;
};
} // namespace MNN
#endif /* CPUQuantizedReshape_hpp */
