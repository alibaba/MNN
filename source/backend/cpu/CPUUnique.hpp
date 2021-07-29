//
//  CPUUnique.hpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUUnique_hpp
#define CPUUnique_hpp

#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
class CPUUnique : public Execution {
public:
    CPUUnique(Backend *bn) : Execution(bn) {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
};     // namespace MNN
#endif /* CPUUnique_hpp */
