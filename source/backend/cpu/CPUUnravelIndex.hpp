//
//  CPUUnravelIndex.hpp
//  MNN
//
//  Created by MNN on 2018/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnravelIndex_hpp
#define CPUUnravelIndex_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUUnravelIndex : public Execution{
public:
    CPUUnravelIndex(Backend *b):Execution(b){
    }
    virtual ~CPUUnravelIndex() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUUnravelIndex_hpp */
