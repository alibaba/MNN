//
//  InsideExpr.hpp
//  MNN
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InsideExpr_hpp
#define InsideExpr_hpp

#include "Solution.hpp"
namespace MNN {
class Backend;
namespace Express {
class DefaultSolutionCreator : public Executor {
public:
    DefaultSolutionCreator();
    virtual ~DefaultSolutionCreator() = default;
    virtual Solution* onCreate(const Op* op, int inputSize, int outputSize) override;

private:
    std::shared_ptr<Backend> mBackend;
};
}; // namespace Express
}; // namespace MNN

#endif /* InsideExpr_hpp */
