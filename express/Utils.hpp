//
//  Utils.hpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Utils_hpp
#define Utils_hpp
#include <MNN/expr/Expr.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>

namespace MNN {
namespace Express {
struct Expr::Inside {
    std::vector<const Variable::Info*> mInputInfos;
    std::vector<Variable::Info> mOutputInfos;
    Executor::Requirement mReq;
    std::shared_ptr<Executor::ComputeCache> mCache;
};
class Utils {
public:
    static void copyInfoToTensor(Tensor* dest, const Variable::Info* source);
    static void copyTensorToInfo(Variable::Info* dest, const Tensor* source);
    static int convertDataType(halide_type_t type);
    static int convertFormat(Dimensionformat format);
    static Express::Dimensionformat revertFormat(int format);
    static halide_type_t revertDataType(int dataType);
};
} // namespace Express
} // namespace MNN
#endif
