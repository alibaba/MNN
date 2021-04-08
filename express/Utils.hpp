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
#include "Type_generated.h"
#include "MNN_generated.h"
#include <MNN/expr/Executor.hpp>
namespace MNN {
namespace Express {
struct BufferStorage {
    size_t size() const {
        return allocated_size - offset;
    }

    const uint8_t* buffer() const {
        return storage.get() + offset;
    }
    size_t allocated_size;
    size_t offset;
    std::unique_ptr<uint8_t> storage;
};
struct Expr::Inside {
    Inside(int outputSize);
    Inside(Tensor* tensor, bool own = false);
    ~ Inside();
    std::vector<Variable::Info> mOutputInfos;
    std::vector<Tensor*> mOutputTensors;
    Executor::Requirement mReq;
    std::shared_ptr<Executor::Unit> mUnit;
    std::shared_ptr<Executor::ComputeCache> mCache;
    int mCacheOffset = 0;
    bool mInfoDirty = true;
    bool mContentDirty = true;
    bool mOwnTensor = true;
    Tensor* mHostTensor = nullptr;
};
class Utils {
public:
    static void copyInfoToTensor(Tensor* dest, const Variable::Info* source);
    static void copyTensorToInfo(Variable::Info* dest, const Tensor* source);
    static DataType convertDataType(halide_type_t type);
    static int convertFormat(Dimensionformat format);
    static Express::Dimensionformat revertFormat(int format);
    static halide_type_t revertDataType(DataType dataType);
    static bool allocMemoryForHostTensor(Tensor* dest);
    static bool releaseMemoryForHostTensor(Tensor* dest);
};

} // namespace Express
} // namespace MNN
#endif
