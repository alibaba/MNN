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
#include "Type_generated.h"
#include "MNN_generated.h"
#include <MNN/expr/Executor.hpp>
#include "core/AutoStorage.h"
namespace MNN {
class Session;
namespace Express {
struct Expr::Inside {
    Inside(int outputSize);
    Inside(Tensor* tensor, bool own = false);
    ~ Inside();
    std::vector<Variable::Info> mOutputInfos;
    std::vector<Tensor*> mOutputTensors;
    Executor::Requirement mReq;
    std::shared_ptr<Executor::ComputeCache> mCache;
    int mCacheOffset = 0;
    bool mInfoDirty = true;
    bool mContentDirty = true;
    bool mOwnTensor = true;
    Tensor* mHostTensor = nullptr;
    std::shared_ptr<Backend> mHoldBackend;
};
struct Executor::DebugTools {
    TensorCallBackWithInfo before = nullptr;
    TensorCallBackWithInfo after = nullptr;
    mutable float flops = 0.0f;
};
struct Executor::SubGraph {
    std::unique_ptr<MNN::SubGraphProtoT> info;
    std::vector<std::string> depends;
};
class Executor::ComputeCache {
public:
    void setShapeDirty();
    void setContentDirty();
    void* mapOutput(int offset, Tensor* dest);

    ~ ComputeCache();
    ComputeCache() {
        // Do nothing
    }

    ErrorCode compute();
    ErrorCode resize();
    ErrorCode resizeImpl();
    Session* getSession() {
        return mSession.get();
    }
    friend class Executor;
private:
    std::set<std::shared_ptr<Expr::Inside>> mInputInside;
    std::set<std::shared_ptr<ComputeCache>> mInputs;
    std::shared_ptr<Session> mSession;
    bool mContentDirty = true;
    bool mShapeDirty = true;
    std::vector<std::shared_ptr<BufferStorage>> mCacheBuffers;
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
    static int gInstanceCount;
#endif
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
    static Tensor* getTensor(VARP var);
    static EXPRP makeRaster(const std::vector<VARP>& vars, const std::vector<int>& regions, const std::vector<int>& shape, halide_type_t dataType, MNN_DATA_FORMAT format);
};

} // namespace Express
} // namespace MNN
#endif
