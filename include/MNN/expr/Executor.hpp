//
//  Executor.hpp
//  MNN
//
//  Created by MNN on 2019/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ErrorCode.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/Tensor.hpp>
#include <vector>
#include <mutex>
#include <MNN/MNNForwardType.h>
namespace MNN {
class Backend;
namespace Express {
class Solution;
class MNN_PUBLIC Executor {
public:
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
        std::vector<bool> supportError;
    };
    virtual ~Executor();
    virtual Requirement onGetRequirement(Expr* expr) const;
    virtual ErrorCode onComputeInfo(Expr* expr);
    virtual ErrorCode onComputeContent(Expr* expr);
    void recycle(Expr* expr);
    void setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread);
    enum GCFlag {
        ALL,
        UNACTIVE
    };
    void gc(GCFlag flag = ALL);
    static std::shared_ptr<Executor> getGlobalExecutor();
private:
    Executor(std::shared_ptr<Backend> backend);
    std::shared_ptr<Backend> mBackend;
    std::map<Expr*, std::shared_ptr<Solution>> mSolutions;
    std::mutex mMutex;
};
} // namespace Express
} // namespace MNN
