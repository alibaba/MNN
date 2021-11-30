//
//  Pipeline.hpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Pipeline_hpp
#define Pipeline_hpp

#include "Schedule.hpp"
#include "core/Execution.hpp"
#include "geometry/GeometryComputer.hpp"

namespace MNN {
struct OperatorInfo::Info {
    std::string name;
    std::string type;
    float flops = 0.0f;
};
class SizeComputer;
/** pipeline. one session may contains multiple pipeline, and one pipeline may contains more than one unit. */
typedef std::map<const Op*, std::pair<std::shared_ptr<Execution>, DataType>> CacheExecutionMap;
class Pipeline : public NonCopyable {
public:
    Pipeline(std::vector<Schedule::PipelineInfo>&& info, std::shared_ptr<Backend> major,
             std::shared_ptr<Backend> backup, std::shared_ptr<Backend> constBackend, bool allocInput, bool outputStatic, Runtime::CompilerType compilerType, CacheExecutionMap& cache);
    ~Pipeline();
    class UnitInfo : public OperatorInfo {
    public:
        UnitInfo()          = default;
        virtual ~UnitInfo() = default;
        void setUp(const Command& cmd, int index, const Op* originOp, int totalIndex);
    };
public:
    /** encode :
       1. compute shape for every op's inputs and outputs;
       2. geometry transform;
       3. copy op, inputs and outputs tensor info to mBuffer
       static_model:  3; dynamic_model: 1,2,3
    */
    ErrorCode encode(bool isStatic = false, bool supportDebug = false);
    /** allocMemory: create Execution and alloc memory for every op */
    ErrorCode allocMemory(bool firstMalloc);
    /** execute this pipline */
    ErrorCode execute();
    ErrorCode executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after);
    std::vector<Schedule::PipelineInfo>& getPipelineInfo();

    float flops() const {
        return mFlops;
    }
    friend class Session;
private:
    std::shared_ptr<Backend> mBackend, mBackupBackend, mConstBackend;
    std::vector<Schedule::PipelineInfo> mInfo;
    bool mAllocInput;
    bool mOutputStatic;
    bool mInit = false;
    float mFlops = 0.0f;
    bool mIsQuantModel = false;
    CacheExecutionMap& mOriginExecution;

    // For gpu or other backend
    std::map<Tensor*, std::shared_ptr<Tensor>> mCacheConstTensors;
#ifndef MNN_BUILD_MINI
    GeometryComputer::Context mContext;
    Runtime::CompilerType mUseGeometry;
#endif
};
} // namespace MNN

#endif /* Pipeline_hpp */
