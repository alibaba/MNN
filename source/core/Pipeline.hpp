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
class Pipeline : public NonCopyable {
public:
    struct TuningAttr {
        bool autoSetOpType;
        int maxTuningNumber;
    };
    Pipeline(Schedule::PipelineInfo&& info, bool allocInput, bool outputStatic, const TuningAttr& tune, const Runtime* rt, const Runtime* cpuRt);
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
    ErrorCode encode(bool supportDebug = false);
    /** allocMemory: create Execution and alloc memory for every op */
    ErrorCode allocMemory(bool firstMalloc);
    /** execute this pipline */
    ErrorCode execute();
    ErrorCode executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after);
    Schedule::PipelineInfo& getPipelineInfo() {
        return mInfo;
    }

    float flops() const {
        return mFlops;
    }
    friend class Session;
    MNNForwardType getMainForwardType() const  {
        return mInfo.first.cache.first->type();
    }
private:
    void _copyInputs();
    void _pushTuningTask(std::vector<Schedule::OpCacheInfo>&& initInfos);
    void _recycleDynamicMemory(Command* command);
    Schedule::PipelineInfo mInfo;
    bool mAllocInput;
    bool mOutputStatic;
    TuningAttr mTuneAttr;
    float mFlops = 0.0f;
    bool mIsQuantModel = false;

    // For gpu or other backend
    std::map<Tensor*, std::shared_ptr<Tensor>> mCacheConstTensors;
#ifndef MNN_BUILD_MINI
    GeometryComputer::Context mContext;
    Runtime::CompilerType mUseGeometry;
#endif
    const Runtime* mRuntime;
    const Runtime* mCpuRuntime;
};
} // namespace MNN

#endif /* Pipeline_hpp */
