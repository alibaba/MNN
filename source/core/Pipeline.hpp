//
//  Pipeline.hpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Pipeline_hpp
#define Pipeline_hpp

#include "Execution.hpp"
#include "Schedule.hpp"

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
    /**
     * @brief initialize with pipeline info, major backend and backup backend (usually CPU).
     * @param info      given pipeline info.
     * @param major     given major backend used to create execution.
     * @param backup    given backend backend if op is not supported by major backend.
     */
    Pipeline(const std::vector<Schedule::PipelineInfo>& info, Backend* major, Backend* backup);

public:
    /**
     * @brief prepare all units.
     * @return result code.
     */
    ErrorCode prepare();
    /**
     * @brief execute all units.
     * @return result code.
     */
    ErrorCode execute();
    /**
     * @brief execute all units with callbacks.
     * @param before    callback before execute each op.
     * @param after     callback after execute each op.
     * @return result code.
     */
    ErrorCode executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after);
    /**
     * @brief the Pipline need not prepare any more, release all cache used for resize.
     * @return errorcode
     */
    ErrorCode releaseCache();

    /** op unit in pipeline */
    class Unit : public NonCopyable, public OperatorInfo {
    public:
        /**
         * @brief initialize with given op and its in-out tensors.
         * @param op        given op.
         * @param inputs    execution input tensors.
         * @param outputs   execution output tensors.
         */
        Unit(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

        /**
         * @brief prepare unit.
         * @return result code.
         */
        ErrorCode prepare(Backend* major, Backend* backup);
        /**
         * @brief execute unit.
         * @return result code.
         */
        ErrorCode execute();
        /**
         * @brief execute unit with callbacks.
         * @param before    callback before execute each op.
         * @param after     callback after execute each op.
         * @return result code.
         */
        ErrorCode executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after);

    public:
        /** op execution */
        std::shared_ptr<Execution> mExecution;
        /** op type*/
        OpType mType;
        /** input tensors */
        std::vector<Tensor*> mInputs;
        /** output tensors */
        std::vector<Tensor*> mOutputs;
        /** op */
        const Op* mOriginOp;

    private:
        bool _createExecution(Backend* bn, Backend* cpuBn);
        bool _allocTensors(Backend* bn, const std::vector<Tensor*>& tensors);

    private:
        bool mConst                   = false;
        const SizeComputer* mComputer = nullptr;
    };

protected:
    /*Used for Unit Test*/
    const std::vector<std::shared_ptr<Unit>>& getUnit() const {
        return this->mUnits;
    }

private:
    Backend* mBackend;
    Backend* mBackupBackend;
    std::vector<std::shared_ptr<Unit>> mUnits;
};
} // namespace MNN

#endif /* Pipeline_hpp */
