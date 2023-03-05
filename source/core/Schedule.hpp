//
//  Schedule.hpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Schedule_hpp
#define Schedule_hpp

#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <map>
#include <string>
#include <vector>
#include <array>
#include "core/Backend.hpp"
#include "core/Command.hpp"
namespace MNN {

struct Op;
struct Net;

/** net scheduler */
class MNN_PUBLIC Schedule {
public:
    enum Type {
        // Size can be compute separately
        SEPARATE = 0,
        // When size is fixed, the content is fixed
        CONSTANT = 1,
        // Size can't be compute separately
        NOT_SEPERATE
    };
    /** pipeline info */
    struct OpCacheInfo {
        /** op */
        const Op* op;
        /** input tensors */
        std::vector<Tensor*> inputs;
        /** output tensors */
        std::vector<Tensor*> outputs;
        /** schedule type*/
        Schedule::Type type = Schedule::Type::SEPARATE;

        /**Command buffer for cache*/
        CommandBuffer cacheBuffer;

        /**Command buffer for execute*/
        CommandBuffer executeBuffer;
        
        std::map<const Op*, std::shared_ptr<Execution>> executionCache;
    };

    // Backend, Tensor, shape-dirty, content-dirty
    typedef std::tuple<Tensor*, std::shared_ptr<Tensor>, bool, bool> TENSORCACHE;
    struct BackendCache {
        Backend::Info info;
        BackendConfig config;
        std::pair<std::shared_ptr<Backend>, std::shared_ptr<Backend>> cache;
        bool needComputeShape = true;
        bool needComputeGeometry = true;
        std::map<Tensor*, TENSORCACHE> inputTensorCopyCache;
    };
    typedef std::pair<BackendCache, std::vector<OpCacheInfo>> PipelineInfo;

    /** schedule info */
    struct ScheduleInfo {
        /** pipelines with backend info */
        std::vector<PipelineInfo> pipelineInfo;
        /** input tensors map */
        std::map<std::string, Tensor*> inputTensors;
        /** output tensors map */
        std::map<std::string, Tensor*> outputTensor;
        /** all tensors */
        std::vector<std::shared_ptr<Tensor>> allTensors;
        /** input valid for resize*/
        bool validForResize;
        /** Default Backend*/
        std::shared_ptr<Backend> defaultBackend;
        /** size need input's content*/
        bool needInputContentForShape = false;
    };

    /**
     * @breif schedule net ops to pipeline with configuration.
     * @param net       given net.
     * @param config    given configuration.
     * @return schedule info.
     */
    static bool schedule(ScheduleInfo& result, const Net* net, const std::vector<ScheduleConfig>& config, const RuntimeInfo& runtimeInfo);
    static MNNForwardType getApprociateType(const ScheduleConfig& config);
};
} // namespace MNN

#endif /* Schedule_hpp */
