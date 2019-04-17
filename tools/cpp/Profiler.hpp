//
//  Profiler.hpp
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Profiler_hpp
#define Profiler_hpp

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
#include "Interpreter.hpp"
#include "Tensor.hpp"

namespace MNN {

/** Profiler for Ops */
class Profiler {
public:
    /**
     * @brief get shared instance.
     */
    static Profiler* getInstance();
    /**
     * @brief start profiler with op, name and inout tensors.
     * @param op        given op.
     */
    void start(const OperatorInfo* info);
    /**
     * @brief end profiler with op name and type.
     * @param name      op name.
     */
    void end(const OperatorInfo* info);
    /**
     * print profiler time result grouped and sorter by type.
     * @param loops     loop count.
     */
    void printTimeByType(int loops = 1);

private:
    ~Profiler() = default;

private:
    struct Record {
        std::string name;
        std::string type;
        int64_t order;
        int64_t calledTimes;
        float costTime;
        float flops;
    };

    static Profiler* gInstance;
    uint64_t mStartTime = 0;
    uint64_t mEndTime   = 0;
    uint32_t mOrder     = 0;
    float mTotalTime    = 0.0f;
    float mTotalMFlops  = 0.0f;
    std::map<std::string, Record> mMapByType;

private:
    Record& getTypedRecord(const OperatorInfo* info);
};

} // namespace MNN

#endif /* Profiler_hpp */
