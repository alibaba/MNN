//
//  AutoTime.hpp
//  MNN
//
//  Created by MNN on 2018/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_AutoTime_hpp
#define MNN_AutoTime_hpp

#include <stdint.h>
#include <stdio.h>
#include <MNN/MNNDefine.h>

namespace MNN {

class MNN_PUBLIC Timer {
public:
    Timer();
    ~Timer();
    Timer(const Timer&)  = delete;
    Timer(const Timer&&) = delete;
    Timer& operator=(const Timer&)  = delete;
    Timer& operator=(const Timer&&) = delete;
    
    // reset timer
    void reset();
    // get duration (us) from init or latest reset.
    uint64_t durationInUs();
    
    // Get Current Time
    uint64_t current() const {
        return mLastResetTime;
    }
protected:
    uint64_t mLastResetTime;
};

/** time tracing util. prints duration between init and deinit. */
class MNN_PUBLIC AutoTime : Timer {
public:
    AutoTime(int line, const char* func);
    ~AutoTime();
    AutoTime(const AutoTime&)  = delete;
    AutoTime(const AutoTime&&) = delete;
    AutoTime& operator=(const AutoTime&) = delete;
    AutoTime& operator=(const AutoTime&&) = delete;

private:
    int mLine;
    char* mName;
};
} // namespace MNN

#ifdef MNN_OPEN_TIME_TRACE
#define AUTOTIME MNN::AutoTime ___t(__LINE__, __func__)
#else
#define AUTOTIME
#endif

#endif /* AutoTime_hpp */
