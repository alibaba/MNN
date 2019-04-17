//
//  AutoTime.hpp
//  MNN
//
//  Created by MNN on 2018/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AutoTime_hpp
#define AutoTime_hpp

#include <stdint.h>
#include <stdio.h>
#include "MNNDefine.h"

namespace MNN {

/** time tracing util. prints duration between init and deinit. */
class MNN_PUBLIC AutoTime {
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
    uint64_t mCurrentTime;
};
} // namespace MNN

#ifdef MNN_OPEN_TIME_TRACE
#define AUTOTIME MNN::AutoTime ___t(__LINE__, __func__)
#else
#define AUTOTIME
#endif

#endif /* AutoTime_hpp */
