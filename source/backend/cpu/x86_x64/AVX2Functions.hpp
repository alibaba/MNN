//
//  AVX2Functions.hpp
//  MNN
//
//  Created by MNN on b'2021/05/17'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AVX2Functions_hpp
#define AVX2Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "core/Macro.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "cpu_id.h"

namespace MNN {
class AVX2Functions {
public:
    static bool init(int flags);
    static CoreFunctions* get();
    static CoreInt8Functions* getInt8();
};
};

#endif
