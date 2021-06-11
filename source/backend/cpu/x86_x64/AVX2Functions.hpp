#ifndef AVX2Functions_hpp
#define AVX2Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "core/Macro.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "cpu_id.h"

namespace MNN {
class AVX2Functions {
public:
    static bool init(int flags);
    static CoreFunctions* get();
};
};

#endif
