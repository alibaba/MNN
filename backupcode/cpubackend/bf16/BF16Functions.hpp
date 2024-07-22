#ifndef BF16Functions_hpp
#define BF16Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
namespace MNN {
class BF16Functions {
public:
    static bool init();
    static CoreFunctions* get();
};
};

#endif
