#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Functions_hpp
#define Arm82Functions_hpp
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "core/Macro.h"
#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
class Arm82Functions {
public:
    static bool init();
    static CoreFunctions* get();
};

};

#endif // Arm82Functions_hpp
#endif
