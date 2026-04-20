//
//  RVVFunctions.hpp
//  MNN
//
//  Created by ihb2032 on 2026/04/20.
//  Email: hebome@foxmail.com
//
#ifndef RVVFunctions_hpp
#define RVVFunctions_hpp

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#include "../compute/Int8FunctionsOpt.h"

namespace MNN {
class RVVFunctions {
public:
    static bool init();
    static CoreFunctions* get();
    static CoreInt8Functions* getInt8();
};
} // namespace MNN

#endif
