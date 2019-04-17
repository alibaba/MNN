//
//  GLContext.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLCONTEXT_H
#define GLCONTEXT_H

#include "GLHead.h"
namespace MNN {
class GLContext {
public:
    class nativeContext;
    static nativeContext* init(int version = 2);
    static void destroy(nativeContext* context);
};
} // namespace MNN

#endif
