//
//  MNNMemoryUtils.cpp
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/MNNMemoryUtils.h"
#include <stdint.h>
#include <stdlib.h>
#include "core/Macro.h"
//#define MNN_DEBUG_MEMORY
static inline void **alignPointer(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

extern "C" void *MNNMemoryAllocAlign(size_t size, size_t alignment) {
    MNN_ASSERT(size > 0);

#ifdef MNN_DEBUG_MEMORY
    return malloc(size);
#else
    void **origin = (void **)malloc(size + sizeof(void *) + alignment);
    MNN_ASSERT(origin != NULL);
    if (!origin) {
        return NULL;
    }

    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
#endif
}

extern "C" void *MNNMemoryCallocAlign(size_t size, size_t alignment) {
    MNN_ASSERT(size > 0);

#ifdef MNN_DEBUG_MEMORY
    return calloc(size, 1);
#else
    void **origin = (void **)calloc(size + sizeof(void *) + alignment, 1);
    MNN_ASSERT(origin != NULL)
    if (!origin) {
        return NULL;
    }
    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
#endif
}

extern "C" void MNNMemoryFreeAlign(void *aligned) {
#ifdef MNN_DEBUG_MEMORY
    free(aligned);
#else
    if (aligned) {
        void *origin = ((void **)aligned)[-1];
        free(origin);
    }
#endif
}
