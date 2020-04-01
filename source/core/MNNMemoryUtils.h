//
//  MNNMemoryUtils.h
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNMemoryUtils_h
#define MNNMemoryUtils_h

#include <stdio.h>
#include "core/Macro.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MNN_MEMORY_ALIGN_DEFAULT 64

/**
 * @brief alloc memory with given size & alignment.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
MNN_PUBLIC void* MNNMemoryAllocAlign(size_t size, size_t align);

/**
 * @brief alloc memory with given size & alignment, and fill memory space with 0.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
MNN_PUBLIC void* MNNMemoryCallocAlign(size_t size, size_t align);

/**
 * @brief free aligned memory pointer.
 * @param mem   aligned memory pointer.
 * @warning do NOT pass any pointer NOT returned by `MNNMemoryAllocAlign` or `MNNMemoryCallocAlign`.
 * @sa MNNMemoryAllocAlign
 * @sa MNNMemoryCallocAlign
 */
MNN_PUBLIC void MNNMemoryFreeAlign(void* mem);

#ifdef __cplusplus
}
#endif

#endif /* MNNMemoryUtils_h */
