//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

namespace mnncli {
#ifdef __ANDROID__
   // On Android, use a relative path in the mnncli directory to avoid filesystem permission issues
   const char* const kCachePath = ".mnnmodels";
#else
   // On other platforms, use the home directory
   const char* const kCachePath = "~/.mnnmodels";
#endif
}