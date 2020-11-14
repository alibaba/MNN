//
//  PluginFactory.hpp
//  MNN
//
//  Created by MNN on b'2020/08/12'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PluginFactory_hpp
#define PluginFactory_hpp

#include <MNN/MNNDefine.h>
#include <stdint.h>
extern "C" {
MNN_PUBLIC void* MNNTRTCreatePlugion(const void* opRaw, const void* extraInfo);
MNN_PUBLIC void* MNNTRTCreatePlugionSerial(const char* layerName, const void* serialData, size_t serialLength);
}
#endif