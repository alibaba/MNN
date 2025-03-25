//
//  NeuronAdapterDefine.hpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NeuronAdapterDefine_h
#define NeuronAdapterDefine_h

#ifdef MNN_NeuronAdapter_ENABLED
#ifdef __ANDROID__
#include "NeuronAdapterNeuralNetworks.h"
#define ANDROID_API_LEVEL (android_get_device_api_level())
#else
#undef MNN_NeuronAdapter_ENABLED
#define MNN_NeuronAdapter_ENABLED 0
#define ANDROID_API_LEVEL (0)
#endif
#endif

#endif /* NeuronAdapterDefine_h */
