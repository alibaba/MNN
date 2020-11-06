//
//  TRTDynLoad.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTDynLoad.hpp"

namespace MNN {

std::once_flag TRTDsoFlag;
void *TRTDsoHandle;

#define MNNDL_FUNC_NAME(__name) MNNDL__##__name __name
TRT_TYPE_DEFINE(MNNDL_FUNC_NAME);

} // namespace MNN
