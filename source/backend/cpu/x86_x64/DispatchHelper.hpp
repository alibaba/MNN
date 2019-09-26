//
//  DispatchHelper.hpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DISPATCH_HELPER_HPP
#define DISPATCH_HELPER_HPP

enum CPU_FEATURE {SSE, AVX};

bool cpu_feature_available(CPU_FEATURE feature);

#endif // DISPATCH_HELPER_HPP
