//
//  TestUtils.mm
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(__APPLE__)
#import "TestUtils.h"
#import <Foundation/Foundation.h>

void dispatchMetal(std::function<void(MNNForwardType)> payload, MNNForwardType backend) {
    @autoreleasepool {
        payload(backend);
    }
}
#endif
