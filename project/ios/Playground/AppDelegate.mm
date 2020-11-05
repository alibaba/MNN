//
//  AppDelegate.mm
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "AppDelegate.h"
#import "MNNTestSuite.h"
#import <MNN/expr/Executor.hpp>

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    MNN::BackendConfig config;
    // If want to test metal, change MNN_FORWARD_CPU to MNN_FORWARD_METAL
    MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
    MNNTestSuite::runAll();
    return YES;
}

@end
