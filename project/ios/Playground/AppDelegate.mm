//
//  AppDelegate.mm
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "AppDelegate.h"
#import "MNNTestSuite.h"

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    MNNTestSuite::runAll();
    return YES;
}

@end
