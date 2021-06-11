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
#import "benchmark.h"

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
#define UNITTEST
#ifdef UNITTEST
    // unittest
    {
        MNN::BackendConfig config;
        // If want to test metal, change MNN_FORWARD_CPU to MNN_FORWARD_METAL
        MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
        MNNTestSuite::runAll(config.precision);
    }
#endif
#ifdef BENCHMARK
    // benchmark
    {
        auto bundle = CFBundleGetMainBundle();
        auto url    = CFBundleCopyBundleURL(bundle);
        auto string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
        CFRelease(url);
        auto cstring = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
        auto res     = std::string(cstring) + "/models";
        CFRelease(string);
        iosBenchAll(res.c_str());
    }
#endif
    return YES;
}

@end
