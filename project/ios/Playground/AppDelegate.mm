//
//  AppDelegate.mm
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "TestUtils.h"
#import "AppDelegate.h"
#import "MNNTestSuite.h"
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#import <MNN/expr/Executor.hpp>
#import "benchmark.h"
#define TEST_WORKMODE 0
@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
#if TEST_WORKMODE==0
    // unittest
    {
        MNN::BackendConfig config;
        // If want to test metal, change MNN_FORWARD_CPU to MNN_FORWARD_METAL
        MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
        MNNTestSuite::runAll(2);
    }
#endif
#if TEST_WORKMODE==1
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
#if TEST_WORKMODE==2
    auto bundle = CFBundleGetMainBundle();
    auto url    = CFBundleCopyBundleURL(bundle);
    auto string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
    CFRelease(url);
    auto cstring = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
    auto res     = std::string(cstring) + "/models/mobilenet_v2_auth.mnn";
    
    MNN::Interpreter* interpreter = MNN::Interpreter::createFromFile(res.c_str());
    MNN::ScheduleConfig config;
    interpreter->createSession(config);
#endif
    return YES;
}

@end
