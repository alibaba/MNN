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
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#import "benchmark.h"
#define TEST_WORKMODE 2
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
    auto res     = std::string(cstring) + "/model/MobileNet/v1/mobilenet_v1.caffe.mnn";
    CFRelease(string);
    
    MNN::Interpreter* interpreter = MNN::Interpreter::createFromFile(res.c_str());
    interpreter->setSessionHint(MNN::Interpreter::GEOMETRY_COMPUTE_MASK, 0);
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_NN;
    config.numThread = 1;
    MNN::BackendConfig bnC;
    bnC.precision = MNN::BackendConfig::Precision_Normal;
    config.backendConfig = &bnC;
    auto session = interpreter->createSession(config);
    auto inpDev = interpreter->getSessionInput(session, nullptr);
    auto outDev = interpreter->getSessionOutput(session, nullptr);
    auto input = std::shared_ptr<MNN::Tensor>(new MNN::Tensor(inpDev));
    auto output = std::shared_ptr<MNN::Tensor>(new MNN::Tensor(outDev));
    auto inputHost = input->host<float>();
    int inputSize = input->elementSize();
    for (int v=0; v<inputSize; ++v) {
        inputHost[v] = (float)rand() / RAND_MAX;
    }
    auto outputHost = output->host<float>();
    int outputSize = output->elementSize();

    for (int i=0; i<2; ++i) {
        inpDev->copyFromHostTensor(input.get());
        interpreter->runSession(session);
        outDev->copyToHostTensor(output.get());
        float sum = 0.0f;
        float maxv = 0.0f;
        float minv = 0.0f;
        for (int v=0; v<outputSize; ++v) {
            float value = outputHost[v];
            maxv = ALIMAX(maxv, value);
            minv = ALIMIN(minv, value);
            sum += value;
        }
        float mean = sum / (float)outputSize;
        MNN_PRINT("Size:%d, Max:%f, Min:%f, Avg:%f\n", outputSize, maxv, minv, mean);
    }
    {
        AUTOTIME;
        for (int i=0; i<10; ++i) {
            inpDev->copyFromHostTensor(input.get());
            interpreter->runSession(session);
            outDev->copyToHostTensor(output.get());
        }
    }
    delete interpreter;
#endif
    return YES;
}

@end
