//
//  MobileNetTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

#include <MNN/Interpreter.hpp>
#include <fstream>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"

using namespace MNN;

class MobileNetTest : public MNNTestCase {
public:
    virtual ~MobileNetTest() = default;

    std::string root() {
#ifdef __APPLE__
        auto bundle = CFBundleGetMainBundle();
        auto url    = CFBundleCopyBundleURL(bundle);
        auto string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
        CFRelease(url);
        auto cstring = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
        auto res     = std::string(cstring);
        CFRelease(string);
        return res;
#else
        return "../resource"; // assume run in build dir
#endif
    }

    std::string path() {
        return this->root() + "/model/MobileNet";
    }

    virtual std::string model()                        = 0;
    virtual std::string input()                        = 0;
    virtual std::string expect()                       = 0;
    virtual MNN::Tensor::DimensionType dimensionType() = 0;

    std::shared_ptr<Tensor> tensorFromFile(const Tensor* shape, std::string file) {
        std::shared_ptr<Tensor> result(new Tensor(shape, this->dimensionType()));
        std::ifstream stream(file.c_str());

        auto type = shape->getType();
        if (type.code == halide_type_float) {
            auto data = result->host<float>();
            auto size = result->elementSize();
            for (int i = 0; i < size; ++i) {
                stream >> data[i];
            }
        } else if (type.code == halide_type_int && type.bytes() == 4) {
            auto data = result->host<int32_t>();
            auto size = result->elementSize();
            for (int i = 0; i < size; ++i) {
                stream >> data[i];
            }
        } else if (type.code == halide_type_uint && type.bytes() == 1) {
            auto data = result->host<uint8_t>();
            auto size = result->elementSize();
            for (int i = 0; i < size; ++i) {
                int v = 0;
                stream >> v;
                data[i] = (uint8_t)v;
            }
        }

        return result;
    }

    void input(Session* session, std::string file) {
        auto input = session->getInput(NULL);
        auto given = tensorFromFile(input, file);
        input->copyFromHostTensor(given.get());
    }

    virtual bool run(int precision) {
        auto net = MNN::Interpreter::createFromFile(this->model().c_str());
        if (NULL == net) {
            return false;
        }
        ScheduleConfig cpuconfig;
        cpuconfig.type = MNN_FORWARD_CPU;
        auto CPU       = net->createSession(cpuconfig);
        auto input     = tensorFromFile(net->getSessionInput(CPU, NULL), this->input());
        auto expect    = tensorFromFile(net->getSessionOutput(CPU, NULL), this->expect());

        dispatch([&](MNNForwardType backend) -> void {
            ScheduleConfig config;
            config.type  = MNN_FORWARD_METAL;
            config.numThread  = 1;
            MNN::BackendConfig backendConfig;
            backendConfig.precision = MNN::BackendConfig::Precision_High;
            config.backendConfig = &backendConfig;
            
            auto session = net->createSession(config);
            
            auto outputTensor  = net->getSessionOutput(session, NULL);
            auto inputTensor   = net->getSessionInput(session, NULL);
            std::shared_ptr<MNN::Tensor> hostTensor(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));
            for(int i=0; i<20; i++)//warmm up
            {
                auto timeBegin = getTimeInUs();
                inputTensor->copyFromHostTensor(input.get());
                net->runSession(session);
                outputTensor->copyToHostTensor(hostTensor.get());

                auto timeEnd = getTimeInUs();
                printf("run cost %f ms\n", ((timeEnd - timeBegin) / 1000.0));
            }
            
            float tolerance = backend == MNN_FORWARD_CPU ? 0.04 : 0.1;
            assert(TensorUtils::compareTensors(hostTensor.get(), expect.get(), tolerance, true));
        });
        delete net;
        return true;
    }
};

class MobileNetV1Test : public MobileNetTest {
    virtual ~MobileNetV1Test() = default;
    virtual std::string model() override {
        return this->path() + "/v1/mobilenet_v1.caffe.mnn";
    }
    virtual std::string input() override {
        return this->path() + "/flt_input.txt";
    }
    virtual std::string expect() override {
        return this->path() + "/v1/expect.txt";
    }
    virtual MNN::Tensor::DimensionType dimensionType() override {
        return MNN::Tensor::CAFFE;
    }
};

class MobileNetV2Test : public MobileNetTest {
    virtual ~MobileNetV2Test() = default;
    virtual std::string model() override {
        return this->path() + "/v2/mobilenet_v2.caffe.mnn";
    }
    virtual std::string input() override {
        return this->path() + "/flt_input.txt";
    }
    virtual std::string expect() override {
        return this->path() + "/v2/expect_caffe.txt";
    }
    virtual MNN::Tensor::DimensionType dimensionType() override {
        return MNN::Tensor::CAFFE;
    }
};

class MobileNetV2TFLiteTest : public MobileNetTest {
    virtual ~MobileNetV2TFLiteTest() = default;
    virtual std::string model() override {
        return this->path() + "/v2/mobilenet_v2_1.0_224.tflite.mnn";
    }
    virtual std::string input() override {
        return this->path() + "/flt_input.txt";
    }
    virtual std::string expect() override {
        return this->path() + "/v2/expect_tflite.txt";
    }
    virtual MNN::Tensor::DimensionType dimensionType() override {
        return MNN::Tensor::TENSORFLOW;
    }
};

class MobileNetV2TFLiteQntTest : public MobileNetTest {
    virtual ~MobileNetV2TFLiteQntTest() = default;
    virtual std::string model() override {
        return this->path() + "/v2/mobilenet_v2_1.0_224_quant.tflite.mnn";
    }
    virtual std::string input() override {
        return this->path() + "/qnt_input.txt";
    }
    virtual std::string expect() override {
        return this->path() + "/v2/expect_tflite_qnt.txt";
    }
    virtual MNN::Tensor::DimensionType dimensionType() override {
        return MNN::Tensor::TENSORFLOW;
    }
};

MNNTestSuiteRegister(MobileNetV1Test, "model/mobilenet/1/caffe");
MNNTestSuiteRegister(MobileNetV2Test, "model/mobilenet/2/caffe");
MNNTestSuiteRegister(MobileNetV2TFLiteTest, "model/mobilenet/2/tflite");
MNNTestSuiteRegister(MobileNetV2TFLiteQntTest, "model/mobilenet/2/tflite_qnt");


class ModelTest : public MNNTestCase {
public:
    virtual ~ModelTest() = default;

    std::string root() {
#ifdef __APPLE__
        auto bundle = CFBundleGetMainBundle();
        auto url    = CFBundleCopyBundleURL(bundle);
        auto string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
        CFRelease(url);
        auto cstring = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
        auto res     = std::string(cstring);
        CFRelease(string);
        return res;
#else
        return "../resource"; // assume run in build dir
#endif
    }

    std::string path() {
        return this->root() + "/model/temp.bin";
    }

    virtual bool run(int precision) {
        auto net = MNN::Interpreter::createFromFile(this->path().c_str());
        if (NULL == net) {
            return false;
        }
        ScheduleConfig cpuconfig;
        cpuconfig.type = MNN_FORWARD_CPU;
        BackendConfig bnConfig;
        bnConfig.precision = BackendConfig::Precision_Low;
        cpuconfig.backendConfig = &bnConfig;
        auto session       = net->createSession(cpuconfig);
        net->runSession(session);
        delete net;
        return true;
    }
};
MNNTestSuiteRegister(ModelTest, "model/model_test");
