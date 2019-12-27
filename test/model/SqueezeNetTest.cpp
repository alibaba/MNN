//
//  SqueezeNetTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

#include <fstream>
#include <MNN/Interpreter.hpp>
#include "MNNTestSuite.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

class SqueezeNetTest : public MNNTestCase {
public:
    virtual ~SqueezeNetTest() = default;

    std::string root() {
#ifdef __APPLE__
        auto bundle = CFBundleGetMainBundle();
        auto url    = CFBundleCopyBundleURL(bundle);
        auto string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
        CFRelease(url);
        auto cstring = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
        CFRelease(string);
        return std::string(cstring);
#else
        return "../resource"; // assume run in build dir
#endif
    }

    std::string path() {
        return this->root() + "/model/SqueezeNet";
    }

    virtual std::string model() = 0;
    virtual std::string input() {
        return this->path() + "/input.txt";
    }
    virtual std::string expect() = 0;

    std::shared_ptr<Tensor> tensorFromFile(const Tensor* shape, std::string file) {
        std::shared_ptr<Tensor> result(new Tensor(shape, MNN::Tensor::CAFFE));
        std::ifstream stream(file.c_str());
        auto data = result->host<float>();
        auto size = result->elementSize();
        for (int i = 0; i < size; ++i) {
            stream >> data[i];
        }
        return result;
    }

    void input(Session* session, std::string file) {
        auto input = session->getInput(NULL);
        auto given = tensorFromFile(input, file);
        input->copyFromHostTensor(given.get());
    }

    virtual bool run() {
        auto net = MNN::Interpreter::createFromFile(this->model().c_str());
        if (NULL == net) {
            return false;
        }
        auto CPU    = createSession(net, MNN_FORWARD_CPU);
        auto input  = tensorFromFile(CPU->getInput(NULL), this->input());
        auto expect = tensorFromFile(CPU->getOutput(NULL), this->expect());

        dispatch([&](MNNForwardType backend) -> void {
            auto session = createSession(net, backend);
            session->getInput(NULL)->copyFromHostTensor(input.get());
            session->run();
            auto output = session->getOutput(NULL);
            assert(TensorUtils::compareTensors(output, expect.get(), 0.01, true));
        });
        delete net;
        return true;
    }
};

class SqueezeNetV1_0Test : public SqueezeNetTest {
    virtual ~SqueezeNetV1_0Test() = default;
    virtual std::string model() {
        return this->path() + "/v1.0/squeezenet_v1.0.caffe.mnn";
    }
    virtual std::string expect() {
        return this->path() + "/v1.0/expect.txt";
    }
};

class SqueezeNetV1_1Test : public SqueezeNetTest {
    virtual ~SqueezeNetV1_1Test() = default;
    virtual std::string model() override {
        return this->path() + "/v1.1/squeezenet_v1.1.caffe.mnn";
    }
    virtual std::string expect() override {
        return this->path() + "/v1.1/expect.txt";
    }
};

MNNTestSuiteRegister(SqueezeNetV1_0Test, "model/squeezenet/1.0");
MNNTestSuiteRegister(SqueezeNetV1_1Test, "model/squeezenet/1.1");
