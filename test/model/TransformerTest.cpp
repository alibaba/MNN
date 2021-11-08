//
//  TransformerTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/06/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

#include <fstream>
#include <iostream>
#include <string>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

#define TEST_OR_RETURN(expr, ...)   \
    {                               \
        if (!(expr)) {              \
            MNN_ERROR(__VA_ARGS__); \
            return false;           \
        }                           \
    }

class TransformerTest : public MNNTestCase {
public:
    virtual ~TransformerTest() = default;

    std::string root() const {
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

    std::string path() const {
        return this->root() + "/model/Transformer";
    }
    std::string model_path() const {
        return path() + "/transformer.mnn";
    }
    std::string input_path() const {
        return path() + "/input.txt";
    }

    void SetupGlobalExecutor() {
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        config.precision = MNN::BackendConfig::Precision_Normal;
        config.power     = MNN::BackendConfig::Power_High;
        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1 /*num_threads*/);
    }

    int ReadInputFromFile(const char* input_file, std::vector<float>* input) {
        std::ifstream f(input_file);
        float input_data = 0.f;
        while (f.peek() != EOF) {
            f >> input_data;
            if (f.eof()) {
                break;
            }
            input->push_back(input_data);
        }
        f.close();
        return input->size();
    }

    virtual bool run(int precision) {
        SetupGlobalExecutor();

        auto varMap = Variable::loadMap(model_path().c_str());
        std::vector<float> input;
        ReadInputFromFile(input_path().c_str(), &input);
        TEST_OR_RETURN(input.size() == 600 * 80, "Input size mismatch. %d is expected, but get %d.\n", 600 * 80,
                       int(input.size()));
        for (int i = 0; i < input.size(); ++i) {
            varMap["tf_loss_fn/Placeholder"]->writeMap<float>()[i] = input.at(i);
        }
        varMap["tf_loss_fn/Placeholder_1"]->writeMap<int>()[0] = 600;

        auto output      = varMap["tf_loss_fn/ForwardPass/jca_decoder/transformer_decoder/decode/strided_slice_3"];
        const auto& dims = output->getInfo()->dim;
        TEST_OR_RETURN(dims.size() == 2, "Output dimension should be 2, other than %d.\n", int(dims.size()));
        TEST_OR_RETURN(dims[0] == 1, "%s\n", "Dimension 0 should be 1.");
        TEST_OR_RETURN(dims[1] == 45, "%s\n", "Dimension 1 should be 45.");
        float sum = 0.f;
        for (int i = 0; i < output->getInfo()->size; ++i) {
            sum += output->readMap<int>()[i];
        }
        TEST_OR_RETURN(sum == 8300, "%s\n", "The sum of output should be 8300.\n");
        return true;
    }
};

#undef TEST_OR_RETURN

// MNNTestSuiteRegister(TransformerTest, "model/transformer");
