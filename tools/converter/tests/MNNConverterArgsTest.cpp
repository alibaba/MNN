#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli.hpp"
#include "cxxopts.hpp"

namespace {

class TestContext {
public:
    void expect(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "FAILED: " << message << std::endl;
            ++mFailures;
        }
    }

    int failures() const { return mFailures; }

private:
    int mFailures = 0;
};

bool initializeConfig(const std::string& modelFile, const std::vector<std::string>& options, modelConfig& config) {
    std::vector<std::string> arguments = {
        "MNNConverterArgsTest", "--framework=MNN", "--modelFile=" + modelFile, "--MNNModel=MNNConverterArgsTest.mnn",
        "--bizCode=MNN",
    };
    arguments.insert(arguments.end(), options.begin(), options.end());

    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) {
        argv.emplace_back(&argument[0]);
    }
    return MNN::Cli::initializeMNNConvertArgs(config, static_cast<int>(argv.size()), argv.data());
}

void testBoolParser(TestContext& test) {
    // Other strings retain legacy stream behavior and are intentionally not asserted.
    bool value = false;
    cxxopts::values::parse_value("true", value);
    test.expect(value, "textual true should parse as true");

    value = true;
    cxxopts::values::parse_value("false", value);
    test.expect(!value, "textual false should parse as false");

    value = false;
    cxxopts::values::parse_value("1", value);
    test.expect(value, "numeric 1 should parse as true");

    value = true;
    cxxopts::values::parse_value("0", value);
    test.expect(!value, "numeric 0 should parse as false");
}

void testWeightQuantAsymmetric(const std::string& modelFile, TestContext& test) {
    struct Case {
        std::vector<std::string> options;
        bool expected;
        const char* description;
    };
    const std::vector<Case> cases = {
        {{}, true, "omitted weightQuantAsymmetric should keep its default"},
        {{"--weightQuantAsymmetric"}, true, "bare weightQuantAsymmetric should enable it"},
        {{"--weightQuantAsymmetric=true"}, true, "textual true should enable weightQuantAsymmetric"},
        {{"--weightQuantAsymmetric=1"}, true, "numeric 1 should enable weightQuantAsymmetric"},
        {{"--weightQuantAsymmetric=false"}, false, "textual false should disable weightQuantAsymmetric"},
        {{"--weightQuantAsymmetric=0"}, false, "numeric 0 should disable weightQuantAsymmetric"},
        {{"--weightQuantAsymmetric=true", "--weightQuantAsymmetric=false"},
         false,
         "the last repeated weightQuantAsymmetric value should win"},
        {{"--weightQuantAsymmetric=false", "--weightQuantAsymmetric=true"},
         true,
         "the last repeated weightQuantAsymmetric value should win"},
    };

    for (const auto& testCase : cases) {
        modelConfig config;
        test.expect(initializeConfig(modelFile, testCase.options, config), "converter arguments should initialize");
        test.expect(config.weightQuantAsymmetric == testCase.expected, testCase.description);
    }
}

void testKeepInputFormat(const std::string& modelFile, TestContext& test) {
    struct Case {
        std::string option;
        bool expected;
    };
    const std::vector<Case> cases = {
        {"--keepInputFormat", true},        {"--keepInputFormat=true", true}, {"--keepInputFormat=1", true},
        {"--keepInputFormat=false", false}, {"--keepInputFormat=0", false},
    };

    for (const auto& testCase : cases) {
        modelConfig config;
        test.expect(initializeConfig(modelFile, {testCase.option}, config), "converter arguments should initialize");
        test.expect(config.keepInputFormat == testCase.expected,
                    testCase.option + " should set keepInputFormat correctly");
    }
}

void testHqqRequiresAsymmetricQuant(const std::string& modelFile, TestContext& test) {
    const std::vector<std::vector<std::string>> cases = {
        {"--weightQuantAsymmetric=false", "--hqq"},
        {"--hqq", "--weightQuantAsymmetric=false"},
    };

    for (const auto& options : cases) {
        modelConfig config;
        test.expect(initializeConfig(modelFile, options, config), "converter arguments should initialize");
        test.expect(!config.weightQuantAsymmetric, "explicit false should keep asymmetric quant disabled");
        test.expect(!config.useHQQ, "HQQ should remain disabled when asymmetric quant is disabled");
    }
}

void testHelpText(TestContext& test) {
    modelConfig config;
    std::vector<std::string> arguments = {"MNNConverterArgsTest", "--help"};
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.emplace_back(&argument[0]);
    }

    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    const bool initialized = MNN::Cli::initializeMNNConvertArgs(config, static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(previous);

    test.expect(!initialized && config.cliExitCode == 0, "--help should exit successfully");
    test.expect(output.str().find("use asymmetric weight quantization, default:") != std::string::npos &&
                    output.str().find("true; set to 0/false for symmetric") != std::string::npos,
                "help should describe the asymmetric quant default");
    test.expect(output.str().find("use HQQ weight quantization; requires") != std::string::npos &&
                    output.str().find("weightQuantAsymmetric=true, default: false") != std::string::npos,
                "help should describe the HQQ prerequisite");
}

} // namespace

int main() {
    const std::string modelFile = "MNNConverterArgsTest.model";
    {
        std::ofstream fixture(modelFile);
        if (!fixture) {
            std::cerr << "FAILED: could not create model fixture" << std::endl;
            return 1;
        }
    }

    TestContext test;
    testBoolParser(test);
    testWeightQuantAsymmetric(modelFile, test);
    testKeepInputFormat(modelFile, test);
    testHqqRequiresAsymmetricQuant(modelFile, test);
    testHelpText(test);

    std::remove(modelFile.c_str());
    if (test.failures() != 0) {
        std::cerr << test.failures() << " converter argument test(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All converter argument tests passed" << std::endl;
    return 0;
}
