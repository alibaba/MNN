#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include "cli.hpp"
#include "onnx.pb.h"

struct PadCase {
    PadCase(const char* name, int opset, const char* mode, bool hasValue, float value,
            std::vector<float> input, std::vector<float> expected, int inputType = onnx::TensorProto_DataType_FLOAT)
        : name(name), opset(opset), mode(mode), hasValue(hasValue), value(value), input(input), expected(expected),
          inputType(inputType) {
    }

    const char* name;
    int opset;
    const char* mode;
    bool hasValue;
    float value;
    std::vector<float> input;
    std::vector<float> expected;
    int inputType;
};

static void addTensorInfo(onnx::ValueInfoProto* info, const char* name, int type, int size) {
    info->set_name(name);
    auto* tensorType = info->mutable_type()->mutable_tensor_type();
    tensorType->set_elem_type(type);
    tensorType->mutable_shape()->add_dim()->set_dim_value(size);
}

static bool runCase(const PadCase& test, const std::string& directory) {
    const std::string onnxPath = directory + "/" + test.name + ".onnx";
    const std::string mnnPath = directory + "/" + test.name + ".mnn";
    onnx::ModelProto model;
    model.set_ir_version(7);
    model.add_opset_import()->set_version(test.opset);
    auto* graph = model.mutable_graph();
    graph->set_name(test.name);
    addTensorInfo(graph->add_input(), "X", test.inputType, static_cast<int>(test.input.size()));
    addTensorInfo(graph->add_output(), "Y", test.inputType, static_cast<int>(test.expected.size()));
    auto* node = graph->add_node();
    node->set_name(test.name);
    node->set_op_type("Pad");
    node->add_input("X");
    node->add_output("Y");
    // Put value before mode/pads to cover attribute-order independence.
    if (test.opset < 11) {
        if (test.hasValue) {
            auto* value = node->add_attribute();
            value->set_name("value");
            value->set_type(onnx::AttributeProto_AttributeType_FLOAT);
            value->set_f(test.value);
        }
        auto* pads = node->add_attribute();
        pads->set_name("pads");
        pads->set_type(onnx::AttributeProto_AttributeType_INTS);
        pads->add_ints(1);
        pads->add_ints(0);
    } else {
        node->add_input("pads");
        auto* pads = graph->add_initializer();
        pads->set_name("pads");
        pads->set_data_type(onnx::TensorProto_DataType_INT64);
        pads->add_dims(2);
        pads->add_int64_data(1);
        pads->add_int64_data(0);
        if (test.hasValue) {
            node->add_input("constant_value");
            auto* value = graph->add_initializer();
            value->set_name("constant_value");
            value->set_data_type(onnx::TensorProto_DataType_FLOAT);
            value->add_float_data(test.value);
        }
    }
    if (test.mode != nullptr) {
        auto* mode = node->add_attribute();
        mode->set_name("mode");
        mode->set_type(onnx::AttributeProto_AttributeType_STRING);
        mode->set_s(test.mode);
    }
    {
        std::ofstream file(onnxPath, std::ios::binary);
        if (!model.SerializeToOstream(&file)) {
            return false;
        }
    }
    modelConfig converter;
    converter.model = modelConfig::ONNX;
    converter.modelFile = onnxPath;
    converter.MNNModel = mnnPath;
    converter.keepInputFormat = true;
    converter.optimizeLevel = 1;
    if (!MNN::Cli::convertModel(converter)) {
        return false;
    }
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(mnnPath.c_str()));
    if (!net) {
        return false;
    }
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig = &backend;
    auto* session = net->createSession(config);
    if (!session) {
        return false;
    }
    auto* input = net->getSessionInput(session, "X");
    if (input == nullptr || input->elementSize() != static_cast<int>(test.input.size()) ||
        input->getType() != halide_type_of<float>()) {
        return false;
    }
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    ::memcpy(hostInput.host<float>(), test.input.data(), test.input.size() * sizeof(float));
    input->copyFromHostTensor(&hostInput);
    if (net->runSession(session) != MNN::NO_ERROR) {
        return false;
    }
    auto* output = net->getSessionOutput(session, "Y");
    if (output == nullptr || output->shape() != std::vector<int>{static_cast<int>(test.expected.size())} ||
        output->getType() != halide_type_of<float>()) {
        return false;
    }
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(&hostOutput);
    for (size_t i = 0; i < test.expected.size(); ++i) {
        if (hostOutput.host<float>()[i] != test.expected[i]) {
            std::fprintf(stderr, "%s[%zu]: expected %g, got %g\n", test.name, i, test.expected[i],
                         hostOutput.host<float>()[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: TestOnnxPad EXISTING_OUTPUT_DIRECTORY\n");
        return 2;
    }
    const std::vector<PadCase> cases = {
        {"legacy_value", 10, "constant", true, 9.0f, {2.0f}, {9.0f, 2.0f}},
        {"legacy_negative", 10, "constant", true, -3.25f, {2.0f}, {-3.25f, 2.0f}},
        {"legacy_default_mode", 10, nullptr, true, 9.0f, {2.0f}, {9.0f, 2.0f}},
        {"legacy_zero", 10, "constant", true, 0.0f, {2.0f}, {0.0f, 2.0f}},
        {"legacy_default_value", 10, "constant", false, 0.0f, {2.0f}, {0.0f, 2.0f}},
        {"legacy_reflect", 10, "reflect", true, 9.0f, {2.0f, 3.0f}, {3.0f, 2.0f, 3.0f}},
        {"legacy_edge", 10, "edge", true, 9.0f, {2.0f}, {2.0f, 2.0f}},
        {"modern_value", 11, "constant", true, 9.0f, {2.0f}, {9.0f, 2.0f}},
        {"modern_zero", 11, "constant", true, 0.0f, {2.0f}, {0.0f, 2.0f}},
        {"modern_default", 11, "constant", false, 0.0f, {2.0f}, {0.0f, 2.0f}},
        {"modern_reflect", 11, "reflect", true, 9.0f, {2.0f, 3.0f}, {3.0f, 2.0f, 3.0f}},
        {"modern_edge", 11, "edge", true, 9.0f, {2.0f}, {2.0f, 2.0f}},
        {"legacy_half", 10, "constant", true, 9.0f, {2.0f}, {9.0f, 2.0f}, onnx::TensorProto_DataType_FLOAT16},
        {"legacy_double", 10, "constant", true, 9.0f, {2.0f}, {9.0f, 2.0f}, onnx::TensorProto_DataType_DOUBLE},
    };
    int failed = 0;
    for (const auto& test : cases) {
        const bool passed = runCase(test, argv[1]);
        std::printf("%s %s\n", passed ? "PASS" : "FAIL", test.name);
        failed += !passed;
    }
    std::printf("Pad cases: %zu passed, %d failed\n", cases.size() - failed, failed);
    return failed == 0 ? 0 : 1;
}
