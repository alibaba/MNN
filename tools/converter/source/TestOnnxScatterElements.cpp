#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "cli.hpp"
#include "onnx.pb.h"

static void addTensorShape(onnx::ValueInfoProto* valueInfo, const std::string& name) {
    valueInfo->set_name(name);
    auto* tensorType = valueInfo->mutable_type()->mutable_tensor_type();
    tensorType->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    auto* shape = tensorType->mutable_shape();
    shape->add_dim()->set_dim_value(2);
    shape->add_dim()->set_dim_value(3);
}

static onnx::ModelProto makeModel(int axis, bool hasAxis, const std::string& reduction, bool reductionFirst) {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(16);
    auto* graph = model.mutable_graph();
    graph->set_name("ScatterElementsAttributes");
    addTensorShape(graph->add_input(), "data");
    addTensorShape(graph->add_output(), "output");

    // The targets are unique and in bounds for both axis 0 and axis 1, so a
    // dropped attribute produces a deterministic wrong result, not invalid input.
    auto* indices = graph->add_initializer();
    indices->set_name("indices");
    indices->set_data_type(onnx::TensorProto_DataType_INT64);
    indices->add_dims(2);
    indices->add_dims(2);
    for (int64_t value : {1, 0, 0, 1}) {
        indices->add_int64_data(value);
    }
    auto* updates = graph->add_initializer();
    updates->set_name("updates");
    updates->set_data_type(onnx::TensorProto_DataType_FLOAT);
    updates->add_dims(2);
    updates->add_dims(2);
    for (float value : {17.0f, 19.0f, 23.0f, 29.0f}) {
        updates->add_float_data(value);
    }
    auto* node = graph->add_node();
    node->set_op_type("ScatterElements");
    node->add_input("data");
    node->add_input("indices");
    node->add_input("updates");
    node->add_output("output");
    auto addAxis = [&]() {
        if (hasAxis) {
            auto* attr = node->add_attribute();
            attr->set_name("axis");
            attr->set_type(onnx::AttributeProto_AttributeType_INT);
            attr->set_i(axis);
        }
    };
    auto addReduction = [&]() {
        if (!reduction.empty()) {
            auto* attr = node->add_attribute();
            attr->set_name("reduction");
            attr->set_type(onnx::AttributeProto_AttributeType_STRING);
            attr->set_s(reduction);
        }
    };
    if (reductionFirst) {
        addReduction();
        addAxis();
    } else {
        addAxis();
        addReduction();
    }
    return model;
}

static bool runModel(const std::string& modelPath, const std::vector<float>& expected) {
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!net) {
        return false;
    }
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    config.backendConfig = &backend;
    auto* session = net->createSession(config);
    if (!session) {
        return false;
    }
    auto* input = net->getSessionInput(session, "data");
    const std::vector<int> shape = {2, 3};
    if (!input || input->shape() != shape || input->getType() != halide_type_of<float>()) {
        return false;
    }
    const float data[] = {2, 3, 5, 7, 11, 13};
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    std::memcpy(hostInput.host<float>(), data, sizeof(data));
    if (!input->copyFromHostTensor(&hostInput) || net->runSession(session) != MNN::NO_ERROR) {
        return false;
    }
    auto* output = net->getSessionOutput(session, "output");
    if (!output || output->shape() != shape || output->getType() != halide_type_of<float>()) {
        return false;
    }
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    if (!output->copyToHostTensor(&hostOutput)) {
        return false;
    }
    const float* got = hostOutput.host<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        // All values are exactly representable in float32.
        if (!std::isfinite(got[i]) || got[i] != expected[i]) {
            std::fprintf(stderr, "mismatch at %zu: expected=%g, got=%g\n", i, expected[i], got[i]);
            return false;
        }
    }
    return true;
}

static bool runCase(const std::string& prefix, int axis, bool hasAxis, const std::string& reduction,
                    bool reductionFirst) {
    const std::string name = std::string(hasAxis ? "axis" + std::to_string(axis) : "default_axis") + "_" +
                             (reduction.empty() ? "default_reduction" : reduction) +
                             (reductionFirst ? "_reduction_first" : "_axis_first");
    const std::string onnxPath = prefix + name + ".onnx";
    const std::string mnnPath = prefix + name + ".mnn";
    bool ok;
    {
        std::ofstream stream(onnxPath, std::ios::binary | std::ios::trunc);
        ok = makeModel(axis, hasAxis, reduction, reductionFirst).SerializeToOstream(&stream);
    }
    if (ok) {
        modelConfig config;
        config.model = modelConfig::ONNX;
        config.modelFile = onnxPath;
        config.MNNModel = mnnPath;
        config.keepInputFormat = true;
        ok = MNN::Cli::convertModel(config);
    }
    std::vector<float> expected;
    if (axis == 0) {
        expected = reduction == "add"   ? std::vector<float>{25, 22, 5, 24, 40, 13}
                   : reduction == "mul" ? std::vector<float>{46, 57, 5, 119, 319, 13}
                                        : std::vector<float>{23, 19, 5, 17, 29, 13};
    } else {
        expected = reduction == "add"   ? std::vector<float>{21, 20, 5, 30, 40, 13}
                   : reduction == "mul" ? std::vector<float>{38, 51, 5, 161, 319, 13}
                                        : std::vector<float>{19, 17, 5, 23, 29, 13};
    }
    if (ok) {
        ok = runModel(mnnPath, expected);
    }
    std::remove(onnxPath.c_str());
    std::remove(mnnPath.c_str());
    std::printf("%s: %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    // Use a writable directory (the current directory by default) and isolate
    // generated models from other invocations of this test.
    const std::string directory = argc > 1 ? argv[1] : ".";
    const std::string prefix = directory + "/mnn_scatter_elements_" +
                               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                               "_";
    int passed = 0;
    int total = 0;
    for (int axis : {1, -1, 0}) {
        for (const auto& reduction : {"add", "mul", "none"}) {
            for (bool reductionFirst : {false, true}) {
                passed += runCase(prefix, axis, true, reduction, reductionFirst);
                ++total;
            }
        }
    }
    passed += runCase(prefix, 0, false, "", false);
    passed += runCase(prefix, 1, true, "", false);
    passed += runCase(prefix, 0, false, "add", true);
    passed += runCase(prefix, 0, false, "mul", true);
    total += 4;
    std::printf("ScatterElements conversion: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}
