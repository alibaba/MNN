#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include "MNN_generated.h"
#include "cli.hpp"
#include "onnx.pb.h"

static void addTensorShape(onnx::ValueInfoProto* valueInfo, const std::string& name, const std::vector<int>& dims) {
    valueInfo->set_name(name);
    auto* tensorType = valueInfo->mutable_type()->mutable_tensor_type();
    tensorType->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    auto* shape = tensorType->mutable_shape();
    for (int dim : dims) {
        shape->add_dim()->set_dim_value(dim);
    }
}

static void addFloatInitializer(onnx::GraphProto* graph, const std::string& name, const std::vector<int64_t>& dims,
                                const std::vector<float>& values) {
    auto* tensor = graph->add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(onnx::TensorProto_DataType_FLOAT);
    for (auto dim : dims) {
        tensor->add_dims(dim);
    }
    for (auto value : values) {
        tensor->add_float_data(value);
    }
}

static void addInt64Initializer(onnx::GraphProto* graph, const std::string& name, const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& values) {
    auto* tensor = graph->add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(onnx::TensorProto_DataType_INT64);
    for (auto dim : dims) {
        tensor->add_dims(dim);
    }
    for (auto value : values) {
        tensor->add_int64_data(value);
    }
}

static bool saveModel(const onnx::ModelProto& model, const std::string& fileName) {
    std::ofstream output(fileName, std::ios::binary | std::ios::trunc);
    return model.SerializeToOstream(&output);
}

static bool compareVector(const float* got, const std::vector<float>& expected, float tolerance = 1e-5f) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "mismatch at %zu, expect=%f, got=%f\n", i, expected[i], got[i]);
            return false;
        }
    }
    return true;
}

static bool runMNNModel(const std::string& modelPath, const std::vector<std::pair<std::string, std::vector<float>>>& inputs,
                        const std::vector<float>& expectedOutput) {
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!net) {
        return false;
    }
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    if (!session) {
        return false;
    }
    for (const auto& item : inputs) {
        auto* inputTensor = net->getSessionInput(session, item.first.c_str());
        if (!inputTensor) {
            return false;
        }
        MNN::Tensor hostTensor(inputTensor, inputTensor->getDimensionType());
        ::memcpy(hostTensor.host<float>(), item.second.data(), item.second.size() * sizeof(float));
        inputTensor->copyFromHostTensor(&hostTensor);
    }
    if (net->runSession(session) != MNN::NO_ERROR) {
        return false;
    }
    auto* outputTensor = net->getSessionOutput(session, nullptr);
    if (!outputTensor) {
        return false;
    }
    MNN::Tensor hostOutput(outputTensor, outputTensor->getDimensionType());
    outputTensor->copyToHostTensor(&hostOutput);
    return compareVector(hostOutput.host<float>(), expectedOutput);
}

static bool convertOnnx(const std::string& onnxModel, const std::string& mnnModel) {
    modelConfig config;
    config.model = modelConfig::ONNX;
    config.modelFile = onnxModel;
    config.MNNModel = mnnModel;
    config.keepInputFormat = true;
    return MNN::Cli::convertModel(config);
}

static onnx::ModelProto makeConcatEinsumModel() {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.mutable_opset_import()->Add()->set_version(13);
    auto* graph = model.mutable_graph();
    graph->set_name("ConcatEinsum");

    addTensorShape(graph->add_input(), "x", {2, 3});
    addTensorShape(graph->add_input(), "y", {2, 3});
    addTensorShape(graph->add_output(), "out", {2, 3});

    addFloatInitializer(graph, "weight", {2}, {1.5f, -0.5f});
    addInt64Initializer(graph, "axes", {1}, {0});

    auto* unsqueezeX = graph->add_node();
    unsqueezeX->set_op_type("Unsqueeze");
    unsqueezeX->add_input("x");
    unsqueezeX->add_input("axes");
    unsqueezeX->add_output("x_unsqueezed");

    auto* unsqueezeY = graph->add_node();
    unsqueezeY->set_op_type("Unsqueeze");
    unsqueezeY->add_input("y");
    unsqueezeY->add_input("axes");
    unsqueezeY->add_output("y_unsqueezed");

    auto* concat = graph->add_node();
    concat->set_op_type("Concat");
    concat->add_input("x_unsqueezed");
    concat->add_input("y_unsqueezed");
    concat->add_output("stacked");
    auto* axis = concat->add_attribute();
    axis->set_name("axis");
    axis->set_i(0);

    auto* einsum = graph->add_node();
    einsum->set_op_type("Einsum");
    einsum->add_input("weight");
    einsum->add_input("stacked");
    einsum->add_output("out");
    auto* equation = einsum->add_attribute();
    equation->set_name("equation");
    equation->set_s("i,i...->...");
    return model;
}

static onnx::ModelProto makeReduceEinsumModel() {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.mutable_opset_import()->Add()->set_version(13);
    auto* graph = model.mutable_graph();
    graph->set_name("ReduceEinsum");

    addTensorShape(graph->add_input(), "stacked", {2, 2, 3});
    addTensorShape(graph->add_output(), "out", {2, 3});
    addFloatInitializer(graph, "weight", {2}, {1.5f, -0.5f});

    auto* einsum = graph->add_node();
    einsum->set_op_type("Einsum");
    einsum->add_input("weight");
    einsum->add_input("stacked");
    einsum->add_output("out");
    auto* equation = einsum->add_attribute();
    equation->set_name("equation");
    equation->set_s("i,i...->...");
    return model;
}

static onnx::ModelProto makeConcat3EinsumModel() {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.mutable_opset_import()->Add()->set_version(13);
    auto* graph = model.mutable_graph();
    graph->set_name("Concat3Einsum");

    addTensorShape(graph->add_input(), "x", {2, 3});
    addTensorShape(graph->add_input(), "y", {2, 3});
    addTensorShape(graph->add_input(), "z", {2, 3});
    addTensorShape(graph->add_output(), "out", {2, 3});

    addFloatInitializer(graph, "weight", {3}, {1.0f, -2.0f, 0.5f});
    addInt64Initializer(graph, "axes", {1}, {0});

    auto* unsqueezeX = graph->add_node();
    unsqueezeX->set_op_type("Unsqueeze");
    unsqueezeX->add_input("x");
    unsqueezeX->add_input("axes");
    unsqueezeX->add_output("x_unsqueezed");

    auto* unsqueezeY = graph->add_node();
    unsqueezeY->set_op_type("Unsqueeze");
    unsqueezeY->add_input("y");
    unsqueezeY->add_input("axes");
    unsqueezeY->add_output("y_unsqueezed");

    auto* unsqueezeZ = graph->add_node();
    unsqueezeZ->set_op_type("Unsqueeze");
    unsqueezeZ->add_input("z");
    unsqueezeZ->add_input("axes");
    unsqueezeZ->add_output("z_unsqueezed");

    auto* concat = graph->add_node();
    concat->set_op_type("Concat");
    concat->add_input("x_unsqueezed");
    concat->add_input("y_unsqueezed");
    concat->add_input("z_unsqueezed");
    concat->add_output("stacked");
    auto* axis = concat->add_attribute();
    axis->set_name("axis");
    axis->set_i(0);

    auto* einsum = graph->add_node();
    einsum->set_op_type("Einsum");
    einsum->add_input("weight");
    einsum->add_input("stacked");
    einsum->add_output("out");
    auto* equation = einsum->add_attribute();
    equation->set_name("equation");
    equation->set_s("i,i...->...");
    return model;
}

int main() {
    const std::string concatOnnx = "/tmp/mnn_concat_einsum.onnx";
    const std::string concatMnn = "/tmp/mnn_concat_einsum.mnn";
    const std::string reduceOnnx = "/tmp/mnn_reduce_einsum.onnx";
    const std::string reduceMnn = "/tmp/mnn_reduce_einsum.mnn";
    const std::string concat3Onnx = "/tmp/mnn_concat3_einsum.onnx";
    const std::string concat3Mnn = "/tmp/mnn_concat3_einsum.mnn";

    bool ok = saveModel(makeConcatEinsumModel(), concatOnnx);
    ok = saveModel(makeReduceEinsumModel(), reduceOnnx) && ok;
    ok = saveModel(makeConcat3EinsumModel(), concat3Onnx) && ok;

    ok = convertOnnx(concatOnnx, concatMnn) && ok;
    ok = convertOnnx(reduceOnnx, reduceMnn) && ok;
    ok = convertOnnx(concat3Onnx, concat3Mnn) && ok;

    const std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> y = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const std::vector<float> expected = {-1.5f, 0.5f, 2.5f, 4.5f, 6.5f, 8.5f};
    ok = runMNNModel(concatMnn, {{"x", x}, {"y", y}}, expected) && ok;

    const std::vector<float> stacked = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };
    ok = runMNNModel(reduceMnn, {{"stacked", stacked}}, expected) && ok;

    const std::vector<float> z = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    const std::vector<float> expected3 = {-11.0f, -7.5f, -5.0f, -1.5f, 1.0f, 4.5f};
    ok = runMNNModel(concat3Mnn, {{"x", x}, {"y", y}, {"z", z}}, expected3) && ok;

    ::remove(concatOnnx.c_str());
    ::remove(concatMnn.c_str());
    ::remove(reduceOnnx.c_str());
    ::remove(reduceMnn.c_str());
    ::remove(concat3Onnx.c_str());
    ::remove(concat3Mnn.c_str());
    return ok ? 0 : 1;
}
