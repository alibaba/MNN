#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "cli.hpp"
#include "onnx.pb.h"

using namespace MNN::Express;

struct SliceCase {
    std::string name;
    std::vector<int> shape;
    std::vector<int> starts;
    std::vector<int> ends;
    std::vector<int> steps;
    std::vector<int> axes;
    std::vector<int> outputShape;
    std::vector<float> expected;
};

static void addTensorShape(onnx::ValueInfoProto* value, const std::string& name, int type,
                           const std::vector<int>& dims) {
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    auto* shape = tensor->mutable_shape();
    for (size_t i = 0; i < dims.size(); ++i) {
        auto* dim = shape->add_dim();
        if (dims[i] < 0) {
            dim->set_dim_param(name + "_" + std::to_string(i));
        } else {
            dim->set_dim_value(dims[i]);
        }
    }
}

static void addIndices(onnx::GraphProto* graph, const std::string& name, const std::vector<int>& values) {
    auto* tensor = graph->add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(onnx::TensorProto_DataType_INT64);
    tensor->add_dims(values.size());
    for (int value : values) {
        tensor->add_int64_data(value);
    }
}

static onnx::ModelProto makeModel(const SliceCase& test, int opset, bool runtime, bool dynamicLength) {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(opset);
    auto* graph = model.mutable_graph();
    graph->set_name(test.name);
    addTensorShape(graph->add_input(), "data", onnx::TensorProto_DataType_FLOAT,
                   runtime ? std::vector<int>(test.shape.size(), -1) : test.shape);
    addTensorShape(graph->add_output(), "output", onnx::TensorProto_DataType_FLOAT,
                   std::vector<int>(test.shape.size(), -1));
    auto* node = graph->add_node();
    node->set_op_type("Slice");
    node->add_input("data");
    node->add_input("starts");
    node->add_input("ends");
    if (runtime) {
        const std::vector<int> shape = {dynamicLength ? -1 : static_cast<int>(test.starts.size())};
        for (const auto& name : {"starts", "ends", "steps"}) {
            addTensorShape(graph->add_input(), name, onnx::TensorProto_DataType_INT64, shape);
        }
    } else {
        addIndices(graph, "starts", test.starts);
        addIndices(graph, "ends", test.ends);
        if (!test.steps.empty()) {
            addIndices(graph, "steps", test.steps);
        }
    }
    if (!test.axes.empty() || !test.steps.empty()) {
        // A missing optional axes input must remain an empty serialized placeholder.
        node->add_input(test.axes.empty() ? "" : "axes");
    }
    if (!test.axes.empty()) {
        addIndices(graph, "axes", test.axes);
    }
    if (!test.steps.empty()) {
        node->add_input("steps");
    }
    node->add_output("output");
    return model;
}

static bool checkOutput(Module* module, const SliceCase& test, bool runtime) {
    const int size = std::accumulate(test.shape.begin(), test.shape.end(), 1, std::multiplies<int>());
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.0f);
    std::vector<VARP> inputs = {_Const(data.data(), test.shape, NCHW, halide_type_of<float>())};
    if (runtime) {
        for (const auto& indices : {test.starts, test.ends, test.steps}) {
            inputs.push_back(_Const(indices.data(), {static_cast<int>(indices.size())}, NCHW, halide_type_of<int>()));
        }
    }
    auto outputs = module->onForward(inputs);
    if (outputs.size() != 1 || nullptr == outputs[0]) {
        std::fprintf(stderr, "%s: missing output\n", test.name.c_str());
        return false;
    }
    const auto* info = outputs[0]->getInfo();
    if (!info || info->dim != test.outputShape || info->type != halide_type_of<float>() ||
        info->size != test.expected.size()) {
        std::fprintf(stderr, "%s: wrong output shape or type\n", test.name.c_str());
        return false;
    }
    const auto* got = outputs[0]->readMap<float>();
    if (!got) {
        return false;
    }
    for (size_t i = 0; i < test.expected.size(); ++i) {
        if (!std::isfinite(got[i]) || got[i] != test.expected[i]) {
            std::fprintf(stderr, "%s: mismatch at %zu: expected=%g, got=%g\n", test.name.c_str(), i, test.expected[i],
                         got[i]);
            return false;
        }
    }
    return true;
}

static bool runCase(const std::string& prefix, int opset, const std::vector<SliceCase>& cases, bool runtime = false,
                    bool dynamicLength = false) {
    const std::string name = cases[0].name + "_opset" + std::to_string(opset) +
                             (runtime ? (dynamicLength ? "_dynamic_length" : "_runtime") : "_constant");
    const std::string onnxPath = prefix + name + ".onnx";
    const std::string mnnPath = prefix + name + ".mnn";
    bool ok;
    {
        std::ofstream stream(onnxPath, std::ios::binary | std::ios::trunc);
        ok = makeModel(cases[0], opset, runtime, dynamicLength).SerializeToOstream(&stream);
    }
    if (ok) {
        modelConfig config;
        config.model = modelConfig::ONNX;
        config.modelFile = onnxPath;
        config.MNNModel = mnnPath;
        config.keepInputFormat = true;
        ok = MNN::Cli::convertModel(config);
    }
    if (ok) {
        MNN::BackendConfig backend;
        backend.precision = MNN::BackendConfig::Precision_High;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 1;
        config.backendConfig = &backend;
        std::shared_ptr<Executor::RuntimeManager> runtimeManager(Executor::RuntimeManager::createRuntimeManager(config),
                                                                 Executor::RuntimeManager::destroy);
        const std::vector<std::string> inputs =
            runtime ? std::vector<std::string>{"data", "starts", "ends", "steps"} : std::vector<std::string>{"data"};
        Module::Config moduleConfig;
        moduleConfig.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load(inputs, {"output"}, mnnPath.c_str(), runtimeManager, &moduleConfig),
                                       Module::destroy);
        ok = module != nullptr;
        if (ok) {
            // Reuse the same converted model for changes in values and index-vector length.
            for (const auto& test : cases) {
                ok = checkOutput(module.get(), test, runtime) && ok;
            }
        }
    }
    std::remove(onnxPath.c_str());
    std::remove(mnnPath.c_str());
    std::printf("%s: %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    const std::string directory = argc > 1 ? argv[1] : ".";
    const std::string prefix = directory + "/mnn_slice_" +
                               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                               "_";
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, backend, 1);
    const SliceCase positive = {"positive", {6}, {0}, {6}, {2}, {}, {3}, {0, 2, 4}};
    const SliceCase negative = {"negative", {6}, {5}, {-7}, {-2}, {}, {3}, {5, 3, 1}};
    const SliceCase multiple = {"multiple_axes", {3, 6}, {0, 5}, {3, -7}, {2, -2}, {}, {2, 3}, {5, 3, 1, 17, 15, 13}};
    const SliceCase leading = {
        "leading_axis", {3, 6}, {0}, {3}, {2}, {}, {2, 6}, {0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17}};
    const SliceCase resized = {"resized_data", {4, 4}, {1, 3}, {4, -5}, {2, -2}, {}, {2, 2}, {7, 5, 15, 13}};
    const SliceCase omittedSteps = {"omitted_steps", {6}, {1}, {5}, {}, {}, {4}, {1, 2, 3, 4}};
    int passed = 0;
    int total = 0;
    for (int opset : {10, 13}) {
        for (const auto& test : {positive, negative, multiple, leading, omittedSteps}) {
            passed += runCase(prefix, opset, {test});
            ++total;
            auto explicitAxes = test;
            explicitAxes.name += "_explicit_axes";
            explicitAxes.axes.resize(test.starts.size());
            std::iota(explicitAxes.axes.begin(), explicitAxes.axes.end(), 0);
            passed += runCase(prefix, opset, {explicitAxes});
            ++total;
        }
        for (int axis : {1, -1}) {
            const SliceCase lastAxis = {axis == 1 ? "last_axis_explicit_axes" : "negative_axis_explicit_axes",
                                        {2, 6},
                                        {0},
                                        {6},
                                        {2},
                                        {axis},
                                        {2, 3},
                                        {0, 2, 4, 6, 8, 10}};
            passed += runCase(prefix, opset, {lastAxis});
            ++total;
        }
        passed += runCase(prefix, opset, {positive, negative, positive}, true);
        ++total;
        passed += runCase(prefix, opset, {leading, multiple, resized, leading}, true, true);
        ++total;
    }
    std::printf("Slice conversion: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}
