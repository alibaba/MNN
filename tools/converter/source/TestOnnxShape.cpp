#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "cli.hpp"
#include "onnx.pb.h"

using namespace MNN::Express;

struct ShapeCase {
    std::string name;
    bool hasStart;
    int64_t start;
    bool hasEnd;
    int64_t end;
    std::vector<int> expectedAxes;
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

static onnx::ModelProto makeModel(const ShapeCase& test, const std::vector<int>& inputShape) {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(15);
    auto* graph = model.mutable_graph();
    graph->set_name(test.name);
    addTensorShape(graph->add_input(), "data", onnx::TensorProto_DataType_FLOAT, inputShape);
    addTensorShape(graph->add_output(), "output", onnx::TensorProto_DataType_INT64,
                   {static_cast<int>(test.expectedAxes.size())});
    auto* node = graph->add_node();
    node->set_op_type("Shape");
    node->add_input("data");
    node->add_output("output");
    if (test.hasStart) {
        auto* attr = node->add_attribute();
        attr->set_name("start");
        attr->set_type(onnx::AttributeProto_AttributeType_INT);
        attr->set_i(test.start);
    }
    if (test.hasEnd) {
        auto* attr = node->add_attribute();
        attr->set_name("end");
        attr->set_type(onnx::AttributeProto_AttributeType_INT);
        attr->set_i(test.end);
    }
    return model;
}

static bool checkOutput(Module* module, const ShapeCase& test, const std::vector<int>& inputShape) {
    // Shape only needs the input metadata, including when the input is a scalar.
    auto outputs = module->onForward({_Input(inputShape, NCHW, halide_type_of<float>())});
    if (outputs.size() != 1 || nullptr == outputs[0]) {
        std::fprintf(stderr, "%s: missing output\n", test.name.c_str());
        return false;
    }
    const auto* info = outputs[0]->getInfo();
    const std::vector<int> expectedShape = {static_cast<int>(test.expectedAxes.size())};
    // Preserve MNN's existing int32 Shape output convention.
    if (!info || info->dim != expectedShape || info->type != halide_type_of<int>() ||
        info->size != test.expectedAxes.size()) {
        std::fprintf(stderr, "%s: wrong output shape or type\n", test.name.c_str());
        return false;
    }
    if (test.expectedAxes.empty()) {
        return true;
    }
    const auto* got = outputs[0]->readMap<int>();
    if (!got) {
        std::fprintf(stderr, "%s: missing output data\n", test.name.c_str());
        return false;
    }
    for (size_t i = 0; i < test.expectedAxes.size(); ++i) {
        const int expected = inputShape[test.expectedAxes[i]];
        if (got[i] != expected) {
            std::fprintf(stderr, "%s: mismatch at %zu: expected=%d, got=%d\n", test.name.c_str(), i, expected, got[i]);
            return false;
        }
    }
    return true;
}

static bool runCase(const std::string& prefix, const ShapeCase& test, bool symbolic, bool scalar = false) {
    const std::string name = test.name + (scalar ? "_scalar" : (symbolic ? "_symbolic" : "_static"));
    const std::string onnxPath = prefix + name + ".onnx";
    const std::string mnnPath = prefix + name + ".mnn";
    const std::vector<int> inputShape = scalar ? std::vector<int>{} : std::vector<int>{2, 3, 4};
    bool ok;
    {
        std::ofstream stream(onnxPath, std::ios::binary | std::ios::trunc);
        ok = makeModel(test, symbolic ? std::vector<int>(inputShape.size(), -1) : inputShape)
                 .SerializeToOstream(&stream);
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
        std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(config),
                                                          Executor::RuntimeManager::destroy);
        Module::Config moduleConfig;
        moduleConfig.shapeMutable = true;
        std::shared_ptr<Module> module(Module::load({"data"}, {"output"}, mnnPath.c_str(), runtime, &moduleConfig),
                                       Module::destroy);
        ok = module != nullptr;
        if (ok) {
            ok = checkOutput(module.get(), test, inputShape);
            if (symbolic && !scalar) {
                // Reuse the converted model to verify shape values after a resize.
                ok = checkOutput(module.get(), test, {4, 5, 6}) && ok;
                ok = checkOutput(module.get(), test, inputShape) && ok;
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
    const std::string prefix = directory + "/mnn_shape_" +
                               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                               "_";
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, backend, 1);
    const int64_t min32 = std::numeric_limits<int32_t>::min();
    const int64_t max32 = std::numeric_limits<int32_t>::max();
    const int64_t min64 = std::numeric_limits<int64_t>::min();
    const int64_t max64 = std::numeric_limits<int64_t>::max();
    const std::vector<ShapeCase> cases = {
        {"default", false, 0, false, 0, {0, 1, 2}},
        {"positive", true, 1, true, 2, {1}},
        {"negative", true, -2, true, -1, {1}},
        {"negative_start", true, -1, false, 0, {2}},
        {"negative_end", false, 0, true, -1, {0, 1}},
        {"negative_clamped", true, -4, true, 4, {0, 1, 2}},
        {"reversed", true, 2, true, 1, {}},
        {"zero_end", false, 0, true, 0, {}},
        {"start_at_rank", true, 3, false, 0, {}},
        {"start_min32", true, min32, false, 0, {0, 1, 2}},
        {"end_min32", false, 0, true, min32, {}},
        {"start_max32", true, max32, false, 0, {}},
        {"end_max32", false, 0, true, max32, {0, 1, 2}},
        {"start_below_min32", true, min32 - 1, false, 0, {0, 1, 2}},
        {"end_below_min32", false, 0, true, min32 - 1, {}},
        {"start_above_max32", true, max32 + 1, false, 0, {}},
        {"end_above_max32", false, 0, true, max32 + 1, {0, 1, 2}},
        {"start_min64", true, min64, false, 0, {0, 1, 2}},
        {"end_min64", false, 0, true, min64, {}},
        {"start_max64", true, max64, false, 0, {}},
        {"end_max64", false, 0, true, max64, {0, 1, 2}},
        {"full_extremes", true, min64, true, max64, {0, 1, 2}},
        {"reversed_extremes", true, max64, true, min64, {}},
        {"start_wraps_positive", true, INT64_C(4294967297), false, 0, {}},
        {"end_wraps_positive", false, 0, true, INT64_C(4294967297), {0, 1, 2}},
        {"start_wraps_negative", true, INT64_C(-4294967295), false, 0, {0, 1, 2}},
        {"end_wraps_negative", false, 0, true, INT64_C(-4294967295), {}},
    };
    int passed = 0;
    int total = 0;
    for (const auto& test : cases) {
        for (bool symbolic : {false, true}) {
            passed += runCase(prefix, test, symbolic);
            ++total;
        }
        auto scalar = test;
        scalar.expectedAxes.clear();
        passed += runCase(prefix, scalar, false, true);
        ++total;
    }
    std::printf("Shape conversion: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}
