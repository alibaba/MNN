#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "cli.hpp"
#include "onnx.pb.h"

using namespace MNN::Express;

static void addTensorInfo(onnx::ValueInfoProto* info, const char* name, const std::vector<int>& shape,
                          int type = onnx::TensorProto_DataType_FLOAT) {
    info->set_name(name);
    auto* tensor = info->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    tensor->mutable_shape();
    for (int dim : shape) {
        tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    }
}

static std::vector<float> referenceSoftmax(const std::vector<float>& input, const std::vector<int>& shape, int opset,
                                           int axis) {
    if (axis < 0) {
        axis += static_cast<int>(shape.size());
    }
    int outside = 1;
    int width = shape[axis];
    int inside = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= shape[i];
    }
    for (int i = axis + 1; i < static_cast<int>(shape.size()); ++i) {
        inside *= shape[i];
    }
    // Before opset 13, all dimensions from axis onward form one feature dimension.
    if (opset < 13) {
        width *= inside;
        inside = 1;
    }
    std::vector<float> output(input.size());
    for (int o = 0; o < outside; ++o) {
        for (int i = 0; i < inside; ++i) {
            const int base = o * width * inside + i;
            double maximum = input[base];
            for (int w = 1; w < width; ++w) {
                maximum = std::max(maximum, static_cast<double>(input[base + w * inside]));
            }
            double sum = 0.0;
            for (int w = 0; w < width; ++w) {
                sum += std::exp(static_cast<double>(input[base + w * inside]) - maximum);
            }
            for (int w = 0; w < width; ++w) {
                output[base + w * inside] = std::exp(static_cast<double>(input[base + w * inside]) - maximum) / sum;
            }
        }
    }
    return output;
}

static void addSoftmax(onnx::GraphProto* graph, const char* output, bool hasAxis, int axis) {
    auto* node = graph->add_node();
    node->set_op_type("Softmax");
    node->add_input("X");
    node->add_output(output);
    if (hasAxis) {
        auto* attribute = node->add_attribute();
        attribute->set_name("axis");
        attribute->set_type(onnx::AttributeProto_AttributeType_INT);
        attribute->set_i(axis);
    }
}

static bool runCase(int opset, int rank, bool hasAxis, int axis, const std::string& directory, int customVersion = 0,
                    bool customFirst = false, bool inIf = false, int aliasVersion = 0, bool aliasOnly = false,
                    bool aliasFirst = false) {
    std::string name = "softmax_v" + std::to_string(opset) + "_rank" + std::to_string(rank) + "_axis" +
                       (hasAxis ? std::to_string(axis) : "default");
    if (customVersion != 0) {
        name += "_custom" + std::to_string(customVersion) + (customFirst ? "_first" : "_last");
    }
    if (inIf) {
        name += "_if";
    }
    if (aliasVersion != 0) {
        name += "_alias" + std::to_string(aliasVersion) + (aliasOnly ? "_only" : (aliasFirst ? "_first" : "_last"));
    }
    const std::string onnxPath = directory + "/" + name + ".onnx";
    const std::string mnnPath = directory + "/" + name + ".mnn";
    std::vector<int> shape(rank, 2);
    shape.back() = 3;
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    onnx::ModelProto model;
    model.set_ir_version(7);
    if (customVersion != 0 && customFirst) {
        auto* custom = model.add_opset_import();
        custom->set_domain("com.example");
        custom->set_version(customVersion);
    }
    if (aliasVersion != 0 && (aliasOnly || aliasFirst)) {
        auto* alias = model.add_opset_import();
        alias->set_domain("ai.onnx");
        alias->set_version(aliasVersion);
    }
    if (!aliasOnly) {
        model.add_opset_import()->set_version(opset);
    }
    if (aliasVersion != 0 && !aliasOnly && !aliasFirst) {
        auto* alias = model.add_opset_import();
        alias->set_domain("ai.onnx");
        alias->set_version(aliasVersion);
    }
    if (customVersion != 0 && !customFirst) {
        auto* custom = model.add_opset_import();
        custom->set_domain("com.example");
        custom->set_version(customVersion);
    }
    auto* graph = model.mutable_graph();
    graph->set_name(name);
    addTensorInfo(graph->add_input(), "X", shape);
    addTensorInfo(graph->add_output(), "Y", shape);
    if (inIf) {
        addTensorInfo(graph->add_input(), "condition", {}, onnx::TensorProto_DataType_BOOL);
        auto* node = graph->add_node();
        node->set_op_type("If");
        node->add_input("condition");
        node->add_output("Y");
        for (const char* branch : {"then_branch", "else_branch"}) {
            auto* attr = node->add_attribute();
            attr->set_name(branch);
            attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
            auto* subgraph = attr->mutable_g();
            subgraph->set_name(name + branch);
            addTensorInfo(subgraph->add_output(), "B", shape);
            addSoftmax(subgraph, "B", hasAxis, axis);
        }
    } else {
        addSoftmax(graph, "Y", hasAxis, axis);
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
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig = &backend;
    std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(config));
    if (!runtime) {
        return false;
    }
    std::vector<std::string> inputNames = {"X"};
    if (inIf) {
        inputNames.push_back("condition");
    }
    std::shared_ptr<Module> net(Module::load(inputNames, {"Y"}, mnnPath.c_str(), runtime));
    if (!net) {
        return false;
    }
    auto input = _Input(shape, NCHW, halide_type_of<float>());
    const int effectiveAxis = hasAxis ? axis : (opset < 13 ? 1 : -1);
    // Change values, then restore the original input on the same converted module.
    for (int step = 0; step < 3; ++step) {
        std::vector<float> values(size);
        for (int i = 0; i < size; ++i) {
            values[i] = step == 1 ? (i * 3 + 1) % 11 - 5 : i % 7 - 3;
        }
        auto* inputData = input->writeMap<float>();
        if (inputData == nullptr) {
            return false;
        }
        ::memcpy(inputData, values.data(), values.size() * sizeof(float));
        input->unMap();
        std::vector<VARP> inputs = {input};
        if (inIf) {
            // ONNX bool inputs are normalized to MNN int32.
            inputs.push_back(_Scalar<int>(step == 1 ? 0 : 1));
        }
        auto outputs = net->onForward(inputs);
        if (outputs.size() != 1 || outputs[0].get() == nullptr) {
            return false;
        }
        auto output = _Convert(outputs[0], NCHW);
        const auto* info = output->getInfo();
        if (info == nullptr || info->dim != shape || info->type != halide_type_of<float>()) {
            std::fprintf(stderr, "%s: output shape or type mismatch\n", name.c_str());
            return false;
        }
        const auto* actual = output->readMap<float>();
        if (actual == nullptr) {
            return false;
        }
        const auto expected = referenceSoftmax(values, shape, opset, effectiveAxis);
        for (int i = 0; i < size; ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-6f) {
                std::fprintf(stderr, "%s step=%d[%d]: expected %g, got %g\n", name.c_str(), step, i, expected[i],
                             actual[i]);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: TestOnnxSoftmax EXISTING_OUTPUT_DIRECTORY\n");
        return 2;
    }
    int passed = 0;
    int failed = 0;
    for (int opset : {9, 11, 12, 13, 18}) {
        for (int rank : {1, 2, 3}) {
            for (int caseIndex = 0; caseIndex < 4; ++caseIndex) {
                const bool hasAxis = caseIndex != 0;
                const int axis = caseIndex - 2;
                // Legacy rank 1 needs an explicit valid axis; negative axes start at opset 11.
                if ((rank == 1 && ((!hasAxis && opset < 13) || (hasAxis && axis == 1))) ||
                    (hasAxis && axis < 0 && opset < 11)) {
                    continue;
                }
                const bool success = runCase(opset, rank, hasAxis, axis, argv[1]);
                std::printf("%s opset=%d rank=%d axis=%s\n", success ? "PASS" : "FAIL", opset, rank,
                            hasAxis ? std::to_string(axis).c_str() : "default");
                passed += success;
                failed += !success;
            }
        }
    }
    for (int opset : {9, 11, 12, 13, 18}) {
        for (int caseIndex = 0; caseIndex < 3; ++caseIndex) {
            const bool hasAxis = caseIndex != 0;
            const int axis = caseIndex == 1 ? 1 : -1;
            if (hasAxis && axis < 0 && opset < 11) {
                continue;
            }
            for (int customVersion : {1, 99}) {
                for (bool customFirst : {true, false}) {
                    const bool success = runCase(opset, 3, hasAxis, axis, argv[1], customVersion, customFirst);
                    std::printf("%s opset=%d custom=%d first=%d case=%d\n", success ? "PASS" : "FAIL", opset,
                                customVersion, customFirst, caseIndex);
                    passed += success;
                    failed += !success;
                }
            }
        }
    }
    for (int opset : {11, 13, 18}) {
        for (int caseIndex = 0; caseIndex < 3; ++caseIndex) {
            const bool hasAxis = caseIndex != 0;
            const int axis = caseIndex == 1 ? 1 : -1;
            const bool success = runCase(opset, 3, hasAxis, axis, argv[1], 0, false, true);
            std::printf("%s If opset=%d case=%d\n", success ? "PASS" : "FAIL", opset, caseIndex);
            passed += success;
            failed += !success;
        }
    }
    for (int opset : {9, 11, 12, 13, 18}) {
        for (int caseIndex = 0; caseIndex < 3; ++caseIndex) {
            const bool hasAxis = caseIndex != 0;
            const int axis = caseIndex == 1 ? 1 : -1;
            if (hasAxis && axis < 0 && opset < 11) {
                continue;
            }
            const bool aliasOnly = runCase(opset, 3, hasAxis, axis, argv[1], 0, false, false, opset, true);
            std::printf("%s alias-only opset=%d case=%d\n", aliasOnly ? "PASS" : "FAIL", opset, caseIndex);
            passed += aliasOnly;
            failed += !aliasOnly;
            // Empty-domain imports take precedence, even with a conflicting standard alias.
            const int otherVersion = opset < 13 ? 18 : 11;
            for (bool aliasFirst : {true, false}) {
                const bool success =
                    runCase(opset, 3, hasAxis, axis, argv[1], 0, false, false, otherVersion, false, aliasFirst);
                std::printf("%s standard-precedence opset=%d alias=%d first=%d case=%d\n", success ? "PASS" : "FAIL",
                            opset, otherVersion, aliasFirst, caseIndex);
                passed += success;
                failed += !success;
            }
        }
    }
    std::printf("Softmax cases: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
