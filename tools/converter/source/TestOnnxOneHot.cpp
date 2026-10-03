#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include "cli.hpp"
#include "onnx.pb.h"

static void addTensorInfo(onnx::ValueInfoProto* info, const char* name, int type, const std::vector<int>& shape) {
    info->set_name(name);
    auto* tensor = info->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    tensor->mutable_shape();
    for (int dim : shape) {
        tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    }
}

static bool runCase(int opset, int rank, bool hasAxis, int axis, const std::string& directory) {
    const std::string name = "onehot_v" + std::to_string(opset) + "_rank" + std::to_string(rank) + "_axis" +
                             (hasAxis ? std::to_string(axis) : "default");
    const std::string onnxPath = directory + "/" + name + ".onnx";
    const std::string mnnPath = directory + "/" + name + ".mnn";
    const std::vector<int> inputShape(rank, 2);
    const int depth = 3;
    const int outputAxis = !hasAxis || axis == -1 ? rank : axis;
    std::vector<int> outputShape = inputShape;
    outputShape.insert(outputShape.begin() + outputAxis, depth);

    onnx::ModelProto model;
    model.set_ir_version(7);
    model.add_opset_import()->set_version(opset);
    auto* graph = model.mutable_graph();
    graph->set_name(name);
    addTensorInfo(graph->add_input(), "X", onnx::TensorProto_DataType_INT64, inputShape);
    addTensorInfo(graph->add_output(), "Y", onnx::TensorProto_DataType_FLOAT, outputShape);
    auto* depthTensor = graph->add_initializer();
    depthTensor->set_name("depth");
    depthTensor->set_data_type(onnx::TensorProto_DataType_INT64);
    depthTensor->add_int64_data(depth);
    auto* values = graph->add_initializer();
    values->set_name("values");
    values->set_data_type(onnx::TensorProto_DataType_FLOAT);
    values->add_dims(2);
    values->add_float_data(-2.0f);
    values->add_float_data(5.0f);
    auto* node = graph->add_node();
    node->set_name(name);
    node->set_op_type("OneHot");
    node->add_input("X");
    node->add_input("depth");
    node->add_input("values");
    node->add_output("Y");
    if (hasAxis) {
        auto* attribute = node->add_attribute();
        attribute->set_name("axis");
        attribute->set_type(onnx::AttributeProto_AttributeType_INT);
        attribute->set_i(axis);
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
    // The converter normalizes ONNX int64 indices to MNN int32 tensors.
    if (input == nullptr || input->shape() != inputShape || input->getType() != halide_type_of<int>()) {
        return false;
    }
    const std::vector<int> first = rank == 1 ? std::vector<int>{0, 1} : std::vector<int>{0, 1, 2, 1};
    const std::vector<int> second = rank == 1 ? std::vector<int>{2, 0} : std::vector<int>{2, 0, 1, 2};
    int inside = 1;
    for (int i = outputAxis; i < rank; ++i) {
        inside *= inputShape[i];
    }
    // Reuse the converted session with changing indices and then restore the original input.
    for (const auto& indices : {first, second, first}) {
        MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
        ::memcpy(hostInput.host<int>(), indices.data(), indices.size() * sizeof(int));
        if (!input->copyFromHostTensor(&hostInput) || net->runSession(session) != MNN::NO_ERROR) {
            return false;
        }
        auto* output = net->getSessionOutput(session, "Y");
        if (output == nullptr || output->shape() != outputShape || output->getType() != halide_type_of<float>()) {
            std::fprintf(stderr, "%s: output shape or type mismatch\n", name.c_str());
            return false;
        }
        MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
        if (!output->copyToHostTensor(&hostOutput)) {
            return false;
        }
        const auto* actual = hostOutput.host<float>();
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            for (int c = 0; c < depth; ++c) {
                const int offset = (i / inside * depth + c) * inside + i % inside;
                const float expected = indices[i] == c ? 5.0f : -2.0f;
                if (actual[offset] != expected) {
                    std::fprintf(stderr, "%s[%d]: expected %g, got %g\n", name.c_str(), offset, expected,
                                 actual[offset]);
                    return false;
                }
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: TestOnnxOneHot EXISTING_OUTPUT_DIRECTORY\n");
        return 2;
    }
    int passed = 0;
    int failed = 0;
    for (int opset : {9, 11}) {
        for (int rank : {1, 2}) {
            for (int caseIndex = 0; caseIndex < 4; ++caseIndex) {
                const bool hasAxis = caseIndex != 0;
                const int axis = caseIndex - 2;
                const bool success = runCase(opset, rank, hasAxis, axis, argv[1]);
                std::printf("%s opset=%d rank=%d axis=%s\n", success ? "PASS" : "FAIL", opset, rank,
                            hasAxis ? std::to_string(axis).c_str() : "default");
                passed += success;
                failed += !success;
            }
        }
    }
    std::printf("OneHot cases: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
