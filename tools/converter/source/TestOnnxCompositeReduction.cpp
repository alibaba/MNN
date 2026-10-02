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
#include "MNN_generated.h"
#include "onnx.pb.h"
#include "optimizer/onnxextra/OnnxExtraManager.hpp"

struct ReduceCase {
    const char* name;
    int version;
    bool hasAxes;
    std::vector<int> axes;
    int keepDims; // -1 omits the attribute, whose ONNX default is 1.
    bool noop;
    bool runtimeAxes;
};

static void addTensor(onnx::ValueInfoProto* value, const std::string& name, int type, const std::vector<int>& dims) {
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    auto* shape = tensor->mutable_shape();
    for (int dim : dims) {
        if (dim < 0) {
            shape->add_dim()->set_dim_param("axes_length");
        } else {
            shape->add_dim()->set_dim_value(dim);
        }
    }
}

static void addInt(onnx::NodeProto* node, const std::string& name, int value) {
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto_AttributeType_INT);
    attr->set_i(value);
}

static onnx::ModelProto makeModel(const std::string& type, const ReduceCase& test,
                                  const std::vector<int>& outputShape) {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(test.version);
    auto* graph = model.mutable_graph();
    graph->set_name("CompositeReduction");
    addTensor(graph->add_input(), "data", onnx::TensorProto_DataType_FLOAT, {2, 2});
    addTensor(graph->add_output(), "output", onnx::TensorProto_DataType_FLOAT, outputShape);
    auto* node = graph->add_node();
    node->set_op_type(type);
    node->add_input("data");
    node->add_output("output");
    if (test.keepDims >= 0) {
        addInt(node, "keepdims", test.keepDims);
    }
    if (test.noop) {
        addInt(node, "noop_with_empty_axes", 1);
    }
    if (test.hasAxes) {
        if (test.version < 18) {
            auto* attr = node->add_attribute();
            attr->set_name("axes");
            attr->set_type(onnx::AttributeProto_AttributeType_INTS);
            for (int axis : test.axes) {
                attr->add_ints(axis);
            }
        } else {
            node->add_input("axes");
            if (test.runtimeAxes) {
                addTensor(graph->add_input(), "axes", onnx::TensorProto_DataType_INT64,
                          {std::string(test.name) == "unknown_axes" ? -1 : static_cast<int>(test.axes.size())});
            } else {
                onnx::TensorProto* axes;
                if (std::string(test.name).find("constant_node") == 0) {
                    auto* constant = graph->add_node();
                    constant->set_op_type("Constant");
                    constant->add_output("axes");
                    auto* attr = constant->add_attribute();
                    attr->set_name("value");
                    attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
                    axes = attr->mutable_t();
                    graph->mutable_node()->SwapElements(0, 1);
                } else {
                    axes = graph->add_initializer();
                }
                axes->set_name("axes");
                axes->set_data_type(onnx::TensorProto_DataType_INT64);
                axes->add_dims(test.axes.size());
                for (int axis : test.axes) {
                    axes->add_int64_data(axis);
                }
            }
        }
    }
    return model;
}

static void expectedResult(const std::string& type, const ReduceCase& test, std::vector<float>& data,
                           std::vector<int>& shape, std::vector<float>& expected) {
    data = {3, 4, 5, 12};
    if (type == "ReduceL1" || type == "ReduceL2" || type == "ReduceSumSquare") {
        data[0] = -3; // Empty-axis no-op must still perform Abs or Square.
    }
    bool reduce[2] = {false, false};
    if (test.axes.empty()) {
        reduce[0] = reduce[1] = !test.noop;
    } else {
        for (int axis : test.axes) {
            reduce[axis < 0 ? axis + 2 : axis] = true;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (!reduce[i]) {
            shape.push_back(2);
        } else if (test.keepDims != 0) {
            shape.push_back(1);
        }
    }
    expected.resize((reduce[0] ? 1 : 2) * (reduce[1] ? 1 : 2), 0.0f);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            float value = data[row * 2 + col];
            if (type == "ReduceL1") {
                value = std::fabs(value);
            } else if (type == "ReduceL2" || type == "ReduceSumSquare") {
                value *= value;
            } else if (type == "ReduceLogSumExp") {
                value = std::exp(value);
            }
            const int index = (reduce[0] ? 0 : row) * (reduce[1] ? 1 : 2) + (reduce[1] ? 0 : col);
            expected[index] += value;
        }
    }
    for (float& value : expected) {
        if (type == "ReduceL2") {
            value = std::sqrt(value);
        } else if (type == "ReduceLogSum" || type == "ReduceLogSumExp") {
            value = std::log(value);
        }
    }
}

static bool runModel(const std::string& path, const ReduceCase& test, const std::vector<float>& data,
                     const std::vector<int>& shape, const std::vector<float>& expected) {
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(path.c_str()));
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
    if (!input || input->shape() != std::vector<int>({2, 2}) || input->getType() != halide_type_of<float>()) {
        return false;
    }
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    std::memcpy(hostInput.host<float>(), data.data(), data.size() * sizeof(float));
    if (!input->copyFromHostTensor(&hostInput) || net->runSession(session) != MNN::NO_ERROR) {
        return false;
    }
    auto* output = net->getSessionOutput(session, "output");
    if (!output || output->shape() != shape || output->getType() != halide_type_of<float>()) {
        std::fprintf(stderr, "output shape/type mismatch\n");
        return false;
    }
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    if (!output->copyToHostTensor(&hostOutput)) {
        return false;
    }
    const float* got = hostOutput.host<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(got[i]) ||
            std::fabs(got[i] - expected[i]) > 1e-5f * std::fmax(1.0f, std::fabs(expected[i]))) {
            std::fprintf(stderr, "mismatch at %zu: expected=%g, got=%g\n", i, expected[i], got[i]);
            return false;
        }
    }
    return true;
}

static bool rejectsNonconstantAxes(const std::string& type, const ReduceCase& test) {
    using namespace MNN::Express;
    MNN::OpT op;
    op.type = MNN::OpType_Extra;
    op.name = "nonconstant_axes";
    op.main.type = MNN::OpParameter_Extra;
    op.main.value = new MNN::ExtraT;
    auto* extra = op.main.AsExtra();
    extra->engine = "ONNX";
    extra->type = type;
    std::unique_ptr<MNN::AttributeT> keepDims(new MNN::AttributeT);
    keepDims->key = "keepdims";
    keepDims->i = test.keepDims != 0;
    extra->attr.emplace_back(std::move(keepDims));
    const int size = std::string(test.name) == "unknown_axes" ? -1 : static_cast<int>(test.axes.size());
    auto data = _Input({2, 2}, NCHW, halide_type_of<float>());
    auto axes = _Input({size}, NCHW, halide_type_of<int>());
    const auto transform = OnnxExtraManager::get()->find(type);
    // This checks the exact transform stage, so a missing output model caused
    // by an unrelated converter failure cannot satisfy the rejection test.
    return transform && nullptr == transform->onExecute(Expr::create(&op, {data, axes}));
}

int main(int argc, char** argv) {
    const std::string directory = argc > 1 ? argv[1] : ".";
    const std::string prefix = directory + "/mnn_composite_reduce_" +
                               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                               "_";
    const std::vector<ReduceCase> cases = {
        {"input_axis1", 18, true, {1}, 1, false, false},
        {"input_negative", 18, true, {-1}, 0, false, false},
        {"input_multi", 18, true, {0, 1}, 1, false, false},
        {"input_default_keep", 18, true, {1}, -1, false, false},
        {"legacy_default_keep", 13, true, {1}, -1, false, false},
        {"legacy_squeeze", 13, true, {1}, 0, false, false},
        {"legacy_keep", 13, true, {0}, 1, false, false},
        {"omitted_default", 18, false, {}, -1, false, false},
        {"empty_default", 18, true, {}, 1, false, false},
        {"empty_noop", 18, true, {}, 0, true, false},
        {"omitted_noop", 18, false, {}, -1, true, false},
        {"nonempty_noop", 18, true, {1}, 1, true, false},
        {"omitted_squeeze", 18, false, {}, 0, false, false},
        {"runtime_empty_default", 18, true, {}, 1, false, true},
        {"runtime_empty_noop", 18, true, {}, 0, true, true},
        {"constant_node", 18, true, {1}, 1, false, false},
        {"constant_node_empty", 18, true, {}, 0, true, false},
        {"runtime_all", 18, true, {0, 1}, 1, false, true},
        {"unknown_axes", 18, true, {1}, 1, false, true},
        {"runtime_axis1", 18, true, {1}, 1, false, true},
        {"runtime_negative", 18, true, {-1}, 0, false, true},
    };
    int passed = 0;
    int total = 0;
    int rejected = 0;
    int rejectedTotal = 0;
    for (const std::string type : {"ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"}) {
        for (const auto& test : cases) {
            const std::string name = type + "_" + test.name;
            const std::string onnxPath = prefix + name + ".onnx";
            const std::string mnnPath = prefix + name + ".mnn";
            std::vector<float> data, expected;
            std::vector<int> shape;
            expectedResult(type, test, data, shape, expected);
            bool ok;
            {
                std::ofstream stream(onnxPath, std::ios::binary | std::ios::trunc);
                ok = makeModel(type, test, shape).SerializeToOstream(&stream);
            }
            if (ok) {
                modelConfig config;
                config.model = modelConfig::ONNX;
                config.modelFile = onnxPath;
                config.MNNModel = mnnPath;
                config.keepInputFormat = true;
                const bool converted = MNN::Cli::convertModel(config);
                if (test.runtimeAxes) {
                    // Input axes used to be silently ignored. Only constants
                    // are supported until runtime empty-axis semantics exist.
                    // convertModel currently returns true even if writeFb
                    // rejects an unsupported Extra op. Check the artifact.
                    ok = rejectsNonconstantAxes(type, test) && !std::ifstream(mnnPath, std::ios::binary).good();
                } else {
                    ok = converted && runModel(mnnPath, test, data, shape, expected);
                }
            }
            std::remove(onnxPath.c_str());
            std::remove(mnnPath.c_str());
            std::printf("%s: %s\n", name.c_str(), ok ? "PASS" : "FAIL");
            if (test.runtimeAxes) {
                rejected += ok;
                ++rejectedTotal;
            } else {
                passed += ok;
                ++total;
            }
        }
    }
    std::printf("Composite reduction conversion: %d/%d passed\n", passed, total);
    std::printf("Nonconstant axes rejection: %d/%d passed\n", rejected, rejectedTotal);
    return passed == total && rejected == rejectedTotal ? 0 : 1;
}
