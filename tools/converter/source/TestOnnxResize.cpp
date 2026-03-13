#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"

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

static void addIntInitializer(onnx::GraphProto* graph, const std::string& name, const std::vector<int64_t>& dims,
                              const std::vector<int32_t>& values) {
    auto* tensor = graph->add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(onnx::TensorProto_DataType_INT32);
    for (auto dim : dims) {
        tensor->add_dims(dim);
    }
    for (auto value : values) {
        tensor->add_int32_data(value);
    }
}

static std::string writeModel(const onnx::ModelProto& model, const std::string& fileName) {
    std::ofstream output(fileName, std::ios::binary | std::ios::trunc);
    model.SerializeToOstream(&output);
    return fileName;
}

static std::unique_ptr<MNN::OpT> makeMetaOp() {
    std::unique_ptr<MNN::OpT> meta(new MNN::OpT);
    meta->type = MNN::OpType_Extra;
    meta->main.type = MNN::OpParameter_Extra;
    meta->main.value = new MNN::ExtraT;
    meta->main.AsExtra()->type = "Meta";
    meta->main.AsExtra()->engine = "MNN";
    return meta;
}

static MNN::OpT* findOp(MNN::NetT* net, const std::string& name) {
    for (auto& op : net->oplists) {
        if (op->name == name) {
            return op.get();
        }
    }
    return nullptr;
}

static bool runConvert(const std::string& modelPath, const std::string& opName, MNN::OpType expectedType,
                       int expectedInputs) {
    std::unique_ptr<MNN::NetT> net(new MNN::NetT);
    auto meta = makeMetaOp();
    std::vector<std::string> inputNames;
    if (onnx2MNNNet(modelPath, "MNN", net, meta.get(), inputNames) != 0) {
        return false;
    }
    auto* resizeOp = findOp(net.get(), opName);
    if (resizeOp == nullptr) {
        return false;
    }
    if (resizeOp->type != expectedType) {
        return false;
    }
    if ((int)resizeOp->inputIndexes.size() != expectedInputs) {
        return false;
    }
    return true;
}

static onnx::ModelProto makeResizeModel(const std::vector<int>& inputShape, bool useSizes) {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.mutable_opset_import()->Add()->set_version(16);
    auto* graph = model.mutable_graph();
    graph->set_name("ResizeTest");

    addTensorShape(graph->add_input(), "input", inputShape);
    addTensorShape(graph->add_output(), "output", inputShape);

    auto* node = graph->add_node();
    node->set_op_type("Resize");
    node->add_input("input");
    node->add_input("");
    if (useSizes) {
        node->add_input("");
        node->add_input("sizes");
    } else {
        node->add_input("scales");
    }
    node->add_output("resize_node");
    auto* attr = node->add_attribute();
    attr->set_name("mode");
    attr->set_s("nearest");

    if (useSizes) {
        std::vector<int32_t> sizes(inputShape.begin(), inputShape.end());
        if (inputShape.size() == 3) {
            sizes[2] *= 2;
        } else if (inputShape.size() == 5) {
            sizes[2] *= 2;
            sizes[3] *= 2;
            sizes[4] *= 2;
        }
        addIntInitializer(graph, "sizes", {(int64_t)sizes.size()}, sizes);
    } else {
        std::vector<float> scales(inputShape.size(), 1.0f);
        if (inputShape.size() == 3) {
            scales[2] = 2.0f;
        } else if (inputShape.size() == 5) {
            scales[2] = 2.0f;
            scales[3] = 2.0f;
            scales[4] = 2.0f;
        }
        addFloatInitializer(graph, "scales", {(int64_t)scales.size()}, scales);
    }
    return model;
}

int main() {
    const std::string rank3Scales = "/tmp/mnn_resize_rank3_scales.onnx";
    const std::string rank3Sizes = "/tmp/mnn_resize_rank3_sizes.onnx";
    const std::string rank5Scales = "/tmp/mnn_resize_rank5_scales.onnx";

    writeModel(makeResizeModel({2, 3, 5}, false), rank3Scales);
    writeModel(makeResizeModel({2, 3, 5}, true), rank3Sizes);
    writeModel(makeResizeModel({1, 2, 3, 4, 5}, false), rank5Scales);

    bool ok = true;
    ok = runConvert(rank3Scales, "resize_node", MNN::OpType_Interp, 2) && ok;
    ok = runConvert(rank3Sizes, "resize_node", MNN::OpType_Interp, 2) && ok;
    ok = runConvert(rank5Scales, "resize_node", MNN::OpType_Interp3D, 1) && ok;

    ::remove(rank3Scales.c_str());
    ::remove(rank3Sizes.c_str());
    ::remove(rank5Scales.c_str());
    return ok ? 0 : 1;
}
