#include <fstream>
#include <sstream>
#include "MNN_generated.h"
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "rapidjson/document.h"
#include "core/Execution.hpp"

using namespace MNN;
int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./fuseTest XXX.spirv XXX.json\n");
        return 0;
    }
    {
        ScheduleConfig config;
        std::vector<ScheduleConfig> configs = {config};
        auto rt = Interpreter::createRuntime(configs);
    }
    rapidjson::Document configJson;
    std::ifstream fileNames(argv[2]);
    if (fileNames.fail()) {
        MNN_ERROR("Can' open config file: %s\n", argv[2]);
        return 0;
    }
    {
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        configJson.Parse(outputStr.c_str());
    }
    if (configJson.HasParseError()) {
        MNN_ERROR("Invalid json\n");
        return 0;
    }

    auto type    = MNN_FORWARD_VULKAN;
    auto creator = MNNGetExtraRuntimeCreator(type);
    if (nullptr == creator) {
        MNN_ERROR("Don't support %d\n", type);
        return 0;;
    }
    MNN::Backend::Info info;
    info.type = type;
    BackendConfig user;
    user.precision = BackendConfig::Precision_High;
    info.user = &user;
    std::shared_ptr<Runtime> runtime(creator->onCreate(info));
    std::shared_ptr<Backend> bn(runtime->onCreate(&user));
    
    // Load Config
    std::unique_ptr<MNN::OpT> op(new OpT);
    op->type = OpType_Extra;
    op->main.type = OpParameter_Extra;
    op->main.value = new ExtraT;
    std::vector<std::shared_ptr<MNN::Tensor>> inputs;
    std::vector<std::shared_ptr<MNN::Tensor>> outputs;
    if (configJson.HasMember("inputs")) {
        auto inputArray = configJson["inputs"].GetArray();
        int pos = 0;
        for (auto iter = inputArray.Begin(); iter != inputArray.End(); iter++) {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = "input";
            attr->list.reset(new ListValueT);
            attr->i = (*iter)["binding"].GetInt();
            attr->list->i = {0, pos};
            attr->b = false;

            op->main.AsExtra()->attr.emplace_back(std::move(attr));
            halide_type_t type = halide_type_of<float>();
            std::vector<int> shape;
            if (iter->HasMember("dims")) {
                auto dimArray = (*iter)["dims"].GetArray();
                for (auto shapeIter = dimArray.Begin(); shapeIter != dimArray.End(); shapeIter++) {
                    shape.emplace_back(shapeIter->GetInt());
                }
            }
            // Create Tensor
            std::shared_ptr<MNN::Tensor> tensor(Tensor::createDevice(shape, type, Tensor::CAFFE));
            bn->onAcquireBuffer(tensor.get(), Backend::STATIC);
            TensorUtils::getDescribeOrigin(tensor.get())->setBackend(bn.get());
            bool isFloat = std::string((*iter)["type"].GetString()) == "float";
            if (iter->HasMember("filename")) {
                auto ptr = tensor->map(MNN::Tensor::MAP_TENSOR_WRITE, MNN::Tensor::CAFFE);
                {
                    auto fileName = std::string( (*iter)["filename"].GetString());
                    FUNC_PRINT_ALL(fileName.c_str(), s);
                    std::ifstream is(fileName.c_str());
                    if (is.fail()) {
                        MNN_ERROR("Can't open data file for %d input\n", pos);
                    }
                    auto size = tensor->elementSize();
                    if (isFloat) {
                        auto uptr = (float*)ptr;
                        for (int i=0; i<size; ++i) {
                            float v;
                            is >> v;
                            uptr[i] = v;
                        }
                    } else {
                        auto uptr = (uint32_t*)ptr;
                        for (int i=0; i<size; ++i) {
                            float v;
                            is >> v;
                            uptr[i] = v;
                        }
                    }
                }
                tensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, MNN::Tensor::CAFFE, ptr);
            }
            inputs.emplace_back(tensor);
            pos++;
        }
    }
    if (configJson.HasMember("outputs")) {
        auto inputArray = configJson["outputs"].GetArray();
        int pos = 0;
        for (auto iter = inputArray.Begin(); iter != inputArray.End(); iter++) {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = "input";
            attr->list.reset(new ListValueT);
            attr->i = (*iter)["binding"].GetInt();
            attr->list->i = {1, pos};
            attr->b = false;

            op->main.AsExtra()->attr.emplace_back(std::move(attr));
            halide_type_t type = halide_type_of<float>();
            std::vector<int> shape;
            if (iter->HasMember("dims")) {
                auto dimArray = (*iter)["dims"].GetArray();
                for (auto shapeIter = dimArray.Begin(); shapeIter != dimArray.End(); shapeIter++) {
                    shape.emplace_back(shapeIter->GetInt());
                }
            }
            // Create Tensor
            std::shared_ptr<MNN::Tensor> tensor(Tensor::createDevice(shape, type, Tensor::CAFFE));
            bn->onAcquireBuffer(tensor.get(), Backend::STATIC);
            TensorUtils::getDescribeOrigin(tensor.get())->setBackend(bn.get());
            outputs.emplace_back(tensor);
            pos++;
        }
    }
    if (configJson.HasMember("uniforms")) {
        auto inputArray = configJson["uniforms"].GetArray();
        int pos = 0;
        for (auto iter = inputArray.Begin(); iter != inputArray.End(); iter++) {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = "const";
            attr->list.reset(new ListValueT);
            attr->i = (*iter)["binding"].GetInt();
            attr->b = true;
            attr->tensor.reset(new BlobT);
            attr->tensor->dataType = DataType_DT_INT32;
            std::vector<int> shape;
            int size = 1;
            if (iter->HasMember("dims")) {
                auto dimArray = (*iter)["dims"].GetArray();
                for (auto shapeIter = dimArray.Begin(); shapeIter != dimArray.End(); shapeIter++) {
                    shape.emplace_back(shapeIter->GetInt());
                    size *= shapeIter->GetInt();
                }
            }
            attr->tensor->dims = shape;
            attr->tensor->dataFormat = MNN_DATA_FORMAT_NCHW;
            if (iter->HasMember("data")) {
                auto dimArray = (*iter)["data"].GetArray();
                for (auto shapeIter = dimArray.Begin(); shapeIter != dimArray.End(); shapeIter++) {
                    attr->tensor->int32s.emplace_back(shapeIter->GetInt());
                }
            }
            op->main.AsExtra()->attr.emplace_back(std::move(attr));
        }
    }
    if (configJson.HasMember("group_size")) {
        std::vector<int> shape;
        auto dimArray = configJson["group_size"].GetArray();
        for (auto shapeIter = dimArray.Begin(); shapeIter != dimArray.End(); shapeIter++) {
            shape.emplace_back(shapeIter->GetInt());
        }
        std::unique_ptr<AttributeT> attr(new AttributeT);
        attr->key = "group_size";
        attr->tensor.reset(new BlobT);
        attr->tensor->int32s = shape;
        op->main.AsExtra()->attr.emplace_back(std::move(attr));
    }
    if (configJson.HasMember("local_size")) {
        std::vector<int> shape;
        auto dimArray = configJson["local_size"].GetArray();
        for (auto shapeIter = dimArray.Begin(); shapeIter != dimArray.End(); shapeIter++) {
            shape.emplace_back(shapeIter->GetInt());
        }
        std::unique_ptr<AttributeT> attr(new AttributeT);
        attr->key = "local_size";
        attr->tensor.reset(new BlobT);
        attr->tensor->int32s = shape;
        op->main.AsExtra()->attr.emplace_back(std::move(attr));
    }
    {
        std::ifstream is(argv[1]);
        if (is.fail()) {
            MNN_ERROR("Can't load spirv\n");
            return 0;
        }
        is.seekg(0, std::ios::end);
        std::unique_ptr<AttributeT> attr(new AttributeT);
        attr->key = "spirv";
        attr->tensor.reset(new BlobT);
        attr->tensor->int8s.resize(is.tellg());
        is.seekg(0, std::ios::beg);
        is.read((char*)attr->tensor->int8s.data(), attr->tensor->int8s.size());
        op->main.AsExtra()->attr.emplace_back(std::move(attr));
    }
    std::vector<Tensor*> inputsW(inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        inputsW[i] = inputs[i].get();
    }
    std::vector<Tensor*> outputsW(outputs.size());
    for (int i=0; i<outputs.size(); ++i) {
        outputsW[i] = outputs[i].get();
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Op::Pack(builder, op.get()));
    auto opRaw = flatbuffers::GetRoot<Op>(builder.GetBufferPointer());
    std::shared_ptr<MNN::Execution> exeution(bn->onCreate(inputsW, outputsW, opRaw));
    bn->onResizeBegin();
    exeution->onResize(inputsW, outputsW);
    bn->onResizeEnd();
    bn->onExecuteBegin();
    exeution->onExecute(inputsW, outputsW);
    bn->onExecuteEnd();
    for (int i=0; i<outputsW.size(); ++i) {
        auto ptr = outputsW[i]->map(MNN::Tensor::MAP_TENSOR_READ, MNN::Tensor::CAFFE);
        auto size = outputsW[i]->elementSize();
        auto iPtr = (int32_t*)ptr;
        std::ostringstream fileNameOs;
        fileNameOs << i << ".txt";
        std::ofstream _o(fileNameOs.str().c_str());
        for (int v=0; v<size; ++v) {
            _o << iPtr[v] << "\n";
        }
        outputsW[i]->unmap(MNN::Tensor::MAP_TENSOR_READ, MNN::Tensor::CAFFE, ptr);
    }
    
    exeution.reset();
    inputs.clear();
    outputs.clear();
    bn.reset();
    runtime.reset();

    return 0;
}
