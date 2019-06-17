//
//  transformer.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include "Interpreter.hpp"
#include "LossFunction.hpp"
#include "Macro.h"
#include "OpConverter.hpp"
#include "OpGrad.hpp"
#include "Tensor.hpp"
#include "TensorUtils.hpp"
#include "rapidjson/document.h"

using namespace MNN;
using namespace std;

static void _addBackwardOp(const OpT* forwardOp, NetT* dest, std::map<int, std::vector<int>>& backwardTensors,
                           std::map<int, std::shared_ptr<Tensor>>& tensors) {
    static set<OpType> gDonothing{
        OpType_Const,
        OpType_Input,
    };
    if (gDonothing.find(forwardOp->type) != gDonothing.end()) {
        return;
    }

    // Merge Output Diff If Needed
    bool needBackward = false;
    for (int i = 0; i < forwardOp->outputIndexes.size(); ++i) {
        auto outputIndex = forwardOp->outputIndexes[i];
        auto iter        = backwardTensors.find(outputIndex);
        if (iter == backwardTensors.end()) {
            continue;
        }
        needBackward = true;
        if (iter->second.size() == 1) {
            continue;
        }
        int addId = iter->second[0];
        for (int j = 1; j < iter->second.size(); ++j) {
            auto newTensorId = (int)dest->tensorName.size();
            dest->tensorName.emplace_back(forwardOp->name + "_Merge_" + numberToString(outputIndex) + "_" +
                                          numberToString(j));
            unique_ptr<OpT> newOp(new OpT);
            newOp->type          = OpType_BinaryOp;
            newOp->name          = dest->tensorName[newTensorId];
            newOp->outputIndexes = {newTensorId};
            newOp->inputIndexes  = {addId, iter->second[j]};
            newOp->main.type     = OpParameter_BinaryOp;
            auto elt             = new BinaryOpT;
            elt->T               = DataType_DT_FLOAT;
            elt->opType          = BinaryOpOperation_ADD;
            newOp->main.value    = elt;
            dest->oplists.emplace_back(std::move(newOp));
            addId = newTensorId;
        }
        iter->second = {addId};
    }
    // No diff, just return
    if (!needBackward) {
        MNN_PRINT("%s has no grad tensor input\n", forwardOp->name.c_str());
        return;
    }
    std::vector<Tensor*> inputs;
    inputs.resize(forwardOp->inputIndexes.size());
    for (int i = 0; i < forwardOp->inputIndexes.size(); ++i) {
        if (tensors.find(forwardOp->inputIndexes[i]) != tensors.end()) {
            inputs[i] = tensors[forwardOp->inputIndexes[i]].get();
        }
    }
    std::vector<Tensor*> outputs;
    outputs.resize(forwardOp->outputIndexes.size());
    for (int i = 0; i < forwardOp->outputIndexes.size(); ++i) {
        if (tensors.find(forwardOp->outputIndexes[i]) != tensors.end()) {
            outputs[i] = tensors[forwardOp->outputIndexes[i]].get();
            if (outputs[i]->getType().code != halide_type_float) {
                MNN_PRINT("Not float op, dont grad: %s\n", forwardOp->name.c_str());
                return;
            }
        }
    }
    auto gradCreator = OpGrad::get(forwardOp->type);
    if (nullptr == gradCreator) {
        MNN_PRINT("Can't compute type=%d, name= %s grad\n", forwardOp->type, forwardOp->name.c_str());
        return;
    }

    std::shared_ptr<OpGrad> grad(gradCreator->onCreate(forwardOp, inputs, outputs));
    if (nullptr == grad) {
        MNN_PRINT("Can't compute type=%d, name= %s grad\n", forwardOp->type, forwardOp->name.c_str());
        return;
    }
    FUNC_PRINT_ALL(forwardOp->name.c_str(), s);
    auto res = grad->onGradCommon(dest, forwardOp, backwardTensors);
    if (!res) {
        MNN_ERROR("Error for compute %s grad\n", forwardOp->name.c_str());
    }
}
static void _computeTensorShape(const void* buffer, size_t size, std::map<int, std::shared_ptr<Tensor>>& result,
                                const std::map<std::string, std::vector<int>>& shapes) {
    std::unique_ptr<Interpreter> interp(Interpreter::createFromBuffer(buffer, size));
    const Net* net = GetNet(interp->getModelBuffer().first);
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = interp->createSession(config);
    for (auto& iter : shapes) {
        auto input = interp->getSessionInput(session, iter.first.c_str());
        if (nullptr != input) {
            interp->resizeTensor(input, iter.second);
        }
    }
    interp->resizeSession(session);
    TensorCallBackWithInfo begin = [net, &result](const std::vector<Tensor*>& inputs, const OperatorInfo* info) {
        auto opName = info->name();
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (opName == op->name()->str()) {
                for (int index = 0; index < op->inputIndexes()->size(); ++index) {
                    auto outputIndex = op->inputIndexes()->data()[index];
                    if (result.find(outputIndex) == result.end()) {
                        std::shared_ptr<Tensor> tensor;
                        if (TensorUtils::getDescribe(inputs[index])->isConst) {
                            tensor.reset(Tensor::createHostTensorFromDevice(inputs[index], true));
                        } else {
                            tensor.reset(Tensor::createDevice(inputs[index]->shape(), inputs[index]->buffer().type,
                                                              inputs[index]->getDimensionType()));
                        }
                        result.insert(std::make_pair(outputIndex, tensor));
                    }
                }
                break;
            }
        }
        return false;
    };
    TensorCallBackWithInfo after = [&result, net](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
        auto opName = info->name();
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (opName == op->name()->str()) {
                for (int index = 0; index < op->outputIndexes()->size(); ++index) {
                    auto outputIndex = op->outputIndexes()->data()[index];
                    if (result.find(outputIndex) == result.end()) {
                        std::shared_ptr<Tensor> tensor(Tensor::createDevice(outputs[index]->shape(),
                                                                            outputs[index]->buffer().type,
                                                                            outputs[index]->getDimensionType()));
                        result.insert(std::make_pair(outputIndex, tensor));
                    }
                }
                break;
            }
        }
        return true;
    };
    interp->runSessionWithCallBackInfo(session, begin, after);
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./transformer.out temp.bin dst.bin config.json\n");
        return 0;
    }
    rapidjson::Document document;
    {
        std::ifstream fileNames(argv[3]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        FUNC_PRINT(document.HasParseError());
        FUNC_PRINT(document.IsArray());
        FUNC_PRINT(document.IsObject());
    }
    auto configObject = document.GetObject();
    bool addTrain     = configObject["Train"].GetBool();
    bool hasLoss      = configObject.HasMember("Loss");
    std::map<std::string, std::vector<int>> shapes;
    if (configObject.HasMember("Shape")) {
        auto shapeArray = configObject["Shape"].GetObject();
        for (auto shapeIter = shapeArray.begin(); shapeIter != shapeArray.end(); shapeIter++) {
            auto dimArray = shapeIter->value.GetArray();
            std::vector<int> dims;
            for (auto dimIter = dimArray.begin(); dimIter != dimArray.end(); dimIter++) {
                dims.emplace_back(dimIter->GetInt());
            }
            FUNC_PRINT_ALL(shapeIter->name.GetString(), s);
            shapes.insert(std::make_pair(shapeIter->name.GetString(), dims));
        }
    }

    unique_ptr<NetT> net;
    std::map<int, std::shared_ptr<Tensor>> tensors;
    {
        const char* fileName = argv[1];
        FUNC_PRINT_ALL(fileName, s);
        std::ifstream fs(fileName, std::ifstream::in | std::ifstream::binary);
        std::ostringstream os;
        os << fs.rdbuf();
        auto buffer = os.str();
        _computeTensorShape((const void*)buffer.c_str(), buffer.size(), tensors, shapes);
        net = UnPackNet(buffer.c_str());
    }
    {
        // Turn convolution be trainable convolution
        std::vector<std::unique_ptr<OpT>> newOpLists;
        for (auto& op : net->oplists) {
            auto converter = OpConverter::get(op->type);
            if (nullptr == converter) {
                newOpLists.emplace_back(std::move(op));
                continue;
            }
            auto convertResult = converter->onConvert(op.get(), net.get());
            if (convertResult.opLists.empty()) {
                newOpLists.emplace_back(std::move(op));
                continue;
            }
            for (int i = 0; i < convertResult.tensorNames.size(); ++i) {
                net->tensorName.emplace_back(convertResult.tensorNames[i]);
            }
            for (auto& newOp : convertResult.opLists) {
                newOpLists.emplace_back(std::move(newOp));
            }
        }
        net->oplists = std::move(newOpLists);
    }

    if (addTrain) {
        // Collect Const Variable
        std::vector<int> variables;
        std::vector<std::string> variableLimits;
        if (configObject.HasMember("Optimizor")) {
            auto optimizor = configObject["Optimizor"].GetObject();
            if (optimizor.HasMember("Variables")) {
                auto limitArray = optimizor["Variables"].GetArray();
                for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                    variableLimits.emplace_back(vIter->GetString());
                    MNN_PRINT("Variabale contain : %s \n", vIter->GetString());
                }
            }
        }
        for (auto& op : net->oplists) {
            if (op->type == OpType_Const) {
                // if (op->main.AsBlob()->dataType == DataType_DT_FLOAT && op->name.find("Bias") == string::npos) {
                if (op->main.AsBlob()->dataType == DataType_DT_FLOAT) {
                    bool valid = true;
                    if (!variableLimits.empty()) {
                        valid = false;
                        for (auto& s : variableLimits) {
                            if (op->name.find(s) != std::string::npos) {
                                valid = true;
                                break;
                            }
                        }
                    }
                    if (valid) {
                        MNN_PRINT("Add Variable: %s\n", op->name.c_str());
                        variables.emplace_back(op->outputIndexes[0]);
                    }
                }
            }
        }
        OpT* lossOp = nullptr;
        if (hasLoss) {
            auto lossName = configObject["Loss"].GetObject()["op"].GetString();
            for (auto& op : net->oplists) {
                if (op->name == lossName) {
                    lossOp = op.get();
                    break;
                }
            }
        } else {
            auto lastOp = net->oplists[net->oplists.size() - 1].get();
            MNN_ASSERT(lastOp->outputIndexes.size() == 1);
            lossOp = LossFunction::addSubEclLoss(net.get(), lastOp, tensors);
            //            lossOp = LossFunction::addProbLoss(net.get(), lastOp, tensors);
        }

        std::stack<const OpT*> allOps;
        for (auto& iter : net->oplists) {
            allOps.push(iter.get());
        }

        std::map<int, std::vector<int>> backwardTensors;
        // Add Init Backward
        {
            std::unique_ptr<OpT> newOp(new OpT);
            newOp->type       = OpType_Const;
            newOp->name       = "InitGradLoss";
            newOp->main.type  = OpParameter_Blob;
            auto blob         = new BlobT;
            blob->dataFormat  = MNN_DATA_FORMAT_NHWC;
            blob->dataType    = DataType_DT_FLOAT;
            blob->float32s    = {1.0f};
            blob->dims        = {};
            newOp->main.value = blob;

            newOp->outputIndexes.emplace_back(net->tensorName.size());
            net->tensorName.emplace_back(newOp->name);
            std::shared_ptr<Tensor> tensor(Tensor::createDevice<float>({}));
            tensors[newOp->outputIndexes[0]] = (tensor);
            backwardTensors.insert(make_pair(lossOp->outputIndexes[0], vector<int>{newOp->outputIndexes[0]}));
            net->oplists.emplace_back(std::move(newOp));
        }
        // Add Back Forward
        {
            while (!allOps.empty()) {
                auto op = allOps.top();
                _addBackwardOp(op, net.get(), backwardTensors, tensors);
                allOps.pop();
            }
        }

        // 指定的不更新的variable，TODO
        std::vector<int> noUpdateVariables;

        // delete useless op
        {
            // 需关注变量对应的梯度节点
            std::set<int> varGrads;
            for (auto variable : variables) {
                auto it = std::find(noUpdateVariables.begin(), noUpdateVariables.end(), variable);
                if (it != noUpdateVariables.end()) {
                    continue;
                }

                auto iter = backwardTensors.find(variable);
                if (iter == backwardTensors.end()) {
                    continue;
                }

                for (auto grad : iter->second) {
                    varGrads.insert(grad);
                }
            }

            std::vector<std::unique_ptr<OpT>> validOps;
            std::vector<std::unique_ptr<OpT>> backwardOps; // 梯度求导的op
            bool overLossGrad = false;
            for (auto& op : net->oplists) {
                std::string name = op->name;
                if (overLossGrad) {
                    backwardOps.emplace_back(std::move(op));
                } else {
                    validOps.emplace_back(std::move(op));
                }

                if (lossOp->name == name) {
                    overLossGrad = true;
                }
            }

            std::set<int> linkGrads    = varGrads; // 每一次循环的关联梯度，初始为varGrads
            std::set<int> allLinkGrads = varGrads; // 所有已经关联的梯度

            std::vector<int> validGradOpIndexes; // 有效的grad op index

            /**
             * 算法：寻找以梯度节点作为输出op，这些op的输入节点有间接的贡献，下一次寻找以这些节点作为输出的op，如此迭代重复，直到找到图上所有对变量求导有贡献的op
             * */
            while (true) {
                std::set<int> nextLinkGrads;

                for (int i = 0; i < backwardOps.size(); i++) {
                    auto iter = std::find(validGradOpIndexes.begin(), validGradOpIndexes.end(), i);
                    if (iter != validGradOpIndexes.end()) {
                        continue;
                    }

                    auto& op = backwardOps[i];

                    // op输出包含梯度节点，说明该op对梯度求导有贡献
                    std::vector<int> intersectionGrads;
                    std::set_intersection(op->outputIndexes.begin(), op->outputIndexes.end(), linkGrads.begin(),
                                          linkGrads.end(), inserter(intersectionGrads, intersectionGrads.begin()));
                    if (intersectionGrads.size() > 0) {
                        validGradOpIndexes.emplace_back(i); // 添加有贡献的op index

                        for (int k = 0; k < op->inputIndexes.size(); k++) {
                            auto index = op->inputIndexes[k];
                            auto iter  = std::find(allLinkGrads.begin(), allLinkGrads.end(), index);
                            if (iter == allLinkGrads.end()) {
                                nextLinkGrads.insert(index);
                                allLinkGrads.insert(index);
                            }
                        }
                    }
                }

                linkGrads = nextLinkGrads;
                if (0 == linkGrads.size()) {
                    break;
                }
            }

            // Print Debug Info
            MNN_PRINT("Delete Useless OP:\n");
            for (int i = 0; i < backwardOps.size(); i++) {
                auto iter = std::find(validGradOpIndexes.begin(), validGradOpIndexes.end(), i);
                if (iter == validGradOpIndexes.end()) {
                    auto& op = backwardOps[i];
                    MNN_PRINT("%s\n", op->name.c_str());
                }
            }

            // 裁剪后的op
            std::sort(validGradOpIndexes.begin(), validGradOpIndexes.end());
            for (int i = 0; i < validGradOpIndexes.size(); i++) {
                auto index = validGradOpIndexes[i];
                validOps.emplace_back(std::move(backwardOps[index]));
            }

            net->oplists = std::move(validOps);
        }
        // Add Grad Compute
        if (configObject.HasMember("Optimizor")) {
            std::unique_ptr<OpT> newOp(new OpT);
            newOp->type       = OpType_Input;
            newOp->name       = "LearningRate";
            newOp->main.type  = OpParameter_Input;
            auto elt          = new InputT;
            elt->dims         = {};
            elt->dtype        = DataType_DT_FLOAT;
            elt->dformat      = MNN_DATA_FORMAT_NCHW;
            newOp->main.value = elt;
            newOp->outputIndexes.emplace_back(net->tensorName.size());
            net->tensorName.emplace_back(newOp->name);
            int learnRate = newOp->outputIndexes[0];
            net->oplists.emplace_back(std::move(newOp));

            for (auto variable : variables) {
                auto iter = backwardTensors.find(variable);
                if (iter == backwardTensors.end()) {
                    continue;
                }
                for (auto grad : iter->second) {
                    std::unique_ptr<OpT> scaleOp(new OpT);
                    scaleOp->type      = OpType_BinaryOp;
                    scaleOp->name      = "scale_" + net->tensorName[grad];
                    scaleOp->main.type = OpParameter_BinaryOp;
                    {
                        auto elt            = new BinaryOpT;
                        elt->opType         = 2; // MUL
                        elt->T              = DataType_DT_FLOAT;
                        scaleOp->main.value = elt;
                    }
                    scaleOp->inputIndexes  = {grad, learnRate};
                    scaleOp->outputIndexes = {grad};
                    net->oplists.emplace_back(std::move(scaleOp));

                    std::unique_ptr<OpT> addOp(new OpT);
                    addOp->type      = OpType_BinaryOp;
                    addOp->name      = "update_" + net->tensorName[variable];
                    addOp->main.type = OpParameter_BinaryOp;
                    {
                        auto elt          = new BinaryOpT;
                        elt->T            = DataType_DT_FLOAT;
                        elt->opType       = BinaryOpOperation_ADD;
                        addOp->main.value = elt;
                    }
                    addOp->inputIndexes  = {variable, grad};
                    addOp->outputIndexes = {variable};
                    net->oplists.emplace_back(std::move(addOp));
                }
            }
        }
    }

    {
        const char* outputName = argv[2];
        FUNC_PRINT_ALL(outputName, s);
        net->tensorNumber = net->tensorName.size();

        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, net.get());
        builder.Finish(offset);
        ofstream os(outputName);
        os.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    }

    return 0;
}
