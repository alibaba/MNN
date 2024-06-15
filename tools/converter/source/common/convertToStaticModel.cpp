//
//  convertToStaticModel.cpp
//  MNNConverter
//
//  Created by MNN on 2020/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/FileLoader.hpp"
#include "utils/InitNet.hpp"
#include "core/Command.hpp"
#include "shape/SizeComputer.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "CommonUtils.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN;

#define SET_TYPE(TYPE, type) \
if (tensor->getType() == halide_type_of<type##_t>()) {\
blob->dataType = DataType_DT_##TYPE;

#define CONSTANT_COPY(TYPE, type, bytes) \
SET_TYPE(TYPE, type)\
blob->type##s.resize(tensor->elementSize());\
::memcpy(blob->type##s.data(), tensor->host<type##_t>(), blob->type##s.size() * bytes);\
}

static bool _RemoveDupOutput(MNN::NetT* net, bool abortOpt) {
    std::vector<bool> outputMask(net->tensorName.size(), false);
    std::map<int, TensorDescribeT*> describes;
    for (auto& des : net->extraTensorDescribe) {
        describes.insert(std::make_pair(des->index, des.get()));
    }
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
        auto& op = *iter;
        for (int i=0; i<op->outputIndexes.size(); ++i) {
            auto index = op->outputIndexes[i];
            if (!outputMask[index]) {
                outputMask[index] = true;
                continue;
            }
            if (abortOpt) {
                return false;
            }
            // Dup output, rename it
            int newIndex = (int)net->tensorName.size();
            outputMask.push_back(true);
            std::ostringstream tempOs;
            tempOs << "_" << net->tensorName[index] << "_" << newIndex;
            auto newName = tempOs.str();
            MNN_PRINT("Convert: Dup output %s, replace by %s\n", net->tensorName[index].c_str(), newName.c_str());
            net->tensorName.emplace_back(newName);
            op->outputIndexes[i] = newIndex;
            if (describes.find(index) != describes.end()) {
                auto originDes = describes.find(index)->second;
                std::unique_ptr<TensorDescribeT> newTensorDes;
                flatbuffers::FlatBufferBuilder tempBuilder;
                tempBuilder.Finish(TensorDescribe::Pack(tempBuilder, originDes));
                newTensorDes.reset(flatbuffers::GetRoot<TensorDescribe>(tempBuilder.GetBufferPointer())->UnPack());
                newTensorDes->index = newIndex;
                net->extraTensorDescribe.emplace_back(std::move(newTensorDes));
            }
            for (auto subIter = iter; subIter != net->oplists.end(); ++subIter) {
                auto& subOp = *subIter;
                for (int k=0; k<subOp->inputIndexes.size(); ++k) {
                    if (subOp->inputIndexes[k] == index) {
                        subOp->inputIndexes[k] = newIndex;
                    }
                }
            }
        }
    }
    return true;
}
    

static void _RemoveUnusefulNodes(std::unique_ptr<MNN::NetT>& net) {
    if (!_RemoveDupOutput(net.get(), true)) {
        MNN_PRINT("Can't optimize static model because has loop\n");
        return;
    }
    auto originMode = MNN::Express::ExecutorScope::Current()->getLazyMode();
    MNN::Express::ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
    std::map<std::string, MNN::Express::VARP> varMap;
    auto outputs = std::move(net->outputName);
    {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, net.get()));
        net.reset();
        varMap = MNN::Express::Variable::loadMap(builder.GetBufferPointer(), builder.GetSize());
    }
    std::vector<MNN::Express::VARP> outputVars;
    std::vector<std::string> validOutputs;
    for (auto& name : outputs) {
        auto iter = varMap.find(name);
        if (iter == varMap.end()) {
            MNN_ERROR("Convert Static Model: Can't find %s output, skip\n", name.c_str());
            continue;
        }
        validOutputs.emplace_back(name);
        outputVars.emplace_back(iter->second);
    }
    auto buffer = MNN::Express::Variable::save(outputVars);
    outputVars.clear();
    varMap.clear();
    net.reset(flatbuffers::GetRoot<MNN::Net>(buffer.data())->UnPack());
    buffer.clear();
    net->outputName = validOutputs;
    MNN::Express::ExecutorScope::Current()->setLazyComputeMode(originMode);
}

static void genStaticModel(CommandBuffer buffer, const std::string& modelName, std::map<Tensor*, std::pair<std::string, int>>& tensorNames, std::vector<std::string>&& outputNames, const Net* originNetInfo) {
    MNN_PRINT("gen Static Model ... \n");
    std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
    netT->outputName = std::move(outputNames);
    netT->usage = Usage_INFERENCE_STATIC;
    std::map<Tensor*, int> tensorMap;
    // Add tensorName to new netT
    netT->tensorName.resize(tensorNames.size());
    std::vector<std::unique_ptr<OpT>> inputOps;
    for (auto& iter : tensorNames) {
        netT->tensorName[iter.second.second] = iter.second.first;
        tensorMap.insert(std::make_pair(iter.first, iter.second.second));
        if (TensorUtils::getDescribe(iter.first)->usage == MNN::Tensor::InsideDescribe::INPUT) {
            std::unique_ptr<OpT> input(new OpT);
            input->type = OpType_Input;
            input->name = iter.second.first;
            input->outputIndexes = {iter.second.second};
            input->main.value = new InputT;
            input->main.type = OpParameter_Input;
            input->main.AsInput()->dims = iter.first->shape();
            input->main.AsInput()->dformat = TensorUtils::getDescribe(iter.first)->dimensionFormat;
            auto type = iter.first->getType();
            if (type.code == halide_type_float) {
                if (type.bits == 32) {
                    input->main.AsInput()->dtype = DataType_DT_FLOAT;
                } else if (type.bits == 16) {
                    input->main.AsInput()->dtype = DataType_DT_HALF;
                }
            } else if (type.code == halide_type_int) {
                if (type.bits == 32) {
                    input->main.AsInput()->dtype = DataType_DT_INT32;
                } else if (type.bits == 16) {
                    input->main.AsInput()->dtype = DataType_DT_INT16;
                } else if (type.bits == 8) {
                    input->main.AsInput()->dtype = DataType_DT_INT8;
                }
            } else if (type.code == halide_type_uint) {
                if (type.bits == 16) {
                    input->main.AsInput()->dtype = DataType_DT_UINT16;
                } else if (type.bits == 8) {
                    input->main.AsInput()->dtype = DataType_DT_UINT8;
                }
            }
            inputOps.emplace_back(std::move(input));
        }
    }
    // add Tensors to netT
    for (auto& iterP : buffer.command) {
        auto& iter = *iterP;
        std::function<void(Tensor*)> insertTensor = [&](Tensor* t) {
            if (tensorMap.find(t) == tensorMap.end()) {
                int index = static_cast<int>(tensorMap.size());
                tensorMap.insert(std::make_pair(t, index));
                std::string tensorName = "ExtraTensor_" + std::to_string(index);
                netT->tensorName.push_back(tensorName);
            }
        };
        for (auto& t : iter.inputs) {
            insertTensor(t);
        }
        for (auto& t : iter.outputs) {
            insertTensor(t);
        }
    }
    // add tensors' describe to netT
    for (auto tensorPair : tensorMap) {
        auto tensor = tensorPair.first;
        auto index = tensorPair.second;
        //FUNC_PRINT(index);
        auto des = TensorUtils::getDescribe(tensor);
        if (des->usage == Tensor::InsideDescribe::CONSTANT || des->usage == MNN::Tensor::InsideDescribe::TRAINABLE) {
            std::unique_ptr<OpT> op(new OpT);
            if (des->usage == Tensor::InsideDescribe::CONSTANT) {
                op->type = OpType_Const;
            } else {
                op->type = OpType_TrainableParam;
            }
            auto blob = new BlobT;
            op->main.type = OpParameter_Blob;
            op->main.value = blob;
            blob->dataFormat = des->dimensionFormat;
            for (int d = 0; d < tensor->dimensions();d++) {
                blob->dims.push_back(tensor->buffer().dim[d].extent);
            }
            if (tensor->getType() == halide_type_of<float>()) {
                blob->dataType = DataType_DT_FLOAT;
                blob->float32s.resize(tensor->elementSize());
                ::memcpy(blob->float32s.data(), tensor->host<void>(), blob->float32s.size() * sizeof(float));
            } else {
                CONSTANT_COPY(INT8, int8, 1);
                CONSTANT_COPY(UINT8, uint8, 1);
                CONSTANT_COPY(INT32, int32, 4)
                CONSTANT_COPY(INT64, int64, 8);
            }
            op->outputIndexes.push_back(index);
            netT->oplists.emplace_back(std::move(op));
        }
        auto describe = std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
        describe->index = index;
        describe->blob = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        auto& blob = describe->blob;
        blob->dataFormat = des->dimensionFormat;
        if (tensor->getType() == halide_type_of<float>()) {
            blob->dataType = DataType_DT_FLOAT;
        } else {
            SET_TYPE(INT8, int8)}
            SET_TYPE(UINT8, uint8)}
            SET_TYPE(INT32, int32)}
            SET_TYPE(INT64, int64)}
        }
        for (int d = 0; d < tensor->dimensions();d++) {
            describe->blob->dims.push_back(tensor->buffer().dim[d].extent);
        }
        auto tensorDes = TensorUtils::getDescribe(tensor);
        if (nullptr != tensorDes->quantAttr) {
            describe->quantInfo.reset(new TensorQuantInfoT);
            describe->quantInfo->max = tensorDes->quantAttr->max;
            describe->quantInfo->min = tensorDes->quantAttr->min;
            describe->quantInfo->zero = tensorDes->quantAttr->zero;
            describe->quantInfo->scale = tensorDes->quantAttr->scale;
        }
        for (auto& reg : des->regions) {
            auto regionT = std::unique_ptr<MNN::RegionT>(new MNN::RegionT);
            regionT->src = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
            regionT->dst = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
            regionT->src->offset = reg.src.offset;
            regionT->dst->offset = reg.dst.offset;
            for (int s = 0; s < 3; s++) {
                regionT->src->stride.push_back(reg.src.stride[s]);
                regionT->dst->stride.push_back(reg.dst.stride[s]);
                regionT->size.push_back(reg.size[s]);
            }
            describe->regions.emplace_back(std::move(regionT));
        }
        netT->extraTensorDescribe.emplace_back(std::move(describe));
    }
    // add op to netT
    for (auto&& iter : inputOps) {
        netT->oplists.emplace_back(std::move(iter));
    }
    int idx = 0;
    for (auto& iterP : buffer.command) {
        auto& iter = *iterP;
        auto opt = iter.op->UnPack();
        if (opt->name.size() <= 0) {
            opt->name = std::string("Geometry_") + MNN::EnumNameOpType(opt->type) + std::to_string(idx++);
        }
        opt->inputIndexes.resize(iter.inputs.size());
        opt->outputIndexes.resize(iter.outputs.size());
        for (int i = 0; i < iter.outputs.size(); i++) {
            opt->outputIndexes[i] = tensorMap[iter.outputs[i]];
        }
        for (int i = 0; i < iter.inputs.size(); i++) {
            opt->inputIndexes[i] = tensorMap[iter.inputs[i]];
        }
        netT->oplists.emplace_back(std::move(opt));
    }
    _RemoveUnusefulNodes(netT);
    netT->usage = Usage_INFERENCE_STATIC;
    netT->sourceType = originNetInfo->sourceType();
    if (nullptr != originNetInfo->bizCode()) {
        netT->bizCode = originNetInfo->bizCode()->str();
    }
    if (nullptr != originNetInfo->mnn_uuid()) {
        netT->mnn_uuid = originNetInfo->mnn_uuid()->str();
    }
    netT->extraInfo.reset(new ExtraInfoT);
    netT->extraInfo->version = MNN_VERSION;
    // write netT to file
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();
    std::ofstream output(modelName, std::ofstream::binary);
    output.write((const char*)bufferOutput, sizeOutput);
}

void converToStaticModel(const Net* net, std::map<std::string,std::vector<int>>& inputConfig, std::string mnnFile) {
    // set a backend and context to run resize
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_High;
    config.backendConfig = &backendConfig;
    Backend::Info compute;
    compute.type = config.type;
    compute.numThread = config.numThread;
    compute.user = config.backendConfig;
    const RuntimeCreator* runtimeCreator(MNNGetExtraRuntimeCreator(compute.type));
    std::unique_ptr<Runtime> runtime(runtimeCreator->onCreate(compute));
    std::shared_ptr<Backend> backend(runtime->onCreate());
    BackendConfig defaultConfig;
    defaultConfig.flags = 4;
    std::shared_ptr<Backend> defaultBackend(runtime->onCreate(&defaultConfig));
    std::vector<std::shared_ptr<Tensor>> allTensors;
    allTensors.resize(net->tensorName()->size());
    ErrorCode code = NO_ERROR;
    initConstTensors(allTensors, net, defaultBackend.get(), code, nullptr);
    if (NO_ERROR != code) {
        MNN_ERROR("Init tensor error code = %d\n", code);
        return;
    }
    bool valid = initTensors(allTensors, net);
    // set tensors' shape by inputConfig
    for (int i = 0; i < allTensors.size(); i++) {
        auto name = net->tensorName()->GetAsString(i)->str();
        if (inputConfig.find(name) != inputConfig.end()) {
            auto& dims = inputConfig[name];
            allTensors[i]->buffer().dimensions = dims.size();
            for (int j = 0; j < dims.size(); j++) {
                allTensors[i]->setLength(j, dims[j]);
            }
        }
    }
    std::vector<Schedule::OpCacheInfo> infos;
    initPipelineInfosFromNet(infos, net, allTensors);
    GeometryComputer::Context ctx(Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_ALL, defaultBackend);
    // resize the session's info and store to buffer
    std::vector<Tensor*> constTensors;
    GeometryComputerUtils::buildConstantTensors(infos);
    GeometryComputerUtils::shapeComputeAndGeometryTransform(nullptr, infos, ctx, defaultBackend, runtime->onGetCompilerType());
    std::map<Tensor*, std::pair<std::string, int>> tensorName;
    for (int i = 0; i < net->tensorName()->size(); i++) {
        tensorName[allTensors[i].get()] = std::make_pair(net->tensorName()->GetAsString(i)->str(), i);
    }
    std::vector<std::string> outputNames;
    if (net->outputName() != nullptr) {
        for (int i=0; i<net->outputName()->size(); ++i) {
            outputNames.emplace_back(net->outputName()->GetAsString(i)->str());
        }
    } else {
        for (int i = 0; i < net->tensorName()->size(); i++) {
            if (TensorUtils::getDescribe(allTensors[i].get())->usage == MNN::Tensor::InsideDescribe::OUTPUT) {
                outputNames.emplace_back(net->tensorName()->GetAsString(i)->str());
            }
        }
    }
    CommandBuffer newBuffer;
    for (auto& info : infos) {
        if (info.type == MNN::Schedule::CONSTANT) {
            continue;
        }
        // TODO: Remove inside constant op in future
        auto& buf = info.executeBuffer;
        newBuffer.command.insert(newBuffer.command.end(), buf.command.begin(), buf.command.end());
        newBuffer.extras.insert(newBuffer.extras.end(), buf.extras.begin(), buf.extras.end());
    }
    // store buffer to STATIC model file
    genStaticModel(newBuffer, mnnFile, tensorName, std::move(outputNames), net);
}
