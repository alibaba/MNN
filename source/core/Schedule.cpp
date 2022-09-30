//
//  Schedule.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Schedule.hpp"
#include <algorithm>
#include <iterator>
#include <set>
#include <vector>
#include <unordered_map>
#include "core/Macro.h"
#include "core/RuntimeFactory.hpp"
#include "core/TensorUtils.hpp"
#ifndef MNN_BUILD_MINI
#include "shape/SizeComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#endif
#include "utils/InitNet.hpp"

//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace std;
//#define MNN_AUTO_CHECK_COST
namespace MNN {

MNNForwardType Schedule::getApprociateType(const ScheduleConfig& config) {
    MNNForwardType type = config.type;
    // FIXME: Support Auto determine
    if (MNN_FORWARD_AUTO == config.type) {
	//Define Auto choose priority
        std::vector<MNNForwardType> priorityList;
        priorityList.push_back(MNN_FORWARD_USER_0); //HIAI
        priorityList.push_back(MNN_FORWARD_NN);     //CoreML
        priorityList.push_back(MNN_FORWARD_USER_1); //TensoRT
        priorityList.push_back(MNN_FORWARD_CUDA);   //CUDA
        priorityList.push_back(MNN_FORWARD_OPENCL); //OpenCL
        priorityList.push_back(MNN_FORWARD_METAL);  //METAL
        priorityList.push_back(MNN_FORWARD_CPU);    //CPU

        for (auto bn : priorityList) {
            if (MNNGetExtraRuntimeCreator(bn) != nullptr) {
                type = (MNNForwardType)bn;
                break;
            }
        }
    }
    auto creator = MNNGetExtraRuntimeCreator(type);
    if (nullptr == creator) {
        MNN_PRINT("Can't Find type=%d backend, use %d instead\n", type, config.backupType);
        type = config.backupType;
    }
    return type;
}

static bool _setUpTensorInfo(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net) {
    auto valid = initTensors(tensors, net);
    if (net->usage() != Usage_INFERENCE_STATIC) {
        return valid;
    }
    // static model will set all tensors' shape
    auto describes = net->extraTensorDescribe();
    std::vector<const TensorDescribe*> des(tensors.size());
    for (int i = 0; i < describes->size(); i++) {
        int index  = describes->GetAs<TensorDescribe>(i)->index();
        des[index] = describes->GetAs<TensorDescribe>(i);
    }
    for (int i = 0; i < tensors.size(); ++i) {
        if (TensorUtils::getDescribe(tensors[i].get())->usage != Tensor::InsideDescribe::NORMAL) {
            // Const / Trainable Shape has been inited
            continue;
        }
        auto blob = des[i]->blob();
        auto& tb = tensors[i]->buffer();
        if (auto idims = blob->dims()) {
            for (int d = 0; d < idims->size(); d++) {
                tb.dim[d].extent = idims->Get(d);
            }
            tb.dimensions = idims->size();
        } else {
            tb.dimensions = 0;
        }
        tensors[i]->setType(blob->dataType());
    }
    for (int i = 0; i < tensors.size(); ++i) {
        auto blob                                                   = des[i]->blob();
        TensorUtils::getDescribe(tensors[i].get())->dimensionFormat = blob->dataFormat();
        if (auto regions = des[i]->regions()) {
            auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
            TensorUtils::getDescribe(tensors[i].get())->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
            regs.reserve(regions->size());
            for (int r = 0; r < regions->size(); r++) {
                auto region = regions->GetAs<Region>(r);
                Tensor::InsideDescribe::Region reg;
                reg.origin     = tensors[region->origin()].get();
                reg.src.offset = region->src()->offset();
                reg.dst.offset = region->dst()->offset();
                for (int d = 0; d < 3; d++) {
                    reg.size[d]       = region->size()->data()[d];
                    reg.src.stride[d] = region->src()->stride()->data()[d];
                    reg.dst.stride[d] = region->dst()->stride()->data()[d];
                }
                regs.emplace_back(std::move(reg));
            }
        }
    }
    return valid;
}

static void generateScheduleGraph(vector<const Op*>& ops, const Net* net, const ScheduleConfig& configs,
                                  const vector<shared_ptr<Tensor>>& allTensors) {

        // for (int i = 0; i < net->oplists()->size(); ++i) {
        //     auto op       = net->oplists()->Get(i);
        //     MNN_PRINT("generateScheduleGraph, op type:%s, op name:%s\n", EnumNameOpType(op->type()), op->name()->c_str());
        // }

    if (configs.path.inputs.empty() && configs.path.outputs.empty()) {
        // Use Default Linear schedule
        ops.clear();
        ops.reserve(net->oplists()->size());
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            ops.emplace_back(op);
        }
        return;
    }
    // 0: not set, 1: output, 2:input
    std::vector<int> tensorMask(net->tensorName()->size());
    ::memset(tensorMask.data(), 0, tensorMask.size() * sizeof(int));

    // 0: use, 1: no use
    std::vector<int> opMask(net->oplists()->size());
    ::memset(opMask.data(), 0, opMask.size() * sizeof(int));

    // Set Initial Status
    std::set<std::string> inputNames;
    std::set<std::string> outputNames;
    for (auto& n : configs.path.inputs) {
        inputNames.insert(n);
    }
    for (auto& n : configs.path.outputs) {
        outputNames.insert(n);
    }
    if (configs.path.mode == ScheduleConfig::Path::Mode::Tensor) {
        for (int i=0; i<tensorMask.size(); ++i) {
            auto name = net->tensorName()->GetAsString(i)->c_str();
            if (outputNames.find(name) != outputNames.end()) {
                tensorMask[i] = 1;
            }
            // If both input/output, set as input
            if (inputNames.find(name) != inputNames.end()) {
                tensorMask[i] = 2;
            }
        }
    } else {
        // Op Mode
        for (int i=0; i<opMask.size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (nullptr == op->name()) {
                continue;
            }
            auto name = op->name()->c_str();
            if (outputNames.find(name) != outputNames.end()) {
                opMask[i] = 1;
                if (nullptr != op->outputIndexes()) {
                    for (int j=0; j<op->outputIndexes()->size(); ++j) {
                        auto index = op->outputIndexes()->data()[j];
                        if (tensorMask[index] != 2) {
                            tensorMask[index] = 1;
                        }
                    }
                }
                if (nullptr != op->inputIndexes()) {
                    for (int j=0; j<op->inputIndexes()->size(); ++j) {
                        auto index = op->inputIndexes()->data()[j];
                        if (tensorMask[index] != 2) {
                            tensorMask[index] = 1;
                        }
                    }
                }
            }
            if (inputNames.find(name) != inputNames.end()) {
                opMask[i] = 1;
                if (nullptr != op->outputIndexes()) {
                    for (int j=0; j<op->outputIndexes()->size(); ++j) {
                        auto index = op->outputIndexes()->data()[j];
                        tensorMask[index] = 2;
                    }
                }
            }
        }
    }

    bool change = false;
    do {
        change = false;
        for (int i=0; i<opMask.size(); ++i) {
            if (opMask[i] > 0) {
                continue;
            }
            auto op = net->oplists()->GetAs<Op>(i);
            if (nullptr != op->outputIndexes()) {
                for (int j=0; j<op->outputIndexes()->size(); ++j) {
                    auto index = op->outputIndexes()->data()[j];
                    if (tensorMask[index] == 1) {
                        opMask[i] = 1;
                        change = true;
                    }
                }
            }
            if (nullptr != op->inputIndexes() && opMask[i]) {
                for (int j=0; j<op->inputIndexes()->size(); ++j) {
                    auto index = op->inputIndexes()->data()[j];
                    if (tensorMask[index] != 2) {
                        tensorMask[index] = 1;
                    }
                }
            }
        }
    } while (change);

    for (int i=0; i<opMask.size(); ++i) {
        if (opMask[i] > 0) {
            ops.emplace_back(net->oplists()->GetAs<Op>(i));
        }
    }
}

static vector<Schedule::PipelineInfo> _scheduleUnit(const Net* net, const ScheduleConfig& configs,
                                                    const vector<shared_ptr<Tensor>>& allTensors) {
    vector<Schedule::PipelineInfo> oplists;
    vector<const Op*> ops;
    generateScheduleGraph(ops, net, configs, allTensors);
    initPipelineInfosFromOps(oplists, ops, allTensors);
    return oplists;
}

bool Schedule::schedule(ScheduleInfo& scheduleInfo, const Net* net, const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtimeInfo) {
    if (nullptr == net->oplists()) {
        MNN_PRINT("Empty net for schedule\n");
        return false;
    }
    if (scheduleInfo.defaultBackend.get() == nullptr && scheduleInfo.allTensors.empty()) {
        // Const not init, init it
        BackendConfig defaultConfig;
        defaultConfig.flags = 4;
        scheduleInfo.defaultBackend.reset(runtimeInfo.second->onCreate(&defaultConfig));
        ErrorCode code = NO_ERROR;
        initConstTensors(scheduleInfo.allTensors, net, scheduleInfo.defaultBackend.get(), code);
        if (NO_ERROR != code) {
            MNN_ERROR("Schedule Const init errorcode = %d\n", code);
            return false;
        }
    }
    bool valid = _setUpTensorInfo(scheduleInfo.allTensors, net);
    scheduleInfo.validForResize = valid;
    std::vector<std::shared_ptr<Tensor>>& allTensors = scheduleInfo.allTensors;
    std::vector<std::pair<Backend::Info, std::vector<Schedule::PipelineInfo>>> result;

    for (auto& config : configs) {
        Backend::Info compute;
        compute.type      = getApprociateType(config);
        compute.numThread = config.numThread;
        if(config.type == MNN_FORWARD_AUTO) {
            if(compute.type == MNN_FORWARD_OPENCL || compute.type == MNN_FORWARD_METAL) {
                // AUTO set default gpu-mode MNN_GPU_TUNING_FAST
                compute.numThread = 16;
            }
        }
        compute.user      = config.backendConfig;
        auto oplists      = _scheduleUnit(net, config, allTensors);
        result.emplace_back(std::make_pair(compute, std::move(oplists)));
    }

    scheduleInfo.pipelineInfo = std::move(result);

    // get all used op's output, drop unused op, won't change op order. always insert all Input Ops
    std::vector<const Op*> oplists;
    {
        for (std::pair<Backend::Info, vector<Schedule::PipelineInfo>>& pipeline : scheduleInfo.pipelineInfo) {
            for (auto& info : pipeline.second) {
                oplists.push_back(info.op);
            }
        }
    }
    // set tensors' input/output usage by oplists info
    setInputOutputForOps(allTensors, oplists, net->usage() == Usage_INFERENCE_STATIC);

    // add output index by config info and outputName
    std::unordered_map<std::string, int> tensorNameIndexMap;
    for (int i = 0; i < net->tensorName()->size(); ++i) {
        tensorNameIndexMap[net->tensorName()->Get(i)->str()] = i;
    }
    bool userSetOutput = false;
    for (auto& config : configs) {
        userSetOutput = userSetOutput || (!config.saveTensors.empty());
        for (const auto& name : config.saveTensors) {
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                }
                scheduleInfo.outputTensor.insert(
                           std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
            } else {
                MNN_PRINT("Bad outputname: %s\n", name.c_str());
            }
        }
    }
    if (net->outputName()) {
        userSetOutput = userSetOutput || net->outputName()->size() >= 1;
        for (int i = 0; i < net->outputName()->size(); ++i) {
            std::string name = net->outputName()->Get(i)->str();
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                }
                scheduleInfo.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
            }
        }
    }
    if (scheduleInfo.outputTensor.empty()) {
        userSetOutput = false;
    }
    // add input/output tensor to schedule's input/output
    for (int index = 0; index < allTensors.size(); index++) {
        auto t = allTensors[index].get();
        auto usage = TensorUtils::getDescribe(t)->usage;
        if (usage == Tensor::InsideDescribe::INPUT) {
            scheduleInfo.inputTensors.insert(std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
        if (usage == Tensor::InsideDescribe::OUTPUT && (!userSetOutput)) {
            scheduleInfo.outputTensor.insert(
                       std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
    }
#ifndef MNN_BUILD_MINI
    for (auto iter = scheduleInfo.pipelineInfo.begin(); iter != scheduleInfo.pipelineInfo.end();) {
        auto breakIndex = GeometryComputerUtils::buildConstantTensors(iter->second);
        if (breakIndex >= 0) {
            scheduleInfo.needInputContentForShape = true;
        }
#ifdef MNN_SEPERTE_SIZE
        if (breakIndex >= 0 && (breakIndex + 1) < iter->second.size()) {
            // Split oplist
            std::vector<Schedule::PipelineInfo> fuse;
            std::vector<Schedule::PipelineInfo> separate;
            fuse.insert(fuse.begin(), iter->second.begin(), iter->second.begin() + breakIndex + 1);
            separate.insert(separate.begin(), iter->second.begin() + breakIndex + 1, iter->second.end());
            oplists.clear();
            iter->second = std::move(separate);
            iter = scheduleInfo.pipelineInfo.insert(iter, std::make_pair(iter->first, fuse));
            iter++;
            iter++;
        } else {
            iter++;
        }
#else
        iter++;
#endif
    }
#endif
    return true;
}
} // namespace MNN
