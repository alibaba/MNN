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
#include "shape/SizeComputer.hpp"
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
        // Search Backend Exclude MNN_FORWARD_CPU
        for (int i = 1; i < MNN_FORWARD_ALL; ++i) {
            if (MNNGetExtraRuntimeCreator((MNNForwardType)i) != nullptr) {
                type = (MNNForwardType)i;
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

static bool _setUpTensorInfo(std::vector<std::shared_ptr<Tensor>>& allTensors, const Net* net) {
    bool valid    = true;
    auto& tensors = allTensors;
    tensors.resize(net->tensorName()->size());

    if (net->usage() == Usage_INFERENCE_STATIC) {
        // static model will set all tensors' shape
        auto describes = net->extraTensorDescribe();
        std::vector<const TensorDescribe*> des(tensors.size());
        for (int i = 0; i < describes->size(); i++) {
            int index  = describes->GetAs<TensorDescribe>(i)->index();
            des[index] = describes->GetAs<TensorDescribe>(i);
        }
        for (int i = 0; i < tensors.size(); ++i) {
            auto blob = des[i]->blob();
            if (auto idims = blob->dims()) {
                tensors[i].reset(new Tensor(idims->size()));
                auto& tb = tensors[i]->buffer();
                for (int d = 0; d < idims->size(); d++) {
                    tb.dim[d].extent = idims->Get(d);
                }
            } else {
                tensors[i].reset(new Tensor(1));
            }
            tensors[i]->setType(blob->dataType());
        }
        for (int i = 0; i < tensors.size(); ++i) {
            auto blob                                                   = des[i]->blob();
            TensorUtils::getDescribe(tensors[i].get())->dimensionFormat = blob->dataFormat();
            if (auto regions = des[i]->regions()) {
                auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
                TensorUtils::getDescribe(tensors[i].get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
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
        for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
            auto op = net->oplists()->GetAs<Op>(opIndex);
            if (OpType_Const == op->type()) {
                MNN_ASSERT(nullptr != op->outputIndexes());
                auto index                                            = op->outputIndexes()->data()[0];
                TensorUtils::getDescribe(tensors[index].get())->usage = Tensor::InsideDescribe::CONSTANT;
            }
        }
    } else {
        // Dynamic Model just set input tensor's shape
        valid = initTensors(tensors, net);
    }
    return valid;
}

static void generateScheduleGraph(vector<const Op*>& ops, const Net* net, const ScheduleConfig& configs,
                                  const vector<shared_ptr<Tensor>>& allTensors) {
    if (configs.path.inputs.empty() && configs.path.outputs.empty()) {
        // Use Default Linear schedule
        ops.clear();
        ops.reserve(net->oplists()->size());
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (op->type() != OpType_Input) {
                ops.emplace_back(op);
            }
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
    if (configs.mode == ScheduleConfig::Path::Mode::Tensor) {
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

Schedule::ScheduleInfo Schedule::schedule(const Net* net, const std::vector<ScheduleConfig>& configs) {
    std::vector<std::shared_ptr<Tensor>> allTensors;

    ScheduleInfo schedule;
    if (nullptr == net->oplists()) {
        MNN_PRINT("Error net for schedule\n");
        return schedule;
    }
    bool valid              = _setUpTensorInfo(allTensors, net);
    schedule.validForResize = valid;

    std::vector<std::pair<Backend::Info, std::vector<Schedule::PipelineInfo>>> result;

    for (auto& config : configs) {
        Backend::Info compute;
        compute.type      = getApprociateType(config);
        compute.numThread = config.numThread;
        compute.user      = config.backendConfig;
        auto oplists      = _scheduleUnit(net, config, allTensors);
        result.emplace_back(std::make_pair(compute, std::move(oplists)));
    }

    schedule.pipelineInfo = std::move(result);

    // get all used op's output, drop unused op, won't change op order. always insert all Input Ops
    std::vector<const Op*> oplists;
    {
        for (std::pair<Backend::Info, vector<Schedule::PipelineInfo>>& pipeline : schedule.pipelineInfo) {
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
    for (auto& config : configs) {
        for (const auto& name : config.saveTensors) {
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                } else {
                    schedule.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
                }
            } else {
                MNN_PRINT("Bad outputname: %s\n", name.c_str());
            }
        }
    }
    if (net->outputName()) {
        for (int i = 0; i < net->outputName()->size(); ++i) {
            std::string name = net->outputName()->Get(i)->str();
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                } else {
                    schedule.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
                }
            }
        }
    }
    // add input/output tensor to schedule's input/output
    for (int index = 0; index < allTensors.size(); index++) {
        auto t = allTensors[index].get();
        auto usage = TensorUtils::getDescribe(t)->usage;
        if (usage == Tensor::InsideDescribe::INPUT) {
            schedule.inputTensors.insert(std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
        if (usage == Tensor::InsideDescribe::OUTPUT) {
            schedule.outputTensor.insert(
                       std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
    }
    // move tensors to schedule
    for (auto& t : allTensors) {
        schedule.allTensors.emplace_back(std::make_pair(0, std::move(t)));
    }
    return schedule;
}
} // namespace MNN
