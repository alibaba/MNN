//
//  InitNet.cpp
//  MNN
//
//  Created by MNN on 2018/09/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "InitNet.hpp"
#include "core/TensorUtils.hpp"
#include <unordered_map>
#include "core/OpCommonUtils.hpp"
#include "half.hpp"

namespace MNN {
bool needComputeOp(const Op* op) {
    if (op->type() != OpType_Input && op->type() != OpType_Const && op->type() != OpType_TrainableParam) {
        return true;
    }
    return false;
}
bool computeShapeForBlob(const Blob* parameter, Tensor* output) {
    bool zeroShape = false;
    if (parameter->dims() != nullptr) {
        output->buffer().dimensions = parameter->dims()->size();
        for (int i = 0; i < output->buffer().dimensions; i++) {
            output->buffer().dim[i].extent = parameter->dims()->Get(i);
            if (output->length(i) <= 0) {
                zeroShape = true;
            }
        }
    } else {
        output->buffer().dimensions = 0;
    }
    if (parameter->dataType() == DataType_DT_HALF) {
        output->setType(DataType_DT_FLOAT);
    } else {
        output->setType(parameter->dataType());
    }
    TensorUtils::getDescribe(output)->dimensionFormat = parameter->dataFormat();
    TensorUtils::setLinearLayout(output);
    return zeroShape;
}

bool initConstTensors(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net, Backend* defaultBackend, ErrorCode& code, FileLoader* external) {
    bool valid    = true;
    tensors.resize(net->tensorName()->size());
    // Set up const
    for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
        auto op = net->oplists()->GetAs<Op>(opIndex);
        if (OpType_Const == op->type() || OpType_TrainableParam == op->type()) {
            MNN_ASSERT(nullptr != op->outputIndexes());
            auto index = op->outputIndexes()->data()[0];
            tensors[index].reset(new Tensor);
            TensorUtils::getDescribe(tensors[index].get())->index = index;
            auto parameter = op->main_as_Blob();
            auto output    = tensors[index].get();
            if (op->type() == OpType_TrainableParam) {
                TensorUtils::getDescribe(output)->usage = Tensor::InsideDescribe::TRAINABLE;
            }
            bool zeroShape = computeShapeForBlob(parameter, output);
            TensorUtils::getDescribe(output)->usage = Tensor::InsideDescribe::CONSTANT;
            TensorUtils::getDescribe(output)->isMutable = false;
            TensorUtils::getDescribeOrigin(output)->setBackend(defaultBackend);
            //MNN_PRINT("Const tensor %p is %p bn\n", output, defaultBackend);
            if (zeroShape) {
                continue;
            }
            auto res = defaultBackend->onAcquireBuffer(output, Backend::STATIC);
            if (!res) {
                code = OUT_OF_MEMORY;
                return false;
            }
            if (parameter->dataType() == DataType_DT_HALF) {
                if (nullptr == parameter->uint8s()) {
                    // Error half const
                    code = INVALID_VALUE;
                    return false;
                }
                auto outputPtr = output->host<float>();
                auto size = output->elementSize();
                half_float::half* src = nullptr;
                std::unique_ptr<half_float::half[]> tmp;
                if (USE_EXTERNAL_DATA(parameter)) {
                    tmp.reset((new half_float::half[size]));
                    src = tmp.get();
                    OpCommonUtils::loadExternalDatas(external, {reinterpret_cast<char*>(src)}, parameter->external()->data());
                } else {
                    src = (half_float::half*)parameter->uint8s()->data();
                }
                for (int i=0; i<size; ++i) {
                    outputPtr[i] = src[i];
                }
            } else {
                OpCommonUtils::loadBlobData(external, op, output->host<char>(), output->size());
            }
        } else {
            if (nullptr != op->outputIndexes()) {
                for (int i=0; i<op->outputIndexes()->size(); ++i) {
                    auto index = op->outputIndexes()->data()[i];
                    if (nullptr == tensors[index].get()) {
                        continue;
                    }
                    auto des = TensorUtils::getDescribe(tensors[index].get());
                    if (des->usage == Tensor::InsideDescribe::CONSTANT) {
                        des->usage = Tensor::InsideDescribe::TRAINABLE;
                    }
                }
            }
        }
    }
    return valid;
}

bool initTensors(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net) {
    bool valid    = true;
    auto describes = net->extraTensorDescribe();
    std::vector<const TensorDescribe*> des(tensors.size());
    for (int i=0; i<tensors.size(); ++i) {
        // Init all tensor except for const
        if (tensors[i].get() == nullptr) {
            tensors[i].reset(new Tensor);
            TensorUtils::getDescribe(tensors[i].get())->index = i;
            // MNN_PRINT("initTensors create tensor:%p, index:%d, backend:%d\n", tensors[i].get(), i, TensorUtils::getDescribe(tensors[i].get())->backend);
        }
    }
    if (describes) {
        for (int i = 0; i < describes->size(); i++) {
            int index  = describes->GetAs<TensorDescribe>(i)->index();
            des[index] = describes->GetAs<TensorDescribe>(i);
        }
    }
    for (int i = 0; i < tensors.size(); ++i) {
        if (des[i] != nullptr && des[i]->quantInfo()) {
            TensorUtils::getDescribe(tensors[i].get())->quantAttr.reset(new QuantAttr);
            auto quant   = TensorUtils::getDescribe(tensors[i].get())->quantAttr.get();
            quant->scale =  des[i]->quantInfo()->scale();
            quant->zero  =  des[i]->quantInfo()->zero();
            quant->min   =  des[i]->quantInfo()->min();
            quant->max   =  des[i]->quantInfo()->max();
            // Don't copy datatype, it can be set by backend
        }
    }
    // Set Input Tensor, if the type of input is not the same with ExtraTensorDescribe, use input parameter
    for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
        auto op = net->oplists()->GetAs<Op>(opIndex);
        if (OpType_Input == op->type()) {
            MNN_ASSERT(nullptr != op->outputIndexes());
            MNN_ASSERT(op->outputIndexes()->size() == 1);
            auto index      = op->outputIndexes()->data()[0];
            auto tensor     = tensors[index].get();
            auto& tb        = tensor->buffer();
            auto inputParam = op->main_as_Input();
            if (auto idims = inputParam->dims()) {
                for (int i = 0; i < idims->size(); ++i) {
                    int extent = idims->data()[i];
                    // dim-0 is batch(when input batch is -1, set it to be 1, ignore other dim)
                    if (i == 0 && extent == -1) {
                        extent = 1;
                    }
                    if (extent < 0) {
                        valid = false;
                    }
                    tb.dim[i].extent = extent;
                }
                tb.dimensions = idims->size();
            } else {
                tb.dimensions = 0;
            }
            tensor->setType(inputParam->dtype());
            TensorUtils::getDescribe(tensor)->dimensionFormat = inputParam->dformat();
            TensorUtils::setLinearLayout(tensor);
        }
    }
    if (net->usage() != Usage_INFERENCE_STATIC) {
        return valid;
    }
    // static model will set all tensors' shape
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
void initPipelineInfosFromOps(std::vector<Schedule::OpCacheInfo>& infos, std::vector<const Op*>& ops, const std::vector<std::shared_ptr<Tensor>>& allTensors) {
    for (const Op* op : ops) {
        // MNN_PRINT("initPipelineInfosFromOps, op type:%s, op name:%s\n", EnumNameOpType(op->type()), op->name()->c_str());

        Schedule::OpCacheInfo opInfo;
        opInfo.op = op;
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                opInfo.outputs.push_back(allTensors[data[j]].get());
            }
        }
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                opInfo.inputs.push_back(allTensors[data[j]].get());
            }
        }
        if (needComputeOp(op)) {
            infos.emplace_back(std::move(opInfo));
        }
    }
}

void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors, const std::vector<const Op*>& ops, bool isStatic) {
    std::set<int> inputIndexes;
    std::set<int> outputIndexes;
    // 0. deal virtual tensor for static model:
    // when : A (Any_Op) -----> B (Raster_Op)
    // the tensor will be like below:
    //      A_outputs : a_tensor
    //      B_inputs  : b_tensor (virtual)
    //      b_tensor.describe.origin = a_tensor_ptr
    // b_tensor is not a InputTensot, a_tensor is not a OutputTensor
    // so add b_tensor to OutputIndexes, a_tensor to InputIndexes.
    if (isStatic) {
        std::unordered_map<Tensor*, int> tensorMap;
        for (int index = 0; index < allTensors.size(); index++) {
            tensorMap.insert(std::make_pair(allTensors[index].get(), index));
        }
        for (int index = 0; index < allTensors.size(); index++) {
            auto des = TensorUtils::getDescribe(allTensors[index].get());
            for (int i = 0; i < des->regions.size(); i++) {
                outputIndexes.insert(index);
                MNN_ASSERT(tensorMap.find(des->regions[i].origin) != tensorMap.end());
                int x = tensorMap[des->regions[i].origin];
                inputIndexes.insert(x);
            }
        }
    }
    // 1. insert all output/input index in outputIndexes/inputIndexes
    for (auto op : ops) {
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                outputIndexes.insert(data[j]);
            }
        }
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                inputIndexes.insert(data[j]);
            }
        }
        MNN_ASSERT(OpType_Input != op->type());
    }
    // 2. the index in outputIndexes/inputIndexed but not in inputIndexes/outputIndexes is output/input
    std::set<int> input;
    std::set<int> output;
    std::set_difference(outputIndexes.begin(), outputIndexes.end(), inputIndexes.begin(), inputIndexes.end(),
                        std::inserter(output, output.begin()));
    std::set_difference(inputIndexes.begin(), inputIndexes.end(), outputIndexes.begin(), outputIndexes.end(),
                        std::inserter(input, input.begin()));
    // 3. set usage for Tensor by index
    for (auto index : input) {
        auto des = TensorUtils::getDescribe(allTensors[index].get());
        if (des->usage == Tensor::InsideDescribe::CONSTANT || des->usage == Tensor::InsideDescribe::TRAINABLE) {
            continue;
        }
        des->usage = Tensor::InsideDescribe::INPUT;
    }
    for (auto index : output) {
        auto des = TensorUtils::getDescribe(allTensors[index].get());
        if (des->usage == Tensor::InsideDescribe::NORMAL) {
            des->usage = TensorUsage::OUTPUT;
        }
    }
}

void initPipelineInfosFromNet(std::vector<Schedule::OpCacheInfo>& infos, const Net* net, std::vector<std::shared_ptr<Tensor>>& allTensors) {
    std::vector<const Op*> ops;
    for (int i = 0; i < net->oplists()->size(); i++) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (needComputeOp(op)) {
            ops.push_back(op);
        }
    }
    initPipelineInfosFromOps(infos, ops, allTensors);
    setInputOutputForOps(allTensors, ops);
}
} // namespace MNN
