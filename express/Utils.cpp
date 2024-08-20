//
//  Utils.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Utils.hpp"
#include <map>
#include <set>
#include <stack>
#include <MNN/expr/ExecutorScope.hpp>
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/Session.hpp"
#include "core/MNNMemoryUtils.h"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace Express {
Expr::Inside::Inside(int outputSize) {
    mOutputInfos.resize(outputSize);
    mOutputTensors.resize(outputSize);
    for (int i=0; i<outputSize; ++i) {
        mOutputTensors[i] = new Tensor;
        TensorUtils::getDescribe(mOutputTensors[i])->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
    }
}
Expr::Inside::Inside(Tensor* tensor, bool own) {
    mOutputInfos.resize(1);
    mOutputTensors.resize(1);
    mOutputTensors[0] = tensor;
    Utils::copyTensorToInfo(&mOutputInfos[0], tensor);
    mOutputInfos[0].syncSize();
    mOwnTensor = own;
}

Expr::Inside::~Inside() {
    if (mOwnTensor) {
        for (auto t : mOutputTensors) {
            delete t;
        }
    }
    if (nullptr != mHostTensor) {
        delete mHostTensor;
    }
}


#define CONVERT(src, dst, f)\
if (f == src) return dst;

int Utils::convertFormat(Dimensionformat format) {
    CONVERT(NCHW, MNN_DATA_FORMAT_NCHW, format);
    CONVERT(NHWC, MNN_DATA_FORMAT_NHWC, format);
    CONVERT(NC4HW4, MNN_DATA_FORMAT_NC4HW4, format);
    return MNN_DATA_FORMAT_UNKNOWN;
}

DataType Utils::convertDataType(halide_type_t type) {
    if (type.code == halide_type_float) {
        return DataType_DT_FLOAT;
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        return DataType_DT_UINT8;
    }
    if (type.code == halide_type_int && type.bits == 8) {
        return DataType_DT_INT8;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return DataType_DT_INT32;
    }
    return DataType_DT_INVALID;
}
halide_type_t Utils::revertDataType(DataType dataType) {
    CONVERT(DataType_DT_FLOAT, halide_type_of<float>(), dataType);
    CONVERT(DataType_DT_INT32, halide_type_of<int32_t>(), dataType);
    CONVERT(DataType_DT_INT64, halide_type_of<int32_t>(), dataType);
    CONVERT(DataType_DT_UINT8, halide_type_of<uint8_t>(), dataType);
    CONVERT(DataType_DT_INT8, halide_type_of<int8_t>(), dataType);
    CONVERT(DataType_DT_HALF, halide_type_of<float>(), dataType);
    CONVERT(DataType_DT_BFLOAT16, halide_type_t(halide_type_bfloat, 16), dataType);
    return halide_type_of<float>();
}
Express::Dimensionformat Utils::revertFormat(int format) {
    CONVERT(MNN_DATA_FORMAT_NCHW, Express::NCHW, format);
    CONVERT(MNN_DATA_FORMAT_NHWC, Express::NHWC, format);
    CONVERT(MNN_DATA_FORMAT_NC4HW4, Express::NC4HW4, format);
    return NCHW;
}
void Utils::copyInfoToTensor(Tensor* dest, const Variable::Info* source) {
    if (nullptr == source) {
        dest->buffer().dimensions = 0;
        return;
    }
    for (int i = 0; i < source->dim.size(); ++i) {
        dest->setLength(i, source->dim[i]);
    }
    dest->buffer().dimensions                       = (int)source->dim.size();
    dest->buffer().type                             = source->type;
    TensorUtils::getDescribe(dest)->dimensionFormat = (MNN_DATA_FORMAT)Utils::convertFormat(source->order);
    TensorUtils::setLinearLayout(dest);
}
void Utils::copyTensorToInfo(Variable::Info* shape, const Tensor* tensor) {
    shape->type  = tensor->getType();
    shape->dim   = tensor->shape();
    shape->size  = tensor->elementSize();
    shape->order = Utils::revertFormat(TensorUtils::getDescribe(tensor)->dimensionFormat);
}
bool Utils::allocMemoryForHostTensor(Tensor* dest) {
    if (nullptr != dest->buffer().host) {
        return true;
    }
    if (TensorUtils::getDescribe(dest)->memoryType != Tensor::InsideDescribe::MEMORY_HOST) {
        return false;
    }
    auto size = dest->usize();
    dest->buffer().host = (uint8_t*)MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT);
    return dest->buffer().host != nullptr;
}
bool Utils::releaseMemoryForHostTensor(Tensor* dest) {
    if (nullptr == dest->buffer().host) {
        return true;
    }
    if (TensorUtils::getDescribe(dest)->memoryType != Tensor::InsideDescribe::MEMORY_HOST) {
        return false;
    }
    MNNMemoryFreeAlign(dest->buffer().host);
    dest->buffer().host = nullptr;
    return true;
}
Tensor* Utils::getTensor(VARP var) {
    return (Tensor*)(var->getTensor());
}
EXPRP Utils::makeRaster(const std::vector<VARP>& vars, const std::vector<int>& regions, const std::vector<int>& shape, halide_type_t dataType, MNN_DATA_FORMAT format) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = OpType_Raster;
    auto extra = new ExtraT;
    // set shape
    std::unique_ptr<AttributeT> shapeAttr(new AttributeT);
    shapeAttr->key = "shape";
    shapeAttr->list.reset(new ListValueT);
    shapeAttr->list->i = shape;
    extra->attr.push_back(std::move(shapeAttr));
    // set region
    std::unique_ptr<AttributeT> regionAttr(new AttributeT);
    regionAttr->key = "region";
    regionAttr->list.reset(new ListValueT);
    regionAttr->list->i = regions;
    extra->attr.push_back(std::move(regionAttr));
    // set data type
    if (format != MNN_DATA_FORMAT_UNKNOWN) {
        {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = "code";
            attr->i = dataType.code;
            extra->attr.push_back(std::move(attr));
        }
        {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = "bits";
            attr->i = dataType.bits;
            extra->attr.push_back(std::move(attr));
        }
        {
            std::unique_ptr<AttributeT> attr(new AttributeT);
            attr->key = "format";
            attr->i = (int)format;
            extra->attr.push_back(std::move(attr));
        }
    }
    op->main.type = OpParameter_Extra;
    op->main.value = extra;
    auto expr = Expr::create(std::move(op), vars);
    return expr;
}

void* Executor::ComputeCache::mapOutput(int offset, Tensor* dest) {
    auto tensor = mSession->getTensor(offset);
    auto des = TensorUtils::getDescribe(tensor);
    if (0 == tensor->deviceId() && des->quantAttr.get() == nullptr) {
        auto ptr =  tensor->host<void>();
        Utils::releaseMemoryForHostTensor(dest);
        TensorUtils::getDescribe(dest)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        dest->buffer().host = (uint8_t*)ptr;
        //MNN_ASSERT(nullptr != ptr);
        return ptr;
    }
    if (0 == tensor->usize()) {
        return nullptr;
    }
    Utils::allocMemoryForHostTensor(dest);
    tensor->copyToHostTensor(dest);
    MNN_ASSERT(nullptr != dest->host<void>());
    return dest->host<void>();
}

void Executor::ComputeCache::setShapeDirty() {
    mShapeDirty = true;
}

void Executor::ComputeCache::setContentDirty() {
    mContentDirty = true;
}

Executor::ComputeCache::~ComputeCache() {
    mSession = nullptr;
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
    gInstanceCount--;
    FUNC_PRINT(gInstanceCount);
#endif
}
ErrorCode Executor::ComputeCache::compute() {
    std::stack<ComputeCache*> dfsStack;
    std::set<ComputeCache*> visited;
    dfsStack.push(this);
    ErrorCode code = NO_ERROR;
    auto globalExecutor = ExecutorScope::Current();
    auto debug = globalExecutor->getDebugTools();
    while (!dfsStack.empty()) {
        //printf("stcak = %d\n", dfsStack.size());
        auto cache = dfsStack.top();
        for (auto& c : cache->mInputInside) {
            if (c->mContentDirty) {
                return CALL_BACK_STOP;
            }
        }
        if (cache->mShapeDirty) {
            auto code = cache->resize();
            if (NO_ERROR != code) {
                cache->mShapeDirty = true;
                return code;
            }
        }
        if (!cache->mContentDirty) {
            visited.insert(cache);
            dfsStack.pop();
            continue;
        }
        auto hasUnvisitInput = [&] () {
            for (auto c : cache->mInputs) {
                if (visited.find(c.get()) == visited.end()) {
                    return true;
                }
            }
            return false;
        };
        if (hasUnvisitInput()) {
            for (auto c : cache->mInputs) {
                dfsStack.push(c.get());
            }
        } else {
            visited.insert(cache);
            dfsStack.pop();
            if (debug->after != nullptr && debug->before != nullptr) {
                code = cache->mSession->runWithCallBack(debug->before, debug->after);
            } else {
                code = cache->mSession->run();
            }
            if (NO_ERROR != code) {
                return code;
            }
            cache->mContentDirty = false;
        }
    }
    return NO_ERROR;
}
ErrorCode Executor::ComputeCache::resizeImpl() {
    mShapeDirty = false;
    mSession->setNeedResize();
    mSession->resize();
    mContentDirty = true;
    return NO_ERROR;
}
ErrorCode Executor::ComputeCache::resize() {
    std::stack<ComputeCache*> dfsStack;
    std::set<ComputeCache*> visited;
    dfsStack.push(this);
    while (!dfsStack.empty()) {
        auto cache = dfsStack.top();
        if (!cache->mShapeDirty) {
            visited.insert(cache);
            dfsStack.pop();
            continue;
        }
        for (auto& c : cache->mInputInside) {
            if (c->mInfoDirty) {
                return CALL_BACK_STOP;
            }
        }
        auto hasUnvisitInput = [&] () {
            for (auto c : cache->mInputs) {
                if (visited.find(c.get()) == visited.end()) {
                    return true;
                }
            }
            return false;
        };
        if (hasUnvisitInput()) {
            for (auto c : cache->mInputs) {
                dfsStack.push(c.get());
            }
        } else {
            visited.insert(cache);
            dfsStack.pop();
            auto code = cache->resizeImpl();
            if (code != NO_ERROR) {
                return code;
            }
        }
    }
    return NO_ERROR;
}
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
int Executor::ComputeCache::gInstanceCount = 0;
#endif


} // namespace Express
} // namespace MNN
