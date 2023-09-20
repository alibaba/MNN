//
//  TRTBackend.cpp
//  MNN
//
//  Created by MNN on 2020/07/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTBackend.hpp"
#include <NvInfer.h>
#include <backend/cpu/compute/CommonOptFunction.h>
#include <core/Macro.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <core/TensorUtils.hpp>
#include <map>
#include <mutex>
#include <thread>
#include "TRTBackendTest.hpp"
#include "TRTPlugin.hpp"
#ifdef MNN_TRT_DYNAMIC
#include "TRTDynLoad.hpp"
#endif
#include <string.h>

#ifdef USE_TRT_PROFILER
    #include "TRTProfiler.hpp"
#endif

namespace MNN {
static size_t realSize(const Tensor* t) {
    auto res = t->getType().bytes();
    for (int i = 0; i < t->dimensions(); ++i) {
        res *= t->length(i);
    }
    return res;
}
TRTRuntime::TRTRuntime(const Backend::Info& info) {
    mInfo = info;

    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
    if (nullptr != mInfo.user) {
        precision = mInfo.user->precision;
        power     = mInfo.user->power;
    }

    mPrecision = precision;
}

TRTRuntime::~TRTRuntime() {
}

Backend* TRTRuntime::onCreate(const BackendConfig* config) const {
    return new TRTBackend(this);
}

void TRTRuntime::onGabageCollect(int level) {
    // nothing now
}

static TRTLogger mLogger;

#ifdef USE_TRT_PROFILER
    static SimpleProfiler profiler("MNN TRT Performance");
#endif

std::map<OpType, TRTBackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<OpType, TRTBackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<OpType, TRTBackend::Creator*>; });
    return creators;
};

bool TRTBackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

ErrorCode tensorConvert(const Tensor* input, const Tensor* output) {
    MNNCPUCopyBuffer(input, output);
    return NO_ERROR;
}

void TRTBackend::init() {
    if (mRuntime == nullptr) {
        mRuntime = createInferRuntime(mLogger);
        if (mRuntime == nullptr) {
            MNN_PRINT("createInferRuntime error !!! \n");
        }
    }

    if (mBuilder == nullptr) {
        mBuilder = createInferBuilder(mLogger);
        if (mBuilder == nullptr) {
            MNN_PRINT("createInferRuntime error !!! \n");
        }
    }

    if (mNetwork == nullptr) {
        mNetwork = mBuilder->createNetwork();
        if (mNetwork == nullptr) {
            MNN_PRINT("createInferRuntime error !!! \n");
        }
    }

    MNN_PRINT("Initialized MNN TRT backend.");	
}

void TRTBackend::unInit() {
    for (auto p : mInOutbuffers) {
        cudaFree(p);
    }
    mInOutbuffers.clear();
    if (mContext != nullptr) {
        mContext->destroy();
        mContext = nullptr;
    }
    if (mEngine != nullptr) {
        mEngine->destroy();
        mEngine = nullptr;
    }
    if (mRuntime != nullptr) {
        mRuntime->destroy();
        mRuntime = nullptr;
    }
    if (mBuilder != nullptr) {
        mBuilder->destroy();
        mBuilder = nullptr;
    }
    if (mNetwork != nullptr) {
        mNetwork->destroy();
        mNetwork = nullptr;
    }
    
    mTensorMaps.clear();
    mInputs.clear();
    mOutputs.clear();
}

TRTBackend::TRTBackend(const TRTRuntime* runtime) : Backend(MNN_FORWARD_USER_1) {
    mTRTRuntime = runtime;
    mPrecision  = mTRTRuntime->mPrecision;
    init();
}

TRTBackend::~TRTBackend() {
#ifdef USE_TRT_PROFILER
    std::cout << profiler << std::endl;
#endif
    unInit();
}

// Create Execution
Execution* TRTBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) {
#ifdef TRT_LOG
    printf("TRTBackend::onCreate in type:%s\n", EnumNameOpType(op->type()));
#endif
    auto map  = gCreator();
    auto iter = map->find(op->type());

    if (iter == map->end()) {
        if (op->type() == OpType_Raster) {
            MNN_PRINT("[NPU] Don't support type %d\n", op->type());
            return nullptr;
        }
        MNN_PRINT("[NPU] Don't support type %d, %s\n", op->type(), op->name()->c_str());
        MNN_ASSERT(false);
        return nullptr;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    if (nullptr == exe) {
        MNN_PRINT("[NPU] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        return nullptr;
    }

    return exe;
}

void TRTBackend::onExecuteBegin() const {
#ifdef TRT_LOG
    printf("onExecuteBegin in\n");
#endif
    mContext->enqueue(1, (void**)mInOutbuffers.data(), nullptr, nullptr);
}

void TRTBackend::onExecuteEnd() const {
}

Backend::MemObj* TRTBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    int currentIndex = mTensorMaps.size();
    mTensorMaps.insert(std::make_pair(tensor, std::make_pair(nullptr, currentIndex)));
    bool isInput = TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::INPUT;
    if (isInput) {
        char name[128];
        sprintf(name, "I%d", (int)mInputs.size());
        // FIXME: Compute input info
        auto shape = tensor->shape();
        Dims dims;
        auto type    = tensor->getType();
        auto trtType = nvinfer1::DataType::kFLOAT;
        dims.nbDims = shape.size();
        ::memcpy(dims.d, shape.data(), dims.nbDims * sizeof(int32_t));
        auto input                = mNetwork->addInput(name, trtType, dims);
        mTensorMaps[tensor].first = input;
        mInputs.insert(std::make_pair(tensor, std::make_pair(std::string(name), nullptr)));
    }
    if (TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::OUTPUT) {
        char name[128];
        sprintf(name, "O%d", mOutputs.size());
        mOutputs.insert(std::make_pair(tensor, std::make_pair(std::string(name), nullptr)));
    }
    return new Backend::MemObj;
}

bool TRTBackend::onClearBuffer() {
    return true;
}

template<typename T>
void NHWC2NCHW(const T* source, T* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i * c;
            auto dstArea = dstBatch + i;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci * area] = srcArea[ci];
            }
        }
    }
}

void TRTBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    bool isConst = (TensorUtils::getDescribe(srcTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT ||
                    TensorUtils::getDescribe(dstTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
    if (isConst) {
        // Only occurred in cpu const -> backend const
        MNN_ASSERT(mTensorMaps.find(dstTensor) != mTensorMaps.end());
        Dims dims;
        auto shape  = srcTensor->shape();
        dims.nbDims = shape.size();
        ::memcpy(dims.d, shape.data(), dims.nbDims * sizeof(int32_t));
        auto trtType = nvinfer1::DataType::kFLOAT;
        auto type    = srcTensor->getType();
        if (type != halide_type_of<float>()) {
            // Turn other type to float
            auto totalSize = srcTensor->elementSize();
            std::shared_ptr<ConvolutionCommon::Int8Common> common(new ConvolutionCommon::Int8Common);
            common->weightFloat.reset(totalSize);
            // trtType = nvinfer1::DataType::kFLOAT;
            auto dstFloat = common->weightFloat.get();
            if (type == halide_type_of<int32_t>()) {
                auto src = srcTensor->host<int32_t>();
                for (int v = 0; v < totalSize; ++v) {
                    dstFloat[v] = src[v];
                }
            } else if (type == halide_type_of<uint8_t>()) {
                auto src = srcTensor->host<uint8_t>();
                for (int v = 0; v < totalSize; ++v) {
                    dstFloat[v] = src[v];
                }
            } else if (type == halide_type_of<int8_t>()) {
                auto src = srcTensor->host<int8_t>();
                for (int v = 0; v < totalSize; ++v) {
                    dstFloat[v] = src[v];
                }
            }
            TRTWeight weight{trtType, static_cast<void*>(common->weightFloat.get()), static_cast<size_t>(totalSize)};
            auto const_layer             = mNetwork->addConstant(dims, weight.get());
            mTensorMaps[dstTensor].first = const_layer->getOutput(0);
            pushCache(common);
        } else {
            TRTWeight weight{trtType, static_cast<void*>(srcTensor->host<void>()),
                             static_cast<size_t>(srcTensor->elementSize())};
            auto const_layer             = mNetwork->addConstant(dims, weight.get());
            mTensorMaps[dstTensor].first = const_layer->getOutput(0);
        }
        return;
    }
    MNN_ASSERT(nullptr != mEngine);
#ifdef TRT_LOG
    static int index_ = 0;
    printf("TRTBackend onCopyBuffer in %d, outIdx:%d\n", index_++, output_index);
#endif

    AUTOTIME;
    auto isInputCopy = TensorUtils::getDescribe(dstTensor)->usage == Tensor::InsideDescribe::Usage::INPUT;
    if (isInputCopy) {
        MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
        auto inputIndex = mContext->getEngine().getBindingIndex(mInputs[dstTensor].first.c_str());
        if(data_format == Tensor::DimensionType::CAFFE){
            auto type    = srcTensor->getType();
            if (type == halide_type_of<int32_t>()) {
                auto totalSize = srcTensor->elementSize();
                for (int v = 0; v < totalSize; ++v) {
                    srcTensor->host<float>()[v] = float(srcTensor->host<int>()[v]);
                }
            }else if(type == halide_type_of<uint32_t>()){
                auto totalSize = srcTensor->elementSize();
                for (int v = 0; v < totalSize; ++v) {
                    srcTensor->host<float>()[v] = float(srcTensor->host<uint>()[v]);
                }
            }
            auto status = cudaMemcpy(mInOutbuffers[inputIndex], srcTensor->host<float>(), srcTensor->size(), cudaMemcpyHostToDevice);
            MNN_ASSERT(0 == status);
        }else{
            int area = dstTensor->height() * dstTensor->width();
            int b = dstTensor->batch();
            int c = dstTensor->channel();
            shared_ptr<Tensor> tmpTensor(new Tensor(dstTensor, Tensor::DimensionType::CAFFE, true)); // nchw
            NHWC2NCHW<float>(tmpTensor->host<float>(), srcTensor->host<float>(), b, c, area);
            auto type    = tmpTensor->getType();
            if (type == halide_type_of<int32_t>()) {
                auto totalSize = tmpTensor->elementSize();
                for (int v = 0; v < totalSize; ++v) {
                    tmpTensor->host<float>()[v] = float(tmpTensor->host<int>()[v]);
                }
            }else if(type == halide_type_of<uint32_t>()){
                auto totalSize = tmpTensor->elementSize();
                for (int v = 0; v < totalSize; ++v) {
                    tmpTensor->host<float>()[v] = float(tmpTensor->host<uint>()[v]);
                }
            }
            auto status = cudaMemcpy(mInOutbuffers[inputIndex], tmpTensor->host<float>(), tmpTensor->size(), cudaMemcpyHostToDevice);
            MNN_ASSERT(0 == status);
        }
    } else {
        shared_ptr<Tensor> tmpTensor(new Tensor(srcTensor, srcTensor->getDimensionType(), true)); 
        MNN_ASSERT(dstTensor->host<float>() != nullptr);
        auto outputIndex = mContext->getEngine().getBindingIndex(mOutputs[srcTensor].first.c_str());
        auto status = cudaMemcpy(tmpTensor->host<float>(), mInOutbuffers[outputIndex], tmpTensor->size(), cudaMemcpyDeviceToHost);
        MNN_ASSERT(0 == status);
        tensorConvert(tmpTensor.get(), dstTensor);
    }
}

void TRTBackend::onResizeBegin() {
#ifdef TRT_LOG
    printf("TRTBackend onResizeBegin in\n");
#endif
    unInit();
    init();
}

ErrorCode TRTBackend::onResizeEnd() {
#ifdef TRT_LOG
    printf("\n\nTRTBackend onResizeEnd in\n");
#endif
    for (auto& p : mOutputs) {
        auto iter = mTensorMaps.find(p.first);
        (iter->second.first)->setName(p.second.first.c_str());
        mNetwork->markOutput(*(iter->second.first));
    }

    MNN_ASSERT(mNetwork->getNbInputs() == mInputs.size());
    MNN_ASSERT(mNetwork->getNbOutputs() == mOutputs.size());
    MNN_ASSERT(mOutputs.size() > 0);

    // Build the engine.
    if (mTRTRuntime->mCacheBuffer == nullptr) {
        printf("not use cache buffer!!! \n");
        mBuilder->setMaxBatchSize(1);
        mBuilder->setMaxWorkspaceSize(1024 * 1024);
        if (mPrecision == BackendConfig::Precision_Low) {
            bool support_fp16 = mBuilder->platformHasFastFp16();
            FUNC_PRINT(support_fp16);
            mBuilder->setFp16Mode(support_fp16);
            // mBuilder->setHalf2Mode(false);
        }
        auto cudaEngine = mBuilder->buildCudaEngine(*mNetwork);
        MNN_ASSERT(cudaEngine != nullptr);

        IHostMemory* model = cudaEngine->serialize();

        if (mEngine == nullptr) {
            mEngine = mRuntime->deserializeCudaEngine(model->data(), model->size(), &Singleton<TRTPlugin>::Global());
        }
        mTRTRuntime->mModel = model;
        if (cudaEngine != nullptr) {
            cudaEngine->destroy();
            cudaEngine = nullptr;
        }
    } else {
        printf("use cache buffer!!! \n");
        if (mEngine == nullptr) {
            mEngine = mRuntime->deserializeCudaEngine(mTRTRuntime->mCacheBuffer, mTRTRuntime->mCacheSize, &Singleton<TRTPlugin>::Global());
        }
    }
    if (mEngine == nullptr) {
        MNN_PRINT("deserializeCudaEngine error !!! \n");
    }
    if (mContext == nullptr) {
        mContext = mEngine->createExecutionContext();
        if (mContext == nullptr) {
            MNN_PRINT("createExecutionContext error !!! \n");
        }
    }
#ifdef USE_TRT_PROFILER
    mContext->setProfiler(&profiler);
#endif

    const ICudaEngine& engine = mContext->getEngine();
    MNN_ASSERT(engine.getNbBindings() == mInputs.size() + mOutputs.size());
    mInOutbuffers.resize(engine.getNbBindings());
    for (auto& iter : mInputs) {
        auto inputIndex = engine.getBindingIndex(iter.second.first.c_str());
        auto size       = realSize(iter.first);
        auto status     = cudaMalloc(mInOutbuffers.data() + inputIndex, size);
        MNN_ASSERT(0 == status);
    }
    for (auto& iter : mOutputs) {
        auto outputIndex = engine.getBindingIndex(iter.second.first.c_str());
        auto size       = realSize(iter.first);
        auto status     = cudaMalloc(mInOutbuffers.data() + outputIndex, size);
        MNN_ASSERT(0 == status);
    }

    mCache.clear();
    for (auto l : mEraseLayers) {
        delete l;
    }
    mEraseLayers.clear();
    return NO_ERROR;
}

INetworkDefinition* TRTBackend::getNetwork() {
    return mNetwork;
}

void TRTBackend::cudaErrorCheck(string tag) const {
    auto error_check = cudaPeekAtLastError();
    if (0 != cudaPeekAtLastError()) {
        MNN_PRINT(" === %s === \n", tag.c_str());
        MNN_PRINT("cudaPeekAtLastError error : %s \n", cudaGetErrorName(error_check));
        MNN_PRINT("%s \n", cudaGetErrorString(error_check));
        MNN_PRINT(" ============ !!! \n");
    }
}

ITensor* TRTBackend::getTensorOps(const Tensor* input) {
    auto iter = mTensorMaps.find(input);
    if (iter != mTensorMaps.end()) {
        return (iter->second.first);
    }
    return nullptr;
}

void TRTBackend::setTensorOps(const std::vector<Tensor*>& outputs, vector<ITensor*>&& TRT_op) {
    MNN_ASSERT(outputs.size() == TRT_op.size());
    for (int i = 0; i < outputs.size(); i++) {
        mTensorMaps[outputs[i]].first = TRT_op[i];
    }
}

// Runtime Register
class TRTRuntimeCreator : public RuntimeCreator {
    virtual Runtime* onCreate(const Backend::Info& info) const {
        return new TRTRuntime(info);
    }
    virtual bool onValid(Backend::Info& info) const {
        return true;
    }
};

static bool gResistor = []() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_1, new TRTRuntimeCreator, false);
    return false;
}();
} // namespace MNN

