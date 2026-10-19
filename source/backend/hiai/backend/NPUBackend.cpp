//
//  NPUBackend.cpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBackend.hpp"
#include "HiAIDynamicLoader.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include "HiAIBackendConfig.hpp"
#include <stdlib.h>
#include <unistd.h>
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#ifdef HIAI_DEBUG
#include <android/log.h>
#include <sys/time.h>
#endif
namespace MNN {

namespace {

constexpr std::uint64_t kMaximumHiaiCacheBytes = 256ull * 1024ull * 1024ull;
constexpr char kHiaiCacheMarker[] = "MNN_HIAI_CACHE_PATH_V1\n";
constexpr char kHiaiV600CacheSuffix[] = ".hiai_v600_cache_v1.om";
constexpr char kHiaiV320CacheSuffix[] = ".hiai_v320_cache_v1.om";

std::string HiaiModelIdentity(const void* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        return std::string();
    }
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    std::uint64_t hash = kFnvOffset;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
    char identity[48] = {};
    std::snprintf(identity, sizeof(identity), "%016llx-%zu", static_cast<unsigned long long>(hash), size);
    return identity;
}

std::string HiaiCacheFile(const std::string& basePath, const std::string& modelIdentity, const char* suffix) {
    return basePath.empty() || modelIdentity.empty() ? std::string() : basePath + "." + modelIdentity + suffix;
}

std::string HiaiTemporaryCacheFile(const std::string& path) {
    static std::atomic<std::uint64_t> sequence{0};
    return path + ".tmp-" + std::to_string(getpid()) + "-" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

bool ReadHiaiCacheFile(const std::string& path, std::vector<std::uint8_t>* bytes) {
    if (path.empty() || bytes == nullptr) {
        return false;
    }
    bytes->clear();
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }
    const std::streamoff size = stream.tellg();
    if (size <= 0 || static_cast<std::uint64_t>(size) > kMaximumHiaiCacheBytes) {
        return false;
    }
    bytes->resize(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(bytes->data()), static_cast<std::streamsize>(size)));
}

bool WriteHiaiCacheFileAtomic(const std::string& path, const void* data, std::size_t size) {
    if (path.empty() || data == nullptr || size == 0 || size > kMaximumHiaiCacheBytes) {
        return false;
    }
    const std::string temporaryPath = HiaiTemporaryCacheFile(path);
    (void)std::remove(temporaryPath.c_str());
    std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    stream.close();
    if (!stream || std::rename(temporaryPath.c_str(), path.c_str()) != 0) {
        (void)std::remove(temporaryPath.c_str());
        return false;
    }
    return true;
}

bool EnsureHiaiCachePathMarker(const std::string& path) {
    if (path.empty() || access(path.c_str(), F_OK) == 0) {
        return true;
    }
    return WriteHiaiCacheFileAtomic(path, kHiaiCacheMarker, sizeof(kHiaiCacheMarker) - 1);
}

MNNHiAINativeHandleIoContext* HiaiNativeIoContext(
    const Backend::Info& info,
    bool explicitlyRequestedHiAI) {
    if (!explicitlyRequestedHiAI || info.user == nullptr ||
        info.user->sharedContext == nullptr) {
        return nullptr;
    }
    const auto* config = static_cast<const MNNHiAIBackendConfigV1*>(info.user->sharedContext);
    if (validateHiAIConfig(config) != MNN_HIAI_STATUS_SUCCESS) {
        return nullptr;
    }
    auto* context = config->nativeIoContext;
    if (context == nullptr) {
        return nullptr;
    }
    if (context->magic != MNN_HIAI_NATIVE_HANDLE_IO_MAGIC ||
        context->version != MNN_HIAI_NATIVE_HANDLE_IO_VERSION ||
        context->struct_size < sizeof(MNNHiAINativeHandleIoContext)) {
        return nullptr;
    }
    return context;
}

bool HiaiForceV320(const Backend::Info& info,
                   bool explicitlyRequestedHiAI) {
    const auto* context = HiaiNativeIoContext(info, explicitlyRequestedHiAI);
    return context != nullptr &&
           (context->reserved[MNN_HIAI_SESSION_CONTROL_INDEX] &
            MNN_HIAI_SESSION_FORCE_V320) != 0;
}

} // namespace

    void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(uint8_t));
        for (z = 0; z < depth; ++z) {
            int plane         = z / 4;
            uint8_t* dstPlane = plane * area * 4 + dst;
            int offset        = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(float));
        for (z = 0; z < depth; ++z) {
            int plane       = z / 4;
            float* dstPlane = plane * area * 4 + dst;
            int offset      = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void NHWC2NCHW(const float* source, float* dest, int b, int c, int area) {
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

    void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane             = z / 4;
            const float* srcPlane = plane * area * 4 + src;
            int offset            = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane               = z / 4;
            const uint8_t* srcPlane = plane * area * 4 + src;
            int offset              = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void NCHW2NHWC(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i;
                auto dstArea = dstBatch + i * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci] = srcArea[ci * area];
                }
            }
        }
    }

    ErrorCode tensorConvert(const Tensor* input, const Tensor* output) {
        auto ib     = input->buffer();
        auto ob     = output->buffer();
        auto source = TensorUtils::getDescribe(input)->dimensionFormat;
        auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
        if (ib.dimensions <= 1 || source == dest) {
            ::memcpy(ob.host, ib.host, input->size());
            return NO_ERROR;
        }
        if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
            MNN_ERROR("unknown data format!\nsrc: %s, dst: %s\n", EnumNameMNN_DATA_FORMAT(source), EnumNameMNN_DATA_FORMAT(dest));
            return INVALID_VALUE;
        }
        int area = 1, batch = ib.dim[0].extent, channel;
        if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
            channel = ib.dim[1].extent;
            for (int axis = 2; axis < ib.dimensions; ++axis) {
                area *= ib.dim[axis].extent;
            }
        } else {
            channel = ib.dim[ib.dimensions - 1].extent;
            for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
                area *= ib.dim[axis].extent;
            }
        }
        const int bitLength = ib.type.bytes();

        if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNUnpackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i,
                                    (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNUnpackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

        if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNPackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i, (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNPackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

       if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NHWC2NCHW((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NCHW2NHWC((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else {
            return NOT_SUPPORT;
        }

        return NO_ERROR;
    }
#ifdef HIAI_DEBUG
    bool WriteToBufferFile(ge::Buffer& buffer, std::string om_file_path)
    {
        FILE *fp;
        fp = fopen(om_file_path.c_str(), "wb");
        if (fp == NULL) {
            printf("%s open failed !!!", om_file_path.c_str());
            return false;
        }

        uint32_t write_size = (uint32_t)fwrite(buffer.data(), 1, buffer.size(), fp);
        if (write_size != buffer.size()) {
            fclose(fp);
            printf("write om file failed !!!");
            return false;
        }
        fclose(fp);
        return true;
    }

    bool WriteToOMFile(domi::ModelBufferData om_model_buff, std::string om_file_path)
    {
        FILE *fp;
        fp = fopen(om_file_path.c_str(), "wb");
        if (fp == NULL) {
            printf("%s open failed !!!", om_file_path.c_str());
            return false;
        }

        uint32_t write_size = (uint32_t)fwrite(om_model_buff.data, 1, om_model_buff.length, fp);
        if (write_size != om_model_buff.length) {
            fclose(fp);
            printf("write om file failed !!!");
            return false;
        }
        fclose(fp);
        return true;
    }
#endif

    shared_ptr<hiai::AiModelMngerClient> LoadModelSync(domi::ModelBufferData modelBufferData, string model_name)
    {
        shared_ptr<hiai::AiModelMngerClient> mngerClient = make_shared<hiai::AiModelMngerClient>();
        if (mngerClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client make_shared error.");
            return nullptr;
        }

        int ret = mngerClient->Init(nullptr);
        if (ret != 0) {
            MNN_ERROR("[NPU] Model Manager Init Failed.");
            return nullptr;
        }

        shared_ptr<hiai::AiModelBuilder> mcbuilder = make_shared<hiai::AiModelBuilder>(mngerClient);
        hiai::MemBuffer* buffer = mcbuilder->InputMemBufferCreate(modelBufferData.data, modelBufferData.length);
        if (buffer == nullptr) {
            MNN_ERROR("[NPU] create MemBuffer failed");
            return nullptr;
        }

        shared_ptr<hiai::AiModelDescription> desc = make_shared<hiai::AiModelDescription>(model_name, 4, 0, 0, 0);
        desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

        vector<shared_ptr<hiai::AiModelDescription>> model_desc;
        model_desc.push_back(desc);


        ret = mngerClient->Load(model_desc);
        if (ret != 0) {
            MNN_ERROR("[NPU] Model Load Failed.");
            mngerClient = nullptr;
        }

        mcbuilder->MemBufferDestroy(buffer);
        return mngerClient;
    }

    static inline std::map<OpType, NPUBackend::Creator*>* getCreatorMap() {
        static std::once_flag of;
        static std::map<OpType, NPUBackend::Creator*>* ret = nullptr;
        std::call_once(of, [&]() { ret = new std::map<OpType, NPUBackend::Creator*>; });
        return ret;
    }

    bool NPUBackend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_PRINT("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    NPUBackend::NPUBackend(const NPURuntime* runtime) : Backend(MNN_FORWARD_USER_0) {
        mNPURuntime = runtime;
        mPrecision  = mNPURuntime->mPrecision;
        const bool explicitlyRequestedHiAI = isExplicitHiAISession();
        mNativeIoContext = HiaiNativeIoContext(mNPURuntime->mInfo,
                                               explicitlyRequestedHiAI);
        if (mNativeIoContext != nullptr) {
            mForceV320 =
                (mNativeIoContext->reserved[MNN_HIAI_SESSION_CONTROL_INDEX] &
                 MNN_HIAI_SESSION_FORCE_V320) != 0;
        }
        if (!mForceV320 && explicitlyRequestedHiAI) {
            bool enableAutoTuning = false;
            std::string tuningCacheDirectory;
            enableAutoTuning = mNPURuntime->mHiAIOptions->autoTuning;
            tuningCacheDirectory = mNPURuntime->mHiAIOptions->acceleratorCacheDirectory;
            mHclV600Runtime = HiaiHclV600Runtime::CreateIfSupported(enableAutoTuning, tuningCacheDirectory);
        }
        if (mHclV600Runtime != nullptr) {
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                    MNN_HIAI_SESSION_HCL_V600_SELECTED;
            }
            MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: selected version=%s api=BuildV2/InitV2/RunV3\n",
                      mHclV600Runtime->version().c_str());
        } else if (!mForceV320 && explicitlyRequestedHiAI) {
            mRequiresV320SessionRebuild = true;
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                    MNN_HIAI_SESSION_HCL_V600_FAILED;
            }
            MNN_ERROR("MNN_HIAI_SESSION_REBUILD_AUDIT: HCL V600 unavailable; "
                      "the initial MNN session must be destroyed before "
                      "building V320\n");
        } else if (explicitlyRequestedHiAI) {
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                    MNN_HIAI_SESSION_V320_SELECTED;
            }
            MNN_PRINT("MNN_HIAI_SESSION_REBUILD_AUDIT: selected=V320 force=%d runtime=%s\n",
                      mForceV320 ? 1 : 0,
                      mNPURuntime->mHiaiRuntimeVersion.c_str());
        }
#ifdef HIAI_DEBUG
        // Retrieve a handle to libandroid.
        void *lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);
        // Access the native tracing functions.
        if (lib != NULL) {
            // Use dlsym() to prevent crashes on devices running Android 5.1
            // (API level 22) or lower.
            ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
                dlsym(lib, "ATrace_beginSection"));
            ATrace_endSection = reinterpret_cast<fp_ATrace_endSection>(
                dlsym(lib, "ATrace_endSection"));
            MNN_PRINT("get function ptr :%p,%p",ATrace_beginSection, ATrace_endSection);
        }
#endif
    }
    NPUBackend::~NPUBackend() {
        if (!isExplicitHiAISession()) {
            return;
        }
        bool released = releaseModelResources(true);
        if (mLibAndroid != nullptr) {
            if (dlclose(mLibAndroid) != 0) released = false;
            mLibAndroid = nullptr;
        }
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->reserved
                [MNN_HIAI_SESSION_RELEASE_STATUS_INDEX] =
                    released ? MNN_HIAI_SESSION_RELEASE_SUCCESS
                             : MNN_HIAI_SESSION_RELEASE_FAILED;
        }
        MNN_PRINT("MNN_HIAI_RELEASE_AUDIT: %s\n",
                  released ? "PASS" : "FAIL");
    }

    void NPUBackend::setNetworkInput(const std::vector<Tensor *> &inputs, const Op* op) {
       for (size_t i = 0; i < op->inputIndexes()->size(); i++) {
            auto inputIndex = op->inputIndexes()->data()[i];
            auto inputOpsIndex = inputIndex;
            if (!isExplicitHiAISession() && op->outputIndexes() != nullptr &&
                i < op->outputIndexes()->size()) {
                inputOpsIndex = op->outputIndexes()->data()[i];
            }
            Tensor *inputTensor = inputs[i];
            bool isInput = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
            if (isInput && mGrapMap.find(inputIndex) == mGrapMap.end()) {
                auto opName = string("input") + to_string(inputIndex);
                shared_ptr<hiai::op::Data> data(new hiai::op::Data(opName));
                vector<int64_t> dims;
                for(int32_t i = 0; i < inputTensor->buffer().dimensions; i++) {
                    dims.push_back(inputTensor->buffer().dim[i].extent);
                }
                ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
                if (TensorUtils::getDescribe(inputTensor)->dimensionFormat == MNN_DATA_FORMAT::MNN_DATA_FORMAT_NHWC) {
                    desc.SetFormat(ge::FORMAT_NHWC);
                }
                if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                    desc.SetDataType(ge::DT_INT32);
                }
                if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                    desc.SetDataType(ge::DT_INT64);
                }
                data->update_input_desc_x(desc);
                // map
                vector<pair<shared_ptr<ge::Operator>, string>> ops;
                ops.emplace_back(make_pair(data, ""));
                mGrapMap.insert(make_pair(inputIndex, ops));
                std::pair<int, std::vector<ge::Operator>> item(inputOpsIndex, {*data.get()});
                mInputOps.insert(item);
            }

            bool isConst = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;
            if (isConst && mGrapMap.find(inputIndex) == mGrapMap.end()) {
                auto opName = string("Const") + to_string(inputIndex);
                shared_ptr<hiai::op::Const> mConst(new hiai::op::Const(opName));
                {
                    ge::TensorPtr filter = std::make_shared<ge::Tensor>();
                    vector<int64_t> dims;
                    for(int32_t i = 0; i < inputTensor->buffer().dimensions; i++) {
                        dims.push_back(inputTensor->buffer().dim[i].extent);
                    }
                    ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                        fdesc.SetDataType(ge::DT_INT32);
                    }
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                        fdesc.SetDataType(ge::DT_INT64);
                    }
                    filter->SetTensorDesc(fdesc);
                    filter->SetData((uint8_t *)inputTensor->host<float>(), inputTensor->elementSize() * sizeof(float));
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                        filter->SetData((uint8_t *)inputTensor->host<int32_t>(), inputTensor->elementSize() * sizeof(int32_t));
                    }
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                        filter->SetData((uint8_t *)inputTensor->host<int64_t>(), inputTensor->elementSize() * sizeof(int64_t));
                    }
                    mConst->set_attr_value(filter);
                }
                vector<pair<shared_ptr<ge::Operator>, string>> ops;
                ops.emplace_back(make_pair(mConst, ""));
                mGrapMap.insert(make_pair(inputIndex, ops));
            }
        }
    }

    Execution* NPUBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {

        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        
        if (iter == map->end()) {
            MNN_ERROR("map not find !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[NPU] Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return new HiAIRejectedExecution(this, NOT_SUPPORT);
        }

        auto exe = iter->second->onCreate(inputs, outputs, op, this);

        if (nullptr == exe) {
            MNN_ERROR("nullptr == exe !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[NPU] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return new HiAIRejectedExecution(this, NOT_SUPPORT);
        }

        return exe;
    }

    void NPUBackend::onExecuteBegin() const {
        mGraphExecuted = false;
        // Pipeline stages host inputs before this hook. If no input needed a
        // fresh copy, reset the previous frame here; otherwise onCopyBuffer
        // already reset it before staging and any copy error must survive.
        if (mResetStatusOnNextExecute) {
            mExecutionStatus = NO_ERROR;
            mResetStatusOnNextExecute = false;
        } else if (mExecutionStatus != NO_ERROR) {
            // Preserve this frame's staging error for Pipeline::_enterExecute,
            // but allow a following frame with unchanged inputs to retry.
            mResetStatusOnNextExecute = true;
        }
    }
    
    ErrorCode NPUBackend::runGraphOnce() const {
        if (!mGraphExecuted) {
            mGraphExecuted = true;
            const int nativeCode = mExecutionStatus == NO_ERROR ? process(0) : -1;
            if (nativeCode != 0 && mExecutionStatus == NO_ERROR) mExecutionStatus = INVALID_VALUE;
            if (mNPURuntime->mHiAIOptions) mNPURuntime->mHiAIOptions->report(
                MNN_HIAI_STAGE_EXECUTE, mExecutionStatus, mExecutionStatus == NO_ERROR ?
                "HiAI execution completed" : "HiAI execution or input copy failed", nativeCode);
        }
        return mExecutionStatus;
    }

    void NPUBackend::onExecuteEnd() const {
        mResetStatusOnNextExecute = true;
    }

    Backend::MemObj* NPUBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
        bool isInputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        if(isInputCopy){
            mInputMap.insert(make_pair((unsigned long)tensor, mInputMap.size()));
        }
        // Don't need extra release
        return new Backend::MemObj;
    }

    bool NPUBackend::onClearBuffer() {
        return true;
    }

    void NPUBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
        HiAICopyDiagnostic diagnostic(mNPURuntime->mHiAIOptions.get(), mExecutionStatus);
#ifdef HIAI_DEBUG
        ATrace_beginSection("onCopy");
#endif
        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

        if (isConst) {
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            tmpTensor->buffer().host = srcTensor->buffer().host;
            return;
        }
        
        if (isInputCopy) {
            if (mResetStatusOnNextExecute) {
                mExecutionStatus = NO_ERROR;
                mResetStatusOnNextExecute = false;
            }
            if (mNativeIoContext != nullptr &&
                srcTensor->buffer().flags == MNN_MEMORY_AHARDWAREBUFFER) {
                // Binding happens before the small audio memcpy and before
                // Process(), so both input and output ION handles are ready
                // for this frame. The caller checks result_code immediately.
                if (!const_cast<NPUBackend*>(this)->bindNativeHandleIo()) {
                    mExecutionStatus = INPUT_DATA_ERROR;
                }
                return;
            }
            auto index = mInputMap.find((unsigned long)(const_cast<Tensor*>(dstTensor)));
            if (index == mInputMap.end() || srcTensor->buffer().host == nullptr) {
                MNN_ERROR("MNN_HIAI: invalid input copy binding\n");
                mExecutionStatus = INPUT_DATA_ERROR;
                return;
            }
            if (mHclV600Runtime != nullptr) {
                void* destination = mHclV600Runtime->inputData(index->second);
                const size_t destinationSize =
                    mHclV600Runtime->inputSize(index->second);
                if (destination == nullptr || destinationSize != srcTensor->size()) {
                    MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: input copy mismatch index=%d hcl=%lu mnn=%d\n",
                              index->second, destinationSize, srcTensor->size());
                    mExecutionStatus = INPUT_DATA_ERROR;
                    return;
                }
                memcpy(destination, srcTensor->host<void>(), destinationSize);
                return;
            }
            shared_ptr<hiai::AiTensor> input = mInputTensors[index->second];
            memcpy(input->GetBuffer(), srcTensor->host<void>(), (size_t)input->GetSize());
        } else if(isOutputCopy){
            if (mNativeIoContext != nullptr &&
                dstTensor->buffer().flags == MNN_MEMORY_AHARDWAREBUFFER &&
                mNativeIoContext->result_code ==
                    MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS) {
                // Process() already wrote directly into the output dma-buf.
                return;
            }
            int index;
            bool flag = false;
            for(index = 0; index < mMNNOutTensors.size(); index++) {
                if(mMNNOutTensors[index] == srcTensor) {
                    flag = true;
                    break;
                }
            }
            if (flag == false) {
                MNN_PRINT("MNNTensor and HIAITensor mismatch!");
                mExecutionStatus = INVALID_VALUE;
                return;
            }

            if (mHclV600Runtime != nullptr) {
                void* source = mHclV600Runtime->outputData(index);
                const size_t sourceSize = mHclV600Runtime->outputSize(index);
                if (source == nullptr || sourceSize != dstTensor->size()) {
                    MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: output copy mismatch index=%d hcl=%lu mnn=%d\n",
                              index, sourceSize, dstTensor->size());
                    mExecutionStatus = INVALID_VALUE;
                    return;
                }
                Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
                memcpy(tmpTensor->buffer().host, source, sourceSize);
                return;
            }

            shared_ptr<hiai::AiTensor> output = mOutputTensors[index];
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            memcpy(tmpTensor->buffer().host, output->GetBuffer(), (size_t)output->GetSize());
        }
#ifdef HIAI_DEBUG
        ATrace_endSection();
#endif
    }

    bool NPUBackend::releaseModelResources(bool finalRelease) {
        mGrapMap.clear();
        mOutGEOpMap.clear();
        mInputOps.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mHostInputTensors.clear();
        mHostOutputTensors.clear();
        mImageInputIndex = -1;
        mImageOutputIndex = -1;
        mBoundInputAhb = nullptr;
        mBoundOutputAhb = nullptr;
        mMNNOutTensors.clear();
        mSclipMap.clear();
        bool released = true;
        if (mHclV600Runtime != nullptr) {
            released = (finalRelease ? mHclV600Runtime->release()
                                     : mHclV600Runtime->resetModel()) &&
                       released;
            if (finalRelease) mHclV600Runtime.reset();
        }
        if (mMgrClient != nullptr) {
            const hiai::AIStatus status = mMgrClient->UnLoadModel();
            if (status != hiai::AI_SUCCESS) {
                MNN_ERROR("MNN_HIAI_RELEASE_AUDIT: UnLoadModel failed "
                          "status=%d\n", static_cast<int>(status));
                released = false;
            }
            mMgrClient.reset();
        }
        return released;
    }

    void NPUBackend::onResizeBegin() {
        if (isExplicitHiAISession()) {
            (void)releaseModelResources(false);
            return;
        }
        mGrapMap.clear();
        mOutGEOpMap.clear();
        mInputOps.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mMNNOutTensors.clear();
        mSclipMap.clear();
        if (mMgrClient != nullptr) {
            mMgrClient->UnLoadModel();
        }
    }

    ErrorCode NPUBackend::onResizeEnd() {
        if (mRequiresV320SessionRebuild) {
            mNPURuntime->mHiAIOptions->report(MNN_HIAI_STAGE_RESIZE, INVALID_VALUE, "HiAI V600 failed; recreate with V320");
            MNN_ERROR(
                "MNN_HIAI_SESSION_REBUILD_AUDIT: rejecting the initial "
                "session so the caller can recreate it with V320\n");
            return INVALID_VALUE;
        }
        const auto code = buildIRModelAndLoad();
        if (mNPURuntime->mHiAIOptions) {
            if (code == NO_ERROR) mNPURuntime->mHiAIOptions->ready();
            else mNPURuntime->mHiAIOptions->report(MNN_HIAI_STAGE_RESIZE, code, "HiAI graph build/load failed");
        }
        return code;
    }

    int NPUBackend::getInOutTensorInfo(string modelName) {
        if (mMgrClient == nullptr) {
            return -1;
        }
        int ret = mMgrClient->GetModelIOTensorDim(modelName, mInputDimension, mOutputDimension);
        if (ret != hiai::AI_SUCCESS) {
            MNN_ERROR("[NPU] Get model IO Tensor failed: %d \n", ret);
            return -1;
        }

        MNN_PRINT("mInputDimension : %lu , mOutputDimension : %lu \n", mInputDimension.size(), mOutputDimension.size());

        int input_index = 0;
        for (auto in_dim : mInputDimension)
        {
            shared_ptr<hiai::AiTensor> input = make_shared<hiai::AiTensor>();
            input->Init(&in_dim);
            mInputTensors.push_back(input);
            if (isExplicitHiAISession()) {
                const uint64_t bytes = static_cast<uint64_t>(in_dim.GetNumber()) *
                    in_dim.GetChannel() * in_dim.GetHeight() * in_dim.GetWidth() *
                    sizeof(float);
                if (mNativeIoContext != nullptr &&
                    bytes == mNativeIoContext->input_bytes) {
                    mImageInputIndex = input_index;
                }
            }
            ++input_index;
        }
        auto index = 0;
        for (auto out_dim : mOutputDimension)
        {
            shared_ptr<hiai::AiTensor> output = make_shared<hiai::AiTensor>();
            MNN_PRINT("%d HiAiTensor output DIM:%u,%u,%u,%u\n", index,
                      out_dim.GetNumber(), out_dim.GetChannel(),
                      out_dim.GetHeight(), out_dim.GetWidth());
            output->Init(&out_dim);
            mOutputTensors.push_back(output);
            if (isExplicitHiAISession()) {
                const uint64_t bytes = static_cast<uint64_t>(out_dim.GetNumber()) *
                    out_dim.GetChannel() * out_dim.GetHeight() * out_dim.GetWidth() *
                    sizeof(float);
                if (mNativeIoContext != nullptr &&
                    bytes == mNativeIoContext->output_bytes) {
                    mImageOutputIndex = index;
                }
            }
            index++;
        }
        if (isExplicitHiAISession()) {
            mHostInputTensors = mInputTensors;
            mHostOutputTensors = mOutputTensors;
        }
        index = 0;
        for (auto opMap : mOutGEOpMap) {
            for (auto tensor : opMap.second) {
                mMNNOutTensors.push_back(tensor);
                MNN_PRINT("%d MNNTensor output DIM:%d,%d,%d,%d\n", index,
                          tensor->batch(), tensor->channel(), tensor->height(), tensor->width());
                index++;
            }
        }
        return 0;
    }

    bool NPUBackend::bindNativeHandleIo() {
        if (mNativeIoContext == nullptr ||
            mNativeIoContext->input_ahardware_buffer == nullptr ||
            mNativeIoContext->output_ahardware_buffer == nullptr ||
            mImageInputIndex < 0 || mImageOutputIndex < 0) {
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->result_code =
                    MNN_HIAI_NATIVE_HANDLE_IO_INVALID_CONTEXT;
            }
            return false;
        }
        if (mBoundInputAhb == mNativeIoContext->input_ahardware_buffer &&
            mBoundOutputAhb == mNativeIoContext->output_ahardware_buffer) {
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS;
            return true;
        }

        if (mGetAhbNativeHandle == nullptr) {
            if (mLibAndroid == nullptr) {
                mLibAndroid = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
            }
            if (mLibAndroid != nullptr) {
                mGetAhbNativeHandle =
                    reinterpret_cast<const native_handle_t* (*)(const void*)>(
                        dlsym(mLibAndroid,
                              "AHardwareBuffer_getNativeHandle"));
            }
        }
        if (mGetAhbNativeHandle == nullptr) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_SYMBOL_UNAVAILABLE;
            return false;
        }

        const native_handle_t* input_handle = mGetAhbNativeHandle(
            mNativeIoContext->input_ahardware_buffer);
        const native_handle_t* output_handle = mGetAhbNativeHandle(
            mNativeIoContext->output_ahardware_buffer);
        mNativeIoContext->input_native_fd_count =
            input_handle != nullptr ? input_handle->numFds : 0;
        mNativeIoContext->output_native_fd_count =
            output_handle != nullptr ? output_handle->numFds : 0;
        if (input_handle == nullptr || output_handle == nullptr ||
            input_handle->numFds < 1 || output_handle->numFds < 1 ||
            input_handle->data[0] < 0 || output_handle->data[0] < 0 ||
            mNativeIoContext->input_bytes >
                static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            mNativeIoContext->output_bytes >
                static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_INVALID_BUFFER_HANDLE;
            return false;
        }

        if (mHclV600Runtime != nullptr) {
            const bool bound = mHclV600Runtime->bindNativeHandleIo(
                static_cast<size_t>(mImageInputIndex), input_handle->data[0],
                mNativeIoContext->input_bytes,
                static_cast<size_t>(mImageOutputIndex), output_handle->data[0],
                mNativeIoContext->output_bytes);
            if (!bound) {
                mNativeIoContext->result_code =
                    MNN_HIAI_NATIVE_HANDLE_IO_INPUT_BIND_FAILED;
                MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: NativeHandle bind failed: %s\n",
                          mHclV600Runtime->lastError().c_str());
                return false;
            }
            mBoundInputAhb = mNativeIoContext->input_ahardware_buffer;
            mBoundOutputAhb = mNativeIoContext->output_ahardware_buffer;
            mNativeIoContext->result_code = MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS;
            MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: NativeHandle bound input_fd=%d output_fd=%d bytes=%u/%u\n",
                      input_handle->data[0], output_handle->data[0],
                      mNativeIoContext->input_bytes,
                      mNativeIoContext->output_bytes);
            return true;
        }


        hiai::NativeHandle input_native = {
            input_handle->data[0],
            static_cast<int>(mNativeIoContext->input_bytes), 0};
        hiai::NativeHandle output_native = {
            output_handle->data[0],
            static_cast<int>(mNativeIoContext->output_bytes), 0};
        auto native_input = make_shared<hiai::AiTensor>();
        const int input_ret = native_input->Init(
            input_native, &mInputDimension[mImageInputIndex],
            hiai::HIAI_DATATYPE_FLOAT32);
        if (input_ret != hiai::AI_SUCCESS) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_INPUT_BIND_FAILED;
            MNN_PRINT("MNN_HIAI_NATIVE_HANDLE_IO_AUDIT: input Init failed ret=%d fd_count=%d\n",
                      input_ret, input_handle->numFds);
            return false;
        }
        auto native_output = make_shared<hiai::AiTensor>();
        const int output_ret = native_output->Init(
            output_native, &mOutputDimension[mImageOutputIndex],
            hiai::HIAI_DATATYPE_FLOAT32);
        if (output_ret != hiai::AI_SUCCESS) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_OUTPUT_BIND_FAILED;
            MNN_PRINT("MNN_HIAI_NATIVE_HANDLE_IO_AUDIT: output Init failed ret=%d fd_count=%d\n", output_ret,
                      output_handle->numFds);
            return false;
        }

        mInputTensors = mHostInputTensors;
        mOutputTensors = mHostOutputTensors;
        mInputTensors[mImageInputIndex] = native_input;
        mOutputTensors[mImageOutputIndex] = native_output;
        mBoundInputAhb = mNativeIoContext->input_ahardware_buffer;
        mBoundOutputAhb = mNativeIoContext->output_ahardware_buffer;
        mNativeIoContext->result_code = MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS;
        MNN_PRINT("MNN_HIAI_NATIVE_HANDLE_IO_AUDIT: bound input_fd=%d output_fd=%d bytes=%u/%u\n",
                  input_handle->data[0], output_handle->data[0], mNativeIoContext->input_bytes,
                  mNativeIoContext->output_bytes);
        return true;
    }
    ErrorCode NPUBackend::buildIRModelAndLoad() {
        std::vector<ge::Operator> inputs;
        for (auto input : mInputOps) {
            inputs.push_back(input.second[0]);
        }
        std::vector<ge::Operator> outputOps;
        for (auto outOp : mOutGEOpMap) {
            outputOps.push_back(*outOp.first.get());
        }
        MNN_PRINT("mOutputOps : %lu \n", outputOps.size());

        string graphName = string("Graph1");
        string version = string("model_v000011");
        string modelName = to_string(0);
        mModelName.push_back(modelName);
        ge::Graph graph(graphName);
        graph.SetInputs(inputs).SetOutputs(outputOps);

        ge::Model model(modelName, version);
        model.SetGraph(graph);

        if (isExplicitHiAISession()) {
#if defined(GRAPH_API_EXPORT)
            auto streamNumber =
                ge::AttrValue::CreateFrom(static_cast<int64_t>(1));
#else
            auto streamNumber = ge::AttrValue::CreateFrom<ge::AttrValue::INT>(
                static_cast<int64_t>(1));
#endif
            const auto streamStatus =
                model.SetAttr("stream_num", std::move(streamNumber));
            if (streamStatus != ge::GRAPH_SUCCESS) {
                MNN_ERROR("[NPU] failed to set stream_num on GE model\n");
                return INVALID_VALUE;
            }
        }

        ge::Buffer buffer;
        ge::GraphErrCodeStatus geret = model.Save(buffer);
        if (geret != ge::GRAPH_SUCCESS) {
            MNN_ERROR("[NPU] Model save failed \n");
            if (isExplicitHiAISession()) {
                return INVALID_VALUE;
            }
        }
        const std::string modelIdentity = HiaiModelIdentity(buffer.GetData(), buffer.GetSize());

        if (mHclV600Runtime != nullptr) {
            const bool preferFp16 = mPrecision == BackendConfig::Precision_Low;
            if (!mHclV600Runtime->buildAndLoad(
                    buffer.GetData(), buffer.GetSize(), modelName, preferFp16,
                    HiaiCacheFile(mNPURuntime->mCachePath, modelIdentity, kHiaiV600CacheSuffix))) {
                if (mNativeIoContext != nullptr) {
                    mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_HCL_V600_FAILED;
                }
                MNN_ERROR(
                    "MNN_HIAI_HCL_V600_AUDIT: BuildV2 path failed; full MNN session rebuild with V320 required: %s\n",
                    mHclV600Runtime->lastError().c_str());
                return INVALID_VALUE;
            } else {
                for (size_t i = 0; i < mHclV600Runtime->inputCount(); ++i) {
                    if (mNativeIoContext != nullptr && mHclV600Runtime->inputSize(i) == mNativeIoContext->input_bytes) {
                        mImageInputIndex = static_cast<int>(i);
                    }
                }
                for (size_t i = 0; i < mHclV600Runtime->outputCount(); ++i) {
                    if (mNativeIoContext != nullptr &&
                        mHclV600Runtime->outputSize(i) == mNativeIoContext->output_bytes) {
                        mImageOutputIndex = static_cast<int>(i);
                    }
                }
                int outputIndex = 0;
                for (auto opMap : mOutGEOpMap) {
                    for (auto tensor : opMap.second) {
                        mMNNOutTensors.push_back(tensor);
                        MNN_PRINT("%d MNN/HCL output DIM:%d,%d,%d,%d bytes=%d\n", outputIndex, tensor->batch(),
                                  tensor->channel(), tensor->height(), tensor->width(), tensor->size());
                        ++outputIndex;
                    }
                }
                MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: BuildV2/InitV2 PASS model=%s inputs=%lu outputs=%lu precision=%s\n",
                          modelName.c_str(), mHclV600Runtime->inputCount(), mHclV600Runtime->outputCount(),
                          preferFp16 ? "FP16" : "DEFAULT");
                if (!EnsureHiaiCachePathMarker(mNPURuntime->mCachePath)) {
                    MNN_ERROR(
                        "MNN_HIAI_CACHE_AUDIT: MARKER_WRITE_FAILED "
                        "abi=V600 file=%s\n",
                        mNPURuntime->mCachePath.c_str());
                }
                return NO_ERROR;
            }
        }

        if (isExplicitHiAISession()) {
            std::string loaderError;
            const std::string runtimeLibraryDirectory =
                mNPURuntime->mHiAIOptions != nullptr
                    ? mNPURuntime->mHiAIOptions->runtimeLibraryDirectory
                    : std::string();
            if (!loadHiAIDynamicSymbols(HiAIDynamicLoadScope::Full,
                                        true,
                                        runtimeLibraryDirectory,
                                        &loaderError)) {
                MNN_ERROR("[NPU] HiAI V320 ABI unavailable: %s\n",
                          loaderError.c_str());
                if (mNativeIoContext != nullptr) {
                    mNativeIoContext->reserved
                        [MNN_HIAI_SESSION_STATUS_INDEX] =
                            MNN_HIAI_SESSION_V320_FAILED;
                }
                return INVALID_VALUE;
            }
        }

        domi::HiaiIrBuild ir_build;
        domi::ModelBufferData om_model_buff;
        const std::string v320CacheFile = HiaiCacheFile(mNPURuntime->mCachePath, modelIdentity, kHiaiV320CacheSuffix);
        std::vector<std::uint8_t> cachedV320Model;
        bool loadedFromV320Cache = false;
        if (!v320CacheFile.empty() && ReadHiaiCacheFile(v320CacheFile, &cachedV320Model)) {
            om_model_buff.data = cachedV320Model.data();
            om_model_buff.length = static_cast<std::uint32_t>(cachedV320Model.size());
            mMgrClient = LoadModelSync(om_model_buff, modelName);
            if (mMgrClient != nullptr) {
                loadedFromV320Cache = true;
                MNN_PRINT(
                    "MNN_HIAI_CACHE_AUDIT: HIT abi=V320 file=%s "
                    "bytes=%u\n",
                    v320CacheFile.c_str(), om_model_buff.length);
            } else {
                (void)std::remove(v320CacheFile.c_str());
                cachedV320Model.clear();
                om_model_buff = {};
                MNN_ERROR(
                    "MNN_HIAI_CACHE_AUDIT: INVALID abi=V320 "
                    "file=%s detail=LoadModelSync failed\n",
                    v320CacheFile.c_str());
            }
        } else if (!v320CacheFile.empty()) {
            if (access(v320CacheFile.c_str(), F_OK) == 0) {
                (void)std::remove(v320CacheFile.c_str());
                MNN_ERROR(
                    "MNN_HIAI_CACHE_AUDIT: INVALID abi=V320 "
                    "file=%s detail=empty or oversized cache\n",
                    v320CacheFile.c_str());
            } else {
                MNN_PRINT("MNN_HIAI_CACHE_AUDIT: MISS abi=V320 file=%s\n", v320CacheFile.c_str());
            }
        }
#ifdef HIAI_DEBUG
        WriteToBufferFile(buffer, "/data/local/tmp/test.irpb");
#endif
        bool createBufferSuc = loadedFromV320Cache || ir_build.CreateModelBuff(model, om_model_buff);

        if (!createBufferSuc) {
            MNN_ERROR("[NPU] Create Model Buff failed \n");
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_V320_FAILED;
            }
            if (isExplicitHiAISession()) {
                return INVALID_VALUE;
            }
        }
        bool buildIRSuc = loadedFromV320Cache || ir_build.BuildIRModel(model, om_model_buff);
        if (!buildIRSuc) {
            MNN_ERROR("[NPU] IR model build failed  \n");
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_V320_FAILED;
            }
            ir_build.ReleaseModelBuff(om_model_buff);
            return INVALID_VALUE;
        }
#ifdef HIAI_DEBUG
        WriteToOMFile(om_model_buff, "/data/local/tmp/test.om");
#endif
        if (!loadedFromV320Cache) {
            mMgrClient = LoadModelSync(om_model_buff, modelName);
        }

        if (mMgrClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client is null \n");
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_V320_FAILED;
            }
            if (!loadedFromV320Cache) {
                ir_build.ReleaseModelBuff(om_model_buff);
            }
            return INVALID_VALUE;
        }

        if (!loadedFromV320Cache) {
            if (!v320CacheFile.empty()) {
                if (WriteHiaiCacheFileAtomic(v320CacheFile, om_model_buff.data, om_model_buff.length)) {
                    MNN_PRINT(
                        "MNN_HIAI_CACHE_AUDIT: WRITE abi=V320 "
                        "file=%s bytes=%u\n",
                        v320CacheFile.c_str(), om_model_buff.length);
                } else {
                    MNN_ERROR(
                        "MNN_HIAI_CACHE_AUDIT: WRITE_FAILED abi=V320 "
                        "file=%s bytes=%u\n",
                        v320CacheFile.c_str(), om_model_buff.length);
                }
            }
            ir_build.ReleaseModelBuff(om_model_buff);
        }

        int result = getInOutTensorInfo(modelName);
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                result == 0 ? MNN_HIAI_SESSION_V320_READY : MNN_HIAI_SESSION_V320_FAILED;
        }
        if (result == 0) {
            if (!EnsureHiaiCachePathMarker(mNPURuntime->mCachePath)) {
                MNN_ERROR(
                    "MNN_HIAI_CACHE_AUDIT: MARKER_WRITE_FAILED "
                    "abi=V320 file=%s\n",
                    mNPURuntime->mCachePath.c_str());
            }
            MNN_PRINT("MNN_HIAI_SESSION_REBUILD_AUDIT: V320 graph build/load PASS\n");
        }
        return (result == 0) ? NO_ERROR : INVALID_VALUE;
    }

    int NPUBackend::process(int modelIndex) const {
#ifdef HIAI_DEBUG
        ATrace_beginSection("HIAI process");
#endif
        if (mHclV600Runtime != nullptr) {
            const int ret = mHclV600Runtime->run();
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->process_result = ret;
            }
            if (ret != 0) {
                MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: RunV3 failed status=%d\n", ret);
                mExecutionStatus = INVALID_VALUE;
            }
#ifdef HIAI_DEBUG
            ATrace_endSection();
#endif
            return ret;
        }

        if (mMgrClient == nullptr) {
            MNN_ERROR("MNN_HIAI: Process called without a loaded model\n");
            mExecutionStatus = NO_EXECUTION;
            return -1;
        }

        hiai::AiContext context;
        string key = "model_name";
        string value = to_string(modelIndex);
        context.AddPara(key, value);

        int istamp;

        int ret = mMgrClient->Process(context, *(const_cast<vector<shared_ptr<hiai::AiTensor>>*>(&mInputTensors)), 
                                      *(const_cast<vector<shared_ptr<hiai::AiTensor>>*>(&mOutputTensors)), 1000,
                                      istamp);
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->process_result = ret;
        }
        if (ret != hiai::AI_SUCCESS) {
            MNN_ERROR("MNN_HIAI: Process failed status=%d\n", ret);
            mExecutionStatus = INVALID_VALUE;
        }
#ifdef HIAI_DEBUG
        ATrace_endSection();
#endif
        return ret;
    }

    shared_ptr<ge::Operator> NPUBackend::getInputOps(const Op *op, int index) {
        vector<shared_ptr<ge::Operator>> ops;
        bool find = false;
        for (size_t i = 0; i < op->inputIndexes()->size(); i++){
            auto inputIndex = op->inputIndexes()->data()[i];
            // printf("inputIndex : %d \n", inputIndex);
            auto iter = mGrapMap.find(inputIndex);
            if(iter != mGrapMap.end()){
                find = true;
                auto xOp        = iter->second.back().first;
                ops.emplace_back(xOp);
            }
        }
        if(find == false){
            MNN_PRINT("not find input \n ");
        };
        return ops[index];
    }

    void NPUBackend::setOutputOps(const Op *op, vector<shared_ptr<ge::Operator>>&& HIAI_op,
                                  const std::vector<Tensor *> &outputs){
        if(op->type() == OpType_Slice || op->type() == OpType_TopKV2){
            for (size_t i = 0; i < op->outputIndexes()->size(); i++){
                auto index = op->outputIndexes()->data()[i];
                mSclipMap[index] = i;
            }
        }
        for (size_t i = 0; i < op->outputIndexes()->size(); i++){
            auto index = op->outputIndexes()->data()[i];
            vector<pair<shared_ptr<ge::Operator>, string>> ops;
            for (size_t j = 0; j < HIAI_op.size(); j++){
                ops.emplace_back(make_pair(HIAI_op[j], ""));
            }
            mGrapMap.insert(make_pair(index, ops));
        }

        MNNTensorList tensors;
        for (auto out: outputs)
        {
            bool isOutput = (TensorUtils::getDescribe(out)->usage 
                            ==Tensor::InsideDescribe::Usage::OUTPUT);
            if(isOutput == true){
                tensors.push_back(out);
            }
        }
        if(!tensors.empty()) {
            mOutGEOpMap.insert(make_pair(HIAI_op[HIAI_op.size()-1], tensors));
        }
    }

    NPURuntime::NPURuntime(const Backend::Info& info,
                           const std::string& hiaiRuntimeVersion) {
        mInfo.type = info.type;
        mInfo.numThread = info.numThread;
        mInfo.mode = info.mode;
        mHiaiRuntimeVersion = hiaiRuntimeVersion;
        mHiAIOptions = copyHiAIOptions(info);
        mInfo.user = &mHiAIOptions->config;
        mExplicitHiAI =
            mHiAIOptions != nullptr &&
            mHiAIOptions->explicitConfig;

        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != mInfo.user) {
            precision = mInfo.user->precision;
            power     = mInfo.user->power;
        }

        mPrecision = precision;
    }

    NPURuntime::~NPURuntime() {}

    Backend* NPURuntime::onCreate(const BackendConfig* config, Backend* origin) const {
        return new NPUBackend(this);
    }

    void NPURuntime::onGabageCollect(int level) {
        // nothing now
    }
    Runtime::CompilerType NPURuntime::onGetCompilerType() const {
        return Compiler_Origin;
    }

    struct NPUBackendCreator : RuntimeCreator {

        virtual Runtime* onCreate(const Backend::Info& info) const override {
            AUTOTIME;
            const auto options = copyHiAIOptions(info);
            if (options == nullptr) {
                MNN_ERROR("MNN_HIAI: invalid BackendConfig::sharedContext configuration.\n");
                return nullptr;
            }
            options->report(MNN_HIAI_STAGE_RUNTIME, NO_EXECUTION, "HiAI Runtime initialization failed");
            const auto* hiaiOptions = options.get();
            const bool explicitlyRequestedHiAI =
                hiaiOptions != nullptr &&
                hiaiOptions->explicitConfig;
            const bool forceV320 = HiaiForceV320(info,
                                                 explicitlyRequestedHiAI);
            std::string hclVersion;
            bool useHclV600 = explicitlyRequestedHiAI &&
                              !forceV320 &&
                              HiaiHclV600Runtime::IsSupported(&hclVersion);
            std::string loaderError;
            const auto loaderScope = useHclV600
                ? HiAIDynamicLoadScope::GraphOnly
                : HiAIDynamicLoadScope::Full;
            const std::string runtimeLibraryDirectory =
                explicitlyRequestedHiAI
                    ? hiaiOptions->runtimeLibraryDirectory
                    : std::string();
            if (!loadHiAIDynamicSymbols(loaderScope,
                                        explicitlyRequestedHiAI &&
                                            !useHclV600,
                                        runtimeLibraryDirectory,
                                        &loaderError)) {
                options->report(MNN_HIAI_STAGE_RUNTIME, NO_EXECUTION, loaderError.c_str());
                MNN_ERROR("[NPU] HiAI dynamic loader unavailable: %s\n",
                          loaderError.c_str());
                return nullptr;
            }
            // Some legacy clients publish the HCL plugin only after their
            // complete runtime bootstrap. Preserve that ordering as a second
            // chance without forcing the full client set on direct-HCL paths.
            if (!useHclV600 && explicitlyRequestedHiAI && !forceV320) {
                useHclV600 = HiaiHclV600Runtime::IsSupported(&hclVersion);
            }
            std::string runtimeVersion;
            {
                if (useHclV600) {
                    MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: capability PASS version=%s\n",
                              hclVersion.c_str());
                    options->report(MNN_HIAI_STAGE_RUNTIME, NO_ERROR, "HiAI HCL Runtime ready");
                    return new NPURuntime(info, hclVersion);
                }
                shared_ptr<hiai::AiModelMngerClient> mgrClient = make_shared<hiai::AiModelMngerClient>();
                if(mgrClient.get() == nullptr){
                    MNN_ERROR("mgrClient.get() == NULL");
                    return nullptr;
                }
                
				auto ret = mgrClient->Init(nullptr);
                if (ret != hiai::AI_SUCCESS) {
                    MNN_ERROR("[NPU] AiModelMngerClient Init Failed!\n");
                    return nullptr;
                }
				
                const char* currentversion = mgrClient->GetVersion();
                if(currentversion != nullptr){
                    MNN_PRINT("[NPU] ddk currentversion : %s \n", currentversion);
                    runtimeVersion = currentversion;
                }else{
                    MNN_ERROR("[NPU] current version don't support, return nullptr\n");
                    return nullptr;
                }

                const bool unsupportedVersion = explicitlyRequestedHiAI
                                                    ? string(currentversion).compare("100.320.000.000") < 0
                                                    : string(currentversion).compare("100.330.000.000") <= 0;
                if (unsupportedVersion) {
                    MNN_PRINT("[NPU] current version don't support,version=%s \n", currentversion);
                    return nullptr;
                }
            }

            options->report(MNN_HIAI_STAGE_RUNTIME, NO_ERROR, "HiAI Runtime ready");
            return new NPURuntime(info, runtimeVersion);
        }

        bool onValid(Backend::Info& info) const override {
            (void)info;
            return true;
        }
    };

    bool registerHiAIRuntimeCreator() {
        static std::mutex registrationMutex;
        static bool registered = false;
        std::lock_guard<std::mutex> lock(registrationMutex);
        if (!registered) {
            registered = MNNInsertExtraRuntimeCreator(
                MNN_FORWARD_USER_0, new NPUBackendCreator, false);
        }
        return registered;
    }

#if !defined(MNN_NPU_INTEGRATED_REGISTRATION) && \
    !defined(MNN_HIAI_EXPLICIT_PLUGIN)
    // Preserve the original separate-backend behavior. In the single-library
    // build MNNCore calls registerHiAIRuntimeCreator according to the selected
    // automatic/manual mode instead.
    static const auto __npu_global_initializer = []() {
        return registerHiAIRuntimeCreator();
    }();
#endif
    } // namespace MNN
