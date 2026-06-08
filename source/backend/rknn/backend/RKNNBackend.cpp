#include "RKNNBackend.hpp"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "core/MNNFileUtils.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "rknn_api.h"

namespace MNN {
namespace RKNN {
namespace {

static const char* kRuntimeLibEnv = "MNN_RKNN_RUNTIME_LIB";
static const char* kExtraTypeName = "RKNN";
static const char* kModelPathAttr = "model_path";

class HostMemObj : public Backend::MemObj {
public:
    explicit HostMemObj(size_t size) : mPtr(std::malloc(size)) {
    }
    ~HostMemObj() override {
        std::free(mPtr);
    }
    MemChunk chunk() override {
        return MemChunk(mPtr, 0);
    }
    bool valid() const {
        return nullptr != mPtr;
    }
private:
    void* mPtr = nullptr;
};

struct RKNNApi {
    using Init = int (*)(rknn_context*, void*, uint32_t, uint32_t, rknn_init_extend*);
    using Destroy = int (*)(rknn_context);
    using Query = int (*)(rknn_context, rknn_query_cmd, void*, uint32_t);
    using InputsSet = int (*)(rknn_context, uint32_t, rknn_input[]);
    using Run = int (*)(rknn_context, rknn_run_extend*);
    using OutputsGet = int (*)(rknn_context, uint32_t, rknn_output[], rknn_output_extend*);
    using OutputsRelease = int (*)(rknn_context, uint32_t, rknn_output[]);

    bool loaded = false;
    void* handle = nullptr;
    Init init = nullptr;
    Destroy destroy = nullptr;
    Query query = nullptr;
    InputsSet inputsSet = nullptr;
    Run run = nullptr;
    OutputsGet outputsGet = nullptr;
    OutputsRelease outputsRelease = nullptr;
};

static const RKNNApi* loadApi() {
    static std::once_flag once;
    static RKNNApi api;
    std::call_once(once, []() {
        auto libPath = std::getenv(kRuntimeLibEnv);
        if (nullptr == libPath || libPath[0] == '\0') {
            MNN_ERROR("MNN_RKNN: missing environment variable %s\n", kRuntimeLibEnv);
            return;
        }
        api.handle = dlopen(libPath, RTLD_NOW | RTLD_LOCAL);
        if (nullptr == api.handle) {
            MNN_ERROR("MNN_RKNN: dlopen failed for %s, error: %s\n", libPath, dlerror());
            return;
        }
#define MNN_RKNN_LOAD_SYMBOL(typeName, field, symbol)                                               \
        api.field = reinterpret_cast<RKNNApi::typeName>(dlsym(api.handle, symbol));                 \
        if (nullptr == api.field) {                                                                  \
            MNN_ERROR("MNN_RKNN: dlsym failed for %s\n", symbol);                                   \
            return;                                                                                  \
        }
        MNN_RKNN_LOAD_SYMBOL(Init, init, "rknn_init");
        MNN_RKNN_LOAD_SYMBOL(Destroy, destroy, "rknn_destroy");
        MNN_RKNN_LOAD_SYMBOL(Query, query, "rknn_query");
        MNN_RKNN_LOAD_SYMBOL(InputsSet, inputsSet, "rknn_inputs_set");
        MNN_RKNN_LOAD_SYMBOL(Run, run, "rknn_run");
        MNN_RKNN_LOAD_SYMBOL(OutputsGet, outputsGet, "rknn_outputs_get");
        MNN_RKNN_LOAD_SYMBOL(OutputsRelease, outputsRelease, "rknn_outputs_release");
#undef MNN_RKNN_LOAD_SYMBOL
        api.loaded = true;
    });
    return api.loaded ? &api : nullptr;
}

static std::string getStringAttr(const Extra* extra, const char* key) {
    if (nullptr == extra || nullptr == extra->attr()) {
        return "";
    }
    for (int i = 0; i < extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (nullptr == attr || nullptr == attr->key()) {
            continue;
        }
        if (attr->key()->str() == key && nullptr != attr->s()) {
            return attr->s()->str();
        }
    }
    return "";
}

static std::string resolveModelPath(const Backend* backend, const std::string& path) {
    if (path.empty()) {
        return "";
    }
    if (!path.empty() && path[0] == '/') {
        return path;
    }
    return MNNFilePathConcat(backend->pNPUModelDirPath, path);
}

static rknn_tensor_type mapTensorType(const Tensor* tensor) {
    auto type = tensor->getType();
    if (type.code == halide_type_float && type.bits == 32) {
        return RKNN_TENSOR_FLOAT32;
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        return RKNN_TENSOR_UINT8;
    }
    if (type.code == halide_type_int && type.bits == 8) {
        return RKNN_TENSOR_INT8;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return RKNN_TENSOR_INT32;
    }
    return RKNN_TENSOR_FLOAT32;
}

static rknn_tensor_format mapTensorFormat(const Tensor* tensor) {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    if (format == MNN_DATA_FORMAT_NHWC) {
        return RKNN_TENSOR_NHWC;
    }
    return RKNN_TENSOR_NCHW;
}

static Tensor::DimensionType getHostTensorDimType(const Tensor* tensor) {
    return tensor->getDimensionType();
}

class RKNNExecution : public Execution {
public:
    RKNNExecution(Backend* backend, const Op* op, const RKNNApi* api) : Execution(backend), mApi(api) {
        if (nullptr == op || op->type() != OpType_Extra || nullptr == op->main_as_Extra()) {
            MNN_ERROR("MNN_RKNN: invalid op for RKNN execution\n");
            mValid = false;
            return;
        }
        auto extra = op->main_as_Extra();
        if (extra->type()->str() != kExtraTypeName) {
            MNN_ERROR("MNN_RKNN: unsupported Extra type\n");
            mValid = false;
            return;
        }
        mModelPath = resolveModelPath(backend, getStringAttr(extra, kModelPathAttr));
        if (mModelPath.empty()) {
            MNN_ERROR("MNN_RKNN: Extra(%s) requires attr '%s'\n", kExtraTypeName, kModelPathAttr);
            mValid = false;
            return;
        }
        if (!MNNFileExist(mModelPath.c_str())) {
            MNN_ERROR("MNN_RKNN: model file does not exist: %s\n", mModelPath.c_str());
            mValid = false;
            return;
        }
        if (mApi->init(&mContext, (void*)mModelPath.c_str(), 0, 0, nullptr) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_init failed for %s\n", mModelPath.c_str());
            mValid = false;
            return;
        }
        if (mApi->query(mContext, RKNN_QUERY_IN_OUT_NUM, &mIoNum, sizeof(mIoNum)) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: query in/out num failed\n");
            mValid = false;
            return;
        }
        mInputAttrs.resize(mIoNum.n_input);
        mOutputAttrs.resize(mIoNum.n_output);
        for (uint32_t i = 0; i < mIoNum.n_input; ++i) {
            std::memset(&mInputAttrs[i], 0, sizeof(rknn_tensor_attr));
            mInputAttrs[i].index = i;
            if (mApi->query(mContext, RKNN_QUERY_INPUT_ATTR, &mInputAttrs[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
                MNN_ERROR("MNN_RKNN: query input attr failed: %u\n", i);
                mValid = false;
                return;
            }
        }
        for (uint32_t i = 0; i < mIoNum.n_output; ++i) {
            std::memset(&mOutputAttrs[i], 0, sizeof(rknn_tensor_attr));
            mOutputAttrs[i].index = i;
            if (mApi->query(mContext, RKNN_QUERY_OUTPUT_ATTR, &mOutputAttrs[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
                MNN_ERROR("MNN_RKNN: query output attr failed: %u\n", i);
                mValid = false;
                return;
            }
        }
    }

    ~RKNNExecution() override {
        if (mContext != 0 && nullptr != mApi) {
            mApi->destroy(mContext);
        }
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if ((uint32_t)inputs.size() != mIoNum.n_input || (uint32_t)outputs.size() != mIoNum.n_output) {
            MNN_ERROR("MNN_RKNN: input/output count mismatch, expect %u/%u, got %zu/%zu\n",
                      mIoNum.n_input, mIoNum.n_output, inputs.size(), outputs.size());
            return INVALID_VALUE;
        }
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        std::vector<std::unique_ptr<Tensor>> hostInputs;
        std::vector<rknn_input> rknnInputs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            hostInputs.emplace_back(new Tensor(inputs[i], getHostTensorDimType(inputs[i])));
            if (!MNNCPUCopyBuffer(inputs[i], hostInputs.back().get())) {
                MNN_ERROR("MNN_RKNN: failed to copy input tensor %zu to host\n", i);
                return INVALID_VALUE;
            }
            std::memset(&rknnInputs[i], 0, sizeof(rknn_input));
            rknnInputs[i].index = (uint32_t)i;
            rknnInputs[i].buf = hostInputs.back()->buffer().host;
            rknnInputs[i].size = hostInputs.back()->size();
            rknnInputs[i].pass_through = 0;
            rknnInputs[i].type = mapTensorType(hostInputs.back().get());
            rknnInputs[i].fmt = mapTensorFormat(hostInputs.back().get());
        }
        if (mApi->inputsSet(mContext, (uint32_t)rknnInputs.size(), rknnInputs.data()) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_inputs_set failed\n");
            return INVALID_VALUE;
        }
        if (mApi->run(mContext, nullptr) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_run failed\n");
            return INVALID_VALUE;
        }

        std::vector<rknn_output> rknnOutputs(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::memset(&rknnOutputs[i], 0, sizeof(rknn_output));
            rknnOutputs[i].index = (uint32_t)i;
            rknnOutputs[i].want_float = 1;
            rknnOutputs[i].is_prealloc = 0;
        }
        if (mApi->outputsGet(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data(), nullptr) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_outputs_get failed\n");
            return INVALID_VALUE;
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i]->getType().code != halide_type_float || outputs[i]->getType().bits != 32) {
                MNN_ERROR("MNN_RKNN: only float32 outputs are supported in the first runtime version\n");
                mApi->outputsRelease(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data());
                return NOT_SUPPORT;
            }
            Tensor hostOutput(outputs[i], getHostTensorDimType(outputs[i]));
            auto copySize = ALIMIN((int)hostOutput.size(), (int)rknnOutputs[i].size);
            std::memcpy(hostOutput.buffer().host, rknnOutputs[i].buf, copySize);
            if (!MNNCPUCopyBuffer(&hostOutput, outputs[i])) {
                MNN_ERROR("MNN_RKNN: failed to copy output tensor %zu from host\n", i);
                mApi->outputsRelease(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data());
                return INVALID_VALUE;
            }
        }
        mApi->outputsRelease(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data());
        return NO_ERROR;
    }

private:
    const RKNNApi* mApi = nullptr;
    std::string mModelPath;
    rknn_context mContext = 0;
    rknn_input_output_num mIoNum{};
    std::vector<rknn_tensor_attr> mInputAttrs;
    std::vector<rknn_tensor_attr> mOutputAttrs;
};

} // namespace

RKNNBackend::RKNNBackend(const RKNNRuntime* runtime) : Backend(MNN_FORWARD_USER_2), mRuntime(runtime) {
}

Execution* RKNNBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    auto api = loadApi();
    if (nullptr == api) {
        return nullptr;
    }
    if (nullptr == op || op->type() != OpType_Extra || nullptr == op->main_as_Extra()) {
        return nullptr;
    }
    auto extra = op->main_as_Extra();
    if (extra->type()->str() != kExtraTypeName) {
        return nullptr;
    }
    auto exe = new RKNNExecution(this, op, api);
    if (!exe->valid()) {
        delete exe;
        return nullptr;
    }
    return exe;
}

void RKNNBackend::onResizeBegin() {
}

ErrorCode RKNNBackend::onResizeEnd() {
    return NO_ERROR;
}

void RKNNBackend::onExecuteBegin() const {
}

void RKNNBackend::onExecuteEnd() const {
}

Backend::MemObj* RKNNBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    auto mem = new HostMemObj(tensor->size());
    if (!mem->valid()) {
        delete mem;
        return nullptr;
    }
    return mem;
}

bool RKNNBackend::onClearBuffer() {
    return true;
}

void RKNNBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    MNNCPUCopyBuffer(srcTensor, dstTensor);
}

const Runtime* RKNNBackend::getRuntime() {
    return mRuntime;
}

RKNNRuntime::RKNNRuntime(const Backend::Info& info) : mInfo(info) {
}

Backend* RKNNRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new RKNNBackend(this);
}

void RKNNRuntime::onGabageCollect(int level) {
}

Runtime::CompilerType RKNNRuntime::onGetCompilerType() const {
    return Runtime::Compiler_Origin;
}

Runtime* RKNNRuntimeCreator::onCreate(const Backend::Info& info) const {
    if (nullptr == loadApi()) {
        return nullptr;
    }
    return new RKNNRuntime(info);
}

bool RKNNRuntimeCreator::onValid(Backend::Info& info) const {
    info.mode = Backend::Info::DIRECT;
    return true;
}

} // namespace RKNN

void registerRKNNRuntimeCreator() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_2, new RKNN::RKNNRuntimeCreator, false);
}

} // namespace MNN
