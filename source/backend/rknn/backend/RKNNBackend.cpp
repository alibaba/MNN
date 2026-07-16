#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "MNN/plugin/PluginShapeInference.hpp"
#include "core/Backend.hpp"
#include "core/MNNFileUtils.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "rknn_api.h"
#include "shape/SizeComputer.hpp"

#ifdef MNN_WITH_PLUGIN
namespace MNN {
namespace RKNN {
namespace {

static const char* kRuntimeLibEnv = "MNN_RKNN_RUNTIME_LIB";
static const char* kPluginTypeName = "RKNN";
static const char* kModelPathAttr = "model_path";

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
        if (nullptr == libPath || libPath[0] == 0) {
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
            MNN_ERROR("MNN_RKNN: dlsym failed for %s\n", symbol);                                 \
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

static std::string getStringAttr(const plugin::PluginContext* ctx, const char* key) {
    auto attr = ctx->getAttr(key);
    if (nullptr == attr || nullptr == attr->s()) {
        return "";
    }
    return attr->s()->str();
}

static std::string resolveModelPath(const std::string& dirPath, const std::string& path) {
    if (path.empty()) {
        return "";
    }
    if (path[0] == '/') {
        return path;
    }
    if (dirPath.empty() || dirPath == ".") {
        return path;
    }
    return MNNFilePathConcat(dirPath, path);
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
static bool convertLayoutIfNeeded(const Tensor* tensor, rknn_tensor_format expectFormat,
                                  std::vector<uint8_t>* converted, void** buf, uint32_t* size,
                                  rknn_tensor_format* actualFormat) {
    auto currentFormat = mapTensorFormat(tensor);
    *actualFormat = currentFormat;
    *buf = tensor->buffer().host;
    *size = (uint32_t)tensor->size();

    if (expectFormat == currentFormat) {
        return true;
    }
    if (expectFormat != RKNN_TENSOR_NHWC || currentFormat != RKNN_TENSOR_NCHW) {
        return true;
    }
    if (tensor->dimensions() != 4) {
        MNN_ERROR("MNN_RKNN: unsupported layout conversion for %dD tensor\n", tensor->dimensions());
        return false;
    }

    const int batch = tensor->batch();
    const int channel = tensor->channel();
    const int height = tensor->height();
    const int width = tensor->width();
    const int elementBytes = tensor->getType().bytes();
    if (batch <= 0 || channel <= 0 || height <= 0 || width <= 0 || elementBytes <= 0) {
        MNN_ERROR("MNN_RKNN: invalid tensor shape for layout conversion\n");
        return false;
    }

    converted->resize((size_t)tensor->size());
    auto src = reinterpret_cast<const uint8_t*>(tensor->buffer().host);
    auto dst = converted->data();
    for (int n = 0; n < batch; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channel; ++c) {
                    const size_t srcIndex = ((((size_t)n * (size_t)channel + (size_t)c) * (size_t)height + (size_t)h) * (size_t)width + (size_t)w) * (size_t)elementBytes;
                    const size_t dstIndex = ((((size_t)n * (size_t)height + (size_t)h) * (size_t)width + (size_t)w) * (size_t)channel + (size_t)c) * (size_t)elementBytes;
                    ::memcpy(dst + dstIndex, src + srcIndex, (size_t)elementBytes);
                }
            }
        }
    }
    *buf = converted->data();
    *size = (uint32_t)converted->size();
    *actualFormat = expectFormat;
    return true;
}

static std::string buildProfileString(const RKNNApi* api, rknn_context context) {
    std::ostringstream oss;
    rknn_perf_run perfRun;
    std::memset(&perfRun, 0, sizeof(perfRun));
    auto ret = api->query(context, RKNN_QUERY_PERF_RUN, &perfRun, sizeof(perfRun));
    if (ret == RKNN_SUCC) {
        oss << "npu_run   : " << (double)perfRun.run_duration / 1000.0 << " ms\n";
    } else {
        oss << "npu_run   : unavailable\n";
    }

    rknn_perf_detail perfDetail;
    std::memset(&perfDetail, 0, sizeof(perfDetail));
    ret = api->query(context, RKNN_QUERY_PERF_DETAIL, &perfDetail, sizeof(perfDetail));
    if (ret == RKNN_SUCC && perfDetail.perf_data != nullptr && perfDetail.data_len > 0) {
        oss << "perf_detail:\n";
        oss.write(perfDetail.perf_data, perfDetail.data_len);
        if (perfDetail.perf_data[perfDetail.data_len - 1] != '\n') {
            oss << '\n';
        }
    } else {
        oss << "perf_detail: unavailable\n";
    }
    return oss.str();
}

class RKNNPluginShape : public plugin::InferShapeKernel {
public:
    bool compute(plugin::InferShapeContext* ctx) override {
        for (int i = 0; i < ctx->outputs().size(); ++i) {
            auto key = std::string("o_") + std::to_string(i);
            auto attr = ctx->getAttr(key);
            if (nullptr == attr || nullptr == attr->tensor()) {
                MNN_ERROR("MNN_RKNN: missing output shape attr %s\n", key.c_str());
                return false;
            }
            auto blob = attr->tensor();
            auto dst = ctx->output(i);
            dst->setType(blob->dataType());
            if (nullptr != blob->dims()) {
                dst->buffer().dimensions = blob->dims()->size();
                for (int j = 0; j < blob->dims()->size(); ++j) {
                    dst->setLength(j, blob->dims()->data()[j]);
                }
            } else {
                dst->buffer().dimensions = 0;
            }
            TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
        }
        return true;
    }
};

class RKNNPluginExecute : public plugin::CPUComputeKernel {
public:
    ~RKNNPluginExecute() override {
        if (mContext != 0 && nullptr != mApi) {
            mApi->destroy(mContext);
        }
    }

    bool init(plugin::CPUKernelContext* ctx) override {
        mApi = loadApi();
        if (nullptr == mApi) {
            return false;
        }
        auto runtime = ctx->backend() == nullptr ? nullptr : ctx->backend()->getRuntime();
        mEnableProfile = runtime != nullptr && runtime->hint().enableBackendProfile;
        mModelPath = resolveModelPath(ctx->dir_path(), getStringAttr(ctx, kModelPathAttr));
        if (mModelPath.empty()) {
            MNN_ERROR("MNN_RKNN: Plugin(%s) requires attr %s\n", kPluginTypeName, kModelPathAttr);
            return false;
        }
        if (!MNNFileExist(mModelPath.c_str())) {
            MNN_ERROR("MNN_RKNN: model file does not exist: %s\n", mModelPath.c_str());
            return false;
        }
        uint32_t initFlags = 0;
        if (mEnableProfile) {
            initFlags |= RKNN_FLAG_COLLECT_PERF_MASK;
        }
        if (mApi->init(&mContext, (void*)mModelPath.c_str(), 0, initFlags, nullptr) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_init failed for %s\n", mModelPath.c_str());
            return false;
        }
        if (mApi->query(mContext, RKNN_QUERY_IN_OUT_NUM, &mIoNum, sizeof(mIoNum)) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: query in/out num failed\n");
            return false;
        }
        mInputAttrs.resize(mIoNum.n_input);
        mOutputAttrs.resize(mIoNum.n_output);
        for (uint32_t i = 0; i < mIoNum.n_input; ++i) {
            std::memset(&mInputAttrs[i], 0, sizeof(rknn_tensor_attr));
            mInputAttrs[i].index = i;
            if (mApi->query(mContext, RKNN_QUERY_INPUT_ATTR, &mInputAttrs[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
                MNN_ERROR("MNN_RKNN: query input attr failed: %u\n", i);
                return false;
            }
        }
        for (uint32_t i = 0; i < mIoNum.n_output; ++i) {
            std::memset(&mOutputAttrs[i], 0, sizeof(rknn_tensor_attr));
            mOutputAttrs[i].index = i;
            if (mApi->query(mContext, RKNN_QUERY_OUTPUT_ATTR, &mOutputAttrs[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
                MNN_ERROR("MNN_RKNN: query output attr failed: %u\n", i);
                return false;
            }
        }
        return true;
    }

    bool resize(plugin::CPUKernelContext* ctx) override {
        if ((uint32_t)ctx->inputs().size() != mIoNum.n_input || (uint32_t)ctx->outputs().size() != mIoNum.n_output) {
            MNN_ERROR("MNN_RKNN: input/output count mismatch, expect %u/%u, got %zu/%zu\n",
                      mIoNum.n_input, mIoNum.n_output, ctx->inputs().size(), ctx->outputs().size());
            return false;
        }
        return true;
    }

    bool compute(plugin::CPUKernelContext* ctx) override {
        auto runtime = ctx->backend() == nullptr ? nullptr : ctx->backend()->getRuntime();
        std::vector<std::unique_ptr<Tensor>> hostInputs;
        std::vector<std::vector<uint8_t>> convertedInputs(ctx->inputs().size());
        std::vector<rknn_input> rknnInputs(ctx->inputs().size());
        for (size_t i = 0; i < ctx->inputs().size(); ++i) {
            auto src = ctx->input((int)i);
            hostInputs.emplace_back(new Tensor(src, getHostTensorDimType(src)));
            if (!MNNCPUCopyBuffer(src, hostInputs.back().get())) {
                MNN_ERROR("MNN_RKNN: failed to copy input tensor %zu to host\n", i);
                return false;
            }
            void* inputBuf = hostInputs.back()->buffer().host;
            uint32_t inputSize = (uint32_t)hostInputs.back()->size();
            auto inputFormat = mapTensorFormat(hostInputs.back().get());
            if (!convertLayoutIfNeeded(hostInputs.back().get(), mInputAttrs[i].fmt, &convertedInputs[i], &inputBuf, &inputSize, &inputFormat)) {
                MNN_ERROR("MNN_RKNN: failed to convert input tensor %zu layout\n", i);
                return false;
            }
            std::memset(&rknnInputs[i], 0, sizeof(rknn_input));
            rknnInputs[i].index = (uint32_t)i;
            rknnInputs[i].buf = inputBuf;
            rknnInputs[i].size = inputSize;
            rknnInputs[i].pass_through = 0;
            rknnInputs[i].type = mapTensorType(hostInputs.back().get());
            rknnInputs[i].fmt = inputFormat;
        }
        if (mApi->inputsSet(mContext, (uint32_t)rknnInputs.size(), rknnInputs.data()) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_inputs_set failed\n");
            return false;
        }
        if (mApi->run(mContext, nullptr) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_run failed\n");
            return false;
        }

        std::vector<rknn_output> rknnOutputs(ctx->outputs().size());
        for (size_t i = 0; i < ctx->outputs().size(); ++i) {
            std::memset(&rknnOutputs[i], 0, sizeof(rknn_output));
            rknnOutputs[i].index = (uint32_t)i;
            rknnOutputs[i].want_float = 1;
            rknnOutputs[i].is_prealloc = 0;
        }
        if (mApi->outputsGet(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data(), nullptr) != RKNN_SUCC) {
            MNN_ERROR("MNN_RKNN: rknn_outputs_get failed\n");
            return false;
        }
        if (nullptr != runtime) {
            if (mEnableProfile) {
                runtime->setLastBackendProfile(buildProfileString(mApi, mContext));
            } else {
                runtime->setLastBackendProfile("");
            }
        }

        for (size_t i = 0; i < ctx->outputs().size(); ++i) {
            auto dst = ctx->output((int)i);
            if (dst->getType().code != halide_type_float || dst->getType().bits != 32) {
                MNN_ERROR("MNN_RKNN: only float32 outputs are supported in the first plugin version\n");
                mApi->outputsRelease(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data());
                return false;
            }
            Tensor hostOutput(dst, getHostTensorDimType(dst));
            auto copySize = ALIMIN((int)hostOutput.size(), (int)rknnOutputs[i].size);
            std::memcpy(hostOutput.buffer().host, rknnOutputs[i].buf, copySize);
            if (!MNNCPUCopyBuffer(&hostOutput, dst)) {
                MNN_ERROR("MNN_RKNN: failed to copy output tensor %zu from host\n", i);
                mApi->outputsRelease(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data());
                return false;
            }
        }
        mApi->outputsRelease(mContext, (uint32_t)rknnOutputs.size(), rknnOutputs.data());
        return true;
    }

private:
    const RKNNApi* mApi = nullptr;
    std::string mModelPath;
    rknn_context mContext = 0;
    rknn_input_output_num mIoNum{};
    std::vector<rknn_tensor_attr> mInputAttrs;
    std::vector<rknn_tensor_attr> mOutputAttrs;
    bool mEnableProfile = false;
};

static auto _rknn_plugin_shape_registrar __attribute__((unused)) =
    MNN::plugin::InferShapeKernelRegistrar<RKNNPluginShape>("RKNN");
static auto _rknn_plugin_compute_registrar __attribute__((unused)) =
    MNN::plugin::ComputeKernelRegistrar<RKNNPluginExecute>("RKNN");

} // namespace
} // namespace RKNN
} // namespace MNN
#endif
