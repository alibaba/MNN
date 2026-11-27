#include "llm_napi.h"

#include "diffusion_session.h"
#include "llm_session.h"

#include <MNN/MNNDefine.h>

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <thread>

#include <atomic>
#include <cstring>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using mnn_harmony::BenchmarkRequest;
using mnn_harmony::BenchmarkResult;
using mnn_harmony::BenchmarkSample;
using mnn_harmony::DiffusionRequest;
using mnn_harmony::DiffusionResult;
using mnn_harmony::DiffusionSession;
using mnn_harmony::GenerateRequest;
using mnn_harmony::GenerateResult;
using mnn_harmony::LlmSession;

bool readString(napi_env env, napi_value value, std::string& output) {
    size_t length = 0;
    if (napi_get_value_string_utf8(env, value, nullptr, 0, &length) != napi_ok) {
        return false;
    }
    output.resize(length + 1);
    if (napi_get_value_string_utf8(env, value, &output[0], output.size(), &length) != napi_ok) {
        return false;
    }
    output.resize(length);
    return true;
}

bool readNamedString(napi_env env, napi_value object, const char* name, std::string& output) {
    napi_value property;
    return napi_get_named_property(env, object, name, &property) == napi_ok && readString(env, property, output);
}

bool readNamedInt(napi_env env, napi_value object, const char* name, int& output) {
    napi_value property;
    return napi_get_named_property(env, object, name, &property) == napi_ok &&
           napi_get_value_int32(env, property, &output) == napi_ok;
}

bool readNamedDouble(napi_env env, napi_value object, const char* name, double& output) {
    napi_value property;
    return napi_get_named_property(env, object, name, &property) == napi_ok &&
           napi_get_value_double(env, property, &output) == napi_ok;
}

bool readNamedBool(napi_env env, napi_value object, const char* name, bool& output) {
    napi_value property;
    return napi_get_named_property(env, object, name, &property) == napi_ok &&
           napi_get_value_bool(env, property, &output) == napi_ok;
}

bool readGenerateRequest(napi_env env, napi_value value, GenerateRequest& request) {
    napi_valuetype type = napi_undefined;
    if (napi_typeof(env, value, &type) != napi_ok || type != napi_object ||
        !readNamedString(env, value, "requestId", request.requestId) || request.requestId.empty()) {
        return false;
    }

    napi_value maxTokens;
    if (napi_get_named_property(env, value, "maxTokens", &maxTokens) != napi_ok ||
        napi_get_value_int32(env, maxTokens, &request.maxTokens) != napi_ok || request.maxTokens <= 0) {
        return false;
    }

    napi_value messages;
    bool isArray = false;
    uint32_t length = 0;
    if (napi_get_named_property(env, value, "messages", &messages) != napi_ok ||
        napi_is_array(env, messages, &isArray) != napi_ok || !isArray ||
        napi_get_array_length(env, messages, &length) != napi_ok || length == 0) {
        return false;
    }

    request.messages.reserve(length);
    for (uint32_t index = 0; index < length; ++index) {
        napi_value item;
        std::string role;
        std::string content;
        if (napi_get_element(env, messages, index, &item) != napi_ok || !readNamedString(env, item, "role", role) ||
            !readNamedString(env, item, "content", content) ||
            (role != "system" && role != "user" && role != "assistant")) {
            return false;
        }
        request.messages.emplace_back(std::move(role), std::move(content));
    }
    readNamedString(env, value, "audioOutputPath", request.audioOutputPath);
    napi_value streamAudio;
    if (napi_get_named_property(env, value, "streamAudio", &streamAudio) == napi_ok) {
        bool flag = false;
        if (napi_get_value_bool(env, streamAudio, &flag) == napi_ok) {
            request.streamAudio = flag;
        }
    }
    return true;
}

napi_value createError(napi_env env, const std::string& message) {
    napi_value text;
    napi_value error;
    napi_create_string_utf8(env, message.c_str(), message.size(), &text);
    napi_create_error(env, nullptr, text, &error);
    return error;
}

void setDouble(napi_env env, napi_value object, const char* name, double value) {
    napi_value property;
    napi_create_double(env, value, &property);
    napi_set_named_property(env, object, name, property);
}

void setInt(napi_env env, napi_value object, const char* name, int value) {
    napi_value property;
    napi_create_int32(env, value, &property);
    napi_set_named_property(env, object, name, property);
}

void setBool(napi_env env, napi_value object, const char* name, bool value) {
    napi_value property;
    napi_get_boolean(env, value, &property);
    napi_set_named_property(env, object, name, property);
}

void setString(napi_env env, napi_value object, const char* name, const std::string& value) {
    napi_value property;
    napi_create_string_utf8(env, value.c_str(), value.size(), &property);
    napi_set_named_property(env, object, name, property);
}

napi_value createBenchmarkSample(napi_env env, const BenchmarkSample& sample) {
    napi_value value;
    napi_create_object(env, &value);
    setInt(env, value, "iteration", sample.iteration);
    setInt(env, value, "promptTokens", sample.promptTokens);
    setInt(env, value, "generatedTokens", sample.generatedTokens);
    setDouble(env, value, "prefillMs", sample.prefillMs);
    setDouble(env, value, "decodeMs", sample.decodeMs);
    setDouble(env, value, "totalMs", sample.totalMs);
    return value;
}

struct LoadWork {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::string configPath;
    std::string runtimeConfigJson;
    std::string error;
    double loadMs = 0.0;
    bool reused = false;
};

// HarmonyOS governs scheduling and cluster frequency ramp-up per thread by QoS
// level; threads inheriting the default level keep mid/big CPU clusters at
// ~60-80% of max frequency even under sustained inference load. Raise every
// thread that runs MNN compute to QOS_USER_INTERACTIVE. Resolved lazily via
// dlopen so the binary keeps loading on OS versions without libqos.so.
void RaiseThreadQoS() {
    typedef int (*QoSFunc)(int);
    static QoSFunc setQoS = []() -> QoSFunc {
        void* handle = dlopen("libqos.so", RTLD_NOW);
        if (handle == nullptr) {
            return nullptr;
        }
        return reinterpret_cast<QoSFunc>(dlsym(handle, "OH_QoS_SetThreadQoS"));
    }();
    if (setQoS != nullptr) {
        setQoS(4); // QOS_USER_INTERACTIVE: boost mid/big clusters to max freq
    }
}

struct BigStackTask {
    void* data;
    void* (*entry)(void*);
};

void* RunBigStackTask(void* raw) {
    std::unique_ptr<BigStackTask> task(static_cast<BigStackTask*>(raw));
    RaiseThreadQoS();
    return task->entry(task->data);
}

void RunOnBigStack(void* data, void* (*entry)(void*)) {
    // Omni talker/DiT graphs recurse deeply; libuv worker stacks are far too small for them.
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, 32 * 1024 * 1024);
    pthread_t thread;
    auto task = new BigStackTask{data, entry};
    if (pthread_create(&thread, &attributes, RunBigStackTask, task) == 0) {
        pthread_join(thread, nullptr);
    } else {
        delete task;
        RaiseThreadQoS();
        entry(data);
    }
    pthread_attr_destroy(&attributes);
}

void* runLoadThread(void* raw) {
    auto work = static_cast<LoadWork*>(raw);
    LlmSession::instance().load(work->configPath, work->runtimeConfigJson, work->loadMs, work->reused,
                                work->error);
    return nullptr;
}

napi_value LoadLlm(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    auto context = std::make_unique<LoadWork>();
    if (argc != 2 || !readString(env, args[0], context->configPath) ||
        !readString(env, args[1], context->runtimeConfigJson)) {
        napi_throw_type_error(env, nullptr, "loadLlm expects configPath and runtimeConfigJson");
        return nullptr;
    }

    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName;
    napi_create_string_utf8(env, "MNN LLM load", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_async_work(
        env, nullptr, resourceName,
        [](napi_env, void* data) {
            RunOnBigStack(data, runLoadThread);
        },
        [](napi_env env, napi_status status, void* data) {
            std::unique_ptr<LoadWork> work(static_cast<LoadWork*>(data));
            if (status == napi_ok && work->error.empty()) {
                napi_value result;
                napi_create_object(env, &result);
                setDouble(env, result, "loadMs", work->loadMs);
                setBool(env, result, "reused", work->reused);
                napi_resolve_deferred(env, work->deferred, result);
            } else {
                napi_reject_deferred(env, work->deferred,
                                     createError(env, work->error.empty() ? "LLM load failed" : work->error));
            }
            napi_delete_async_work(env, work->work);
        },
        context.get(), &context->work);
    napi_queue_async_work(env, context.release()->work);
    return promise;
}

void callChunk(napi_env env, napi_value callback, void*, void* data) {
    std::unique_ptr<std::string> chunk(static_cast<std::string*>(data));
    if (!env || !callback) {
        return;
    }
    napi_value receiver;
    napi_value argument;
    napi_get_undefined(env, &receiver);
    napi_create_string_utf8(env, chunk->c_str(), chunk->size(), &argument);
    napi_call_function(env, receiver, callback, 1, &argument, nullptr);
}

struct GenerateWork {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    napi_threadsafe_function callback = nullptr;
    napi_threadsafe_function audioCallback = nullptr;
    GenerateRequest request;
    GenerateResult result;
    std::string error;
};

void callAudio(napi_env env, napi_value callback, void*, void* data) {
    std::unique_ptr<std::vector<int16_t>> samples(static_cast<std::vector<int16_t>*>(data));
    if (!env || !callback) {
        return;
    }
    napi_value receiver;
    napi_get_undefined(env, &receiver);
    napi_value arrayBuffer;
    void* bufferData = nullptr;
    napi_create_arraybuffer(env, samples->size() * sizeof(int16_t), &bufferData, &arrayBuffer);
    if (bufferData != nullptr && !samples->empty()) {
        memcpy(bufferData, samples->data(), samples->size() * sizeof(int16_t));
    }
    napi_value argument;
    napi_create_typedarray(env, napi_int16_array, samples->size(), arrayBuffer, 0, &argument);
    napi_call_function(env, receiver, callback, 1, &argument, nullptr);
}

void* runGenerateThread(void* raw) {
    auto work = static_cast<GenerateWork*>(raw);
    LlmSession::instance().generate(
        work->request,
        [work](const std::string& chunk) {
            auto payload = new std::string(chunk);
            if (napi_call_threadsafe_function(work->callback, payload, napi_tsfn_nonblocking) != napi_ok) {
                delete payload;
            }
        },
        [work](const int16_t* samples, size_t count) {
            if (work->audioCallback == nullptr) {
                return;
            }
            auto payload = new std::vector<int16_t>(samples, samples + count);
            if (napi_call_threadsafe_function(work->audioCallback, payload, napi_tsfn_nonblocking) != napi_ok) {
                delete payload;
            }
        },
        work->result, work->error);
    napi_release_threadsafe_function(work->callback, napi_tsfn_release);
    work->callback = nullptr;
    if (work->audioCallback) {
        napi_release_threadsafe_function(work->audioCallback, napi_tsfn_release);
        work->audioCallback = nullptr;
    }
    return nullptr;
}

napi_value GenerateLlm(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    auto context = std::make_unique<GenerateWork>();
    napi_valuetype callbackType = napi_undefined;
    if (argc < 2 || !readGenerateRequest(env, args[0], context->request) ||
        napi_typeof(env, args[1], &callbackType) != napi_ok || callbackType != napi_function) {
        napi_throw_type_error(env, nullptr, "generateLlm expects a request object and onChunk callback");
        return nullptr;
    }
    napi_valuetype audioType = napi_undefined;
    const bool hasAudio = argc >= 3 && napi_typeof(env, args[2], &audioType) == napi_ok && audioType == napi_function;

    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName;
    napi_create_string_utf8(env, "MNN LLM generate", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_threadsafe_function(env, args[1], nullptr, resourceName, 0, 1, nullptr, nullptr, nullptr, callChunk,
                                    &context->callback);
    if (hasAudio) {
        napi_create_threadsafe_function(env, args[2], nullptr, resourceName, 0, 1, nullptr, nullptr, nullptr,
                                        callAudio, &context->audioCallback);
    }
    LlmSession::instance().registerRequest(context->request.requestId);
    napi_create_async_work(
        env, nullptr, resourceName,
        [](napi_env, void* data) {
            RunOnBigStack(data, runGenerateThread);
        },
        [](napi_env env, napi_status status, void* data) {
            std::unique_ptr<GenerateWork> work(static_cast<GenerateWork*>(data));
            if (work->callback) {
                napi_release_threadsafe_function(work->callback, napi_tsfn_abort);
            }
            if (work->audioCallback) {
                napi_release_threadsafe_function(work->audioCallback, napi_tsfn_abort);
            }
            if (status == napi_ok && work->error.empty()) {
                napi_value result;
                napi_create_object(env, &result);
                setString(env, result, "requestId", work->result.requestId);
                setString(env, result, "text", work->result.text);
                setInt(env, result, "promptTokens", work->result.promptTokens);
                setInt(env, result, "generatedTokens", work->result.generatedTokens);
                setDouble(env, result, "prefillMs", work->result.prefillMs);
                setDouble(env, result, "decodeMs", work->result.decodeMs);
                setDouble(env, result, "totalMs", work->result.totalMs);
                setDouble(env, result, "firstTokenMs", work->result.firstTokenMs);
                setBool(env, result, "canceled", work->result.canceled);
                napi_resolve_deferred(env, work->deferred, result);
            } else {
                napi_reject_deferred(env, work->deferred,
                                     createError(env, work->error.empty() ? "LLM generation failed" : work->error));
            }
            napi_delete_async_work(env, work->work);
        },
        context.get(), &context->work);
    napi_queue_async_work(env, context.release()->work);
    return promise;
}

void callBenchmarkProgress(napi_env env, napi_value callback, void*, void* data) {
    std::unique_ptr<BenchmarkSample> sample(static_cast<BenchmarkSample*>(data));
    if (!env || !callback) {
        return;
    }
    napi_value receiver;
    napi_get_undefined(env, &receiver);
    napi_value argument = createBenchmarkSample(env, *sample);
    napi_call_function(env, receiver, callback, 1, &argument, nullptr);
}

struct BenchmarkWork {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    napi_threadsafe_function callback = nullptr;
    BenchmarkRequest request;
    BenchmarkResult result;
    std::string error;
};

napi_value RunLlmBenchmark(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    auto context = std::make_unique<BenchmarkWork>();
    napi_valuetype callbackType = napi_undefined;
    if (argc != 2 || !readNamedString(env, args[0], "requestId", context->request.requestId) ||
        !readNamedInt(env, args[0], "promptTokens", context->request.promptTokens) ||
        !readNamedInt(env, args[0], "generateTokens", context->request.generateTokens) ||
        !readNamedInt(env, args[0], "repeats", context->request.repeats) || context->request.requestId.empty() ||
        context->request.promptTokens <= 0 || context->request.generateTokens <= 0 || context->request.repeats <= 0 ||
        context->request.repeats > 20 || context->request.promptTokens > 32768 ||
        context->request.generateTokens > 32768 || napi_typeof(env, args[1], &callbackType) != napi_ok ||
        callbackType != napi_function) {
        napi_throw_type_error(env, nullptr, "runLlmBenchmark expects a valid request and progress callback");
        return nullptr;
    }

    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName;
    napi_create_string_utf8(env, "MNN LLM benchmark", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_threadsafe_function(env, args[1], nullptr, resourceName, 0, 1, nullptr, nullptr, nullptr,
                                    callBenchmarkProgress, &context->callback);
    LlmSession::instance().registerRequest(context->request.requestId);
    napi_create_async_work(
        env, nullptr, resourceName,
        [](napi_env, void* data) {
            auto work = static_cast<BenchmarkWork*>(data);
            RaiseThreadQoS();
            LlmSession::instance().benchmark(
                work->request,
                [work](const BenchmarkSample& sample) {
                    auto payload = new BenchmarkSample(sample);
                    if (napi_call_threadsafe_function(work->callback, payload, napi_tsfn_nonblocking) != napi_ok) {
                        delete payload;
                    }
                },
                work->result, work->error);
            napi_release_threadsafe_function(work->callback, napi_tsfn_release);
            work->callback = nullptr;
        },
        [](napi_env env, napi_status status, void* data) {
            std::unique_ptr<BenchmarkWork> work(static_cast<BenchmarkWork*>(data));
            if (work->callback) {
                napi_release_threadsafe_function(work->callback, napi_tsfn_abort);
            }
            if (status == napi_ok && work->error.empty()) {
                napi_value result;
                napi_create_object(env, &result);
                setString(env, result, "requestId", work->result.requestId);
                setBool(env, result, "canceled", work->result.canceled);
                napi_value samples;
                napi_create_array_with_length(env, work->result.samples.size(), &samples);
                for (size_t index = 0; index < work->result.samples.size(); ++index) {
                    napi_set_element(env, samples, index, createBenchmarkSample(env, work->result.samples[index]));
                }
                napi_set_named_property(env, result, "samples", samples);
                napi_resolve_deferred(env, work->deferred, result);
            } else {
                napi_reject_deferred(env, work->deferred,
                                     createError(env, work->error.empty() ? "LLM benchmark failed" : work->error));
            }
            napi_delete_async_work(env, work->work);
        },
        context.get(), &context->work);
    napi_queue_async_work(env, context.release()->work);
    return promise;
}

napi_value CancelLlm(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    std::string requestId;
    if (argc != 1 || !readString(env, args[0], requestId) || requestId.empty()) {
        napi_throw_type_error(env, nullptr, "cancelLlm expects requestId");
        return nullptr;
    }
    LlmSession::instance().cancel(requestId);
    napi_value result;
    napi_get_undefined(env, &result);
    return result;
}

napi_value ResetLlm(napi_env env, napi_callback_info) {
    LlmSession::instance().reset();
    napi_value result;
    napi_get_undefined(env, &result);
    return result;
}

napi_value ReleaseLlm(napi_env env, napi_callback_info) {
    LlmSession::instance().release();
    napi_value result;
    napi_get_undefined(env, &result);
    return result;
}

void callDiffusionProgress(napi_env env, napi_value callback, void*, void* data) {
    std::unique_ptr<int> progress(static_cast<int*>(data));
    if (!env || !callback || !progress) {
        return;
    }
    napi_value receiver;
    napi_get_undefined(env, &receiver);
    napi_value argument;
    napi_create_int32(env, *progress, &argument);
    napi_call_function(env, receiver, callback, 1, &argument, nullptr);
}

struct DiffusionWork {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    napi_threadsafe_function callback = nullptr;
    DiffusionRequest request;
    DiffusionResult result;
    std::string error;
};

void* runDiffusionThread(void* raw) {
    auto work = static_cast<DiffusionWork*>(raw);
    DiffusionSession::instance().run(
        work->request,
        [work](int progress) {
            auto payload = new int(progress);
            if (napi_call_threadsafe_function(work->callback, payload, napi_tsfn_nonblocking) != napi_ok) {
                delete payload;
            }
        },
        work->result, work->error);
    napi_release_threadsafe_function(work->callback, napi_tsfn_release);
    work->callback = nullptr;
    return nullptr;
}

napi_value RunDiffusion(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    auto context = std::make_unique<DiffusionWork>();
    DiffusionRequest& request = context->request;
    napi_valuetype callbackType = napi_undefined;
    const bool valid = argc == 2 &&
                       readNamedString(env, args[0], "requestId", request.requestId) &&
                       readNamedString(env, args[0], "resourcePath", request.resourcePath) &&
                       readNamedString(env, args[0], "prompt", request.prompt) &&
                       readNamedString(env, args[0], "outputPath", request.outputPath) &&
                       readNamedInt(env, args[0], "modelType", request.modelType) &&
                       readNamedInt(env, args[0], "backendType", request.backendType) &&
                       readNamedInt(env, args[0], "memoryMode", request.memoryMode) &&
                       readNamedInt(env, args[0], "width", request.width) &&
                       readNamedInt(env, args[0], "height", request.height) &&
                       readNamedInt(env, args[0], "steps", request.steps) &&
                       readNamedInt(env, args[0], "seed", request.seed) &&
                       !request.requestId.empty() && !request.resourcePath.empty() && !request.prompt.empty() &&
                       !request.outputPath.empty() && napi_typeof(env, args[1], &callbackType) == napi_ok &&
                       callbackType == napi_function;
    if (!valid) {
        napi_throw_type_error(env, nullptr, "runDiffusion expects a valid request and progress callback");
        return nullptr;
    }
    double cfgScale = 4.5;
    if (readNamedDouble(env, args[0], "cfgScale", cfgScale)) {
        request.cfgScale = static_cast<float>(cfgScale);
    }
    bool useCfg = true;
    if (readNamedBool(env, args[0], "useCfg", useCfg)) {
        request.useCfg = useCfg;
    }
    readNamedString(env, args[0], "negativePrompt", request.negativePrompt);
    readNamedString(env, args[0], "imageInput", request.imageInput);
    if (request.steps < 1) {
        request.steps = 1;
    }
    if (request.steps > 50) {
        request.steps = 50;
    }

    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName;
    napi_create_string_utf8(env, "MNN diffusion", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_threadsafe_function(env, args[1], nullptr, resourceName, 0, 1, nullptr, nullptr, nullptr,
                                    callDiffusionProgress, &context->callback);
    napi_create_async_work(
        env, nullptr, resourceName,
        [](napi_env, void* data) {
            // Diffusion graphs (VAE/DiT) recurse deeply during graph build/execute. The NAPI
            // worker uses the musl default stack (128 KB), which overflows on device, so run the
            // generation on a dedicated thread with a generous stack.
            RunOnBigStack(data, runDiffusionThread);
        },
        [](napi_env env, napi_status status, void* data) {
            std::unique_ptr<DiffusionWork> work(static_cast<DiffusionWork*>(data));
            if (work->callback) {
                napi_release_threadsafe_function(work->callback, napi_tsfn_abort);
            }
            if (status == napi_ok && work->error.empty() && work->result.success) {
                napi_value result;
                napi_create_object(env, &result);
                setString(env, result, "requestId", work->result.requestId);
                setBool(env, result, "success", true);
                setDouble(env, result, "totalMs", work->result.totalMs);
                napi_resolve_deferred(env, work->deferred, result);
            } else {
                napi_reject_deferred(env, work->deferred,
                                     createError(env, work->error.empty() ? "Diffusion generation failed" : work->error));
            }
            napi_delete_async_work(env, work->work);
        },
        context.get(), &context->work);
    napi_queue_async_work(env, context.release()->work);
    return promise;
}

// Standalone executables cannot run on HarmonyOS NEXT (mandatory code signing
// rejects every non-system binary under /data, even signed with a release
// profile), so llm_bench's main is exposed as llm_bench_main and driven here
// inside the signed app process — the CLI-equivalent test channel. stdout and
// stderr are redirected to a cache file for the duration of the run and its
// contents are returned to the caller.
extern "C" int llm_bench_main(int argc, char** argv);
extern "C" int runGemvBWRoofline(int M, int K, int threads, int precision);
extern "C" int runGemmRoofline(int M, int oc, int ic, int threads, int precision, int nbit, int pinMode);
extern "C" int runSdotPeak(int threads, int iters, int core);
extern "C" int runSdotN(int n, int iters, int core, int threads);
extern "C" int runSdotMem(int mode, int iters, int core, int threads);
extern "C" int runTopo();
extern "C" int runCpuset();
extern "C" int runSdotPhys(int threads, int iters, int n);

struct NativeBenchWork {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::string args;
    std::string output;
    std::string error;
    int exitCode = 0;
};

std::atomic<bool> gNativeBenchRunning{false};

std::vector<std::string> tokenizeCommandLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::string current;
    bool inQuotes = false;
    bool hasToken = false;
    for (char ch : line) {
        if (ch == '"') {
            inQuotes = !inQuotes;
            hasToken = true;
            continue;
        }
        if (!inQuotes && (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')) {
            if (hasToken) {
                tokens.push_back(current);
                current.clear();
                hasToken = false;
            }
            continue;
        }
        current.push_back(ch);
        hasToken = true;
    }
    if (hasToken) {
        tokens.push_back(current);
    }
    return tokens;
}

#if defined(__aarch64__)
#include <arm_neon.h>
// Pure NEON FMA throughput microbenchmark, bypassing every MNN kernel: 8
// independent accumulator chains saturate the FMA pipes regardless of latency.
// Answers "does this core's fp16 vector unit actually double fp32 throughput"
// (ARM standard cores do; Kirin TaiShan cores are undocumented).
// fp32 uses inline asm as well: at -O2 the compiler eliminates the intrinsic
// version of this constant-operand FMA chain (measured "inf GFLOPS").
#define MNN_F32_FMA4(acc, b, c) \
    asm volatile("fmla %0.4s, %1.4s, %2.4s" : "+w"(acc) : "w"(b), "w"(c))

static double microbenchFmaFp32(int64_t iters, double* sink) {
    float32x4_t a0 = vdupq_n_f32(1.0001f), a1 = vdupq_n_f32(1.0002f);
    float32x4_t a2 = vdupq_n_f32(1.0003f), a3 = vdupq_n_f32(1.0004f);
    float32x4_t a4 = vdupq_n_f32(1.0005f), a5 = vdupq_n_f32(1.0006f);
    float32x4_t a6 = vdupq_n_f32(1.0007f), a7 = vdupq_n_f32(1.0008f);
    const float32x4_t b = vdupq_n_f32(0.9999f), c = vdupq_n_f32(1.00001f);
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < iters; ++i) {
        MNN_F32_FMA4(a0, b, c); MNN_F32_FMA4(a1, b, c);
        MNN_F32_FMA4(a2, b, c); MNN_F32_FMA4(a3, b, c);
        MNN_F32_FMA4(a4, b, c); MNN_F32_FMA4(a5, b, c);
        MNN_F32_FMA4(a6, b, c); MNN_F32_FMA4(a7, b, c);
    }
    const auto t1 = std::chrono::steady_clock::now();
    *sink = vaddvq_f32(vaddq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)),
                                 vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7))));
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<double>(iters) * 8 * 4 * 2 / sec / 1e9; // chains*lanes*FMA
}

// fp16 vector FMA via inline asm: the fp16 vector-arithmetic intrinsics are
// hidden by arm_neon.h unless the TU is compiled with +fp16, which the app
// module is not. "fmla v.8h" is ARMv8.2 FP16, present on every aarch64
// HarmonyOS device (and gated behind __aarch64__ anyway).
// The TU is compiled with -march=armv8.2-a+fp16 (see CMakeLists) so the
// assembler accepts the .8h form; every aarch64 HarmonyOS device has FP16.
#define MNN_F16_FMA8(acc, b, c) \
    asm volatile("fmla %0.8h, %1.8h, %2.8h" : "+w"(acc) : "w"(b), "w"(c))

static double microbenchFmaFp16(int64_t iters, double* sink) {
    // Reinterpret fp32 lanes as fp16 bit patterns; exact values are irrelevant
    // for throughput, only that they stay finite (no denormal slowdown).
    uint16x8_t a0 = vdupq_n_u16(0x3C00), a1 = vdupq_n_u16(0x3C01);
    uint16x8_t a2 = vdupq_n_u16(0x3C02), a3 = vdupq_n_u16(0x3C03);
    uint16x8_t a4 = vdupq_n_u16(0x3C04), a5 = vdupq_n_u16(0x3C05);
    uint16x8_t a6 = vdupq_n_u16(0x3C06), a7 = vdupq_n_u16(0x3C07);
    const uint16x8_t b = vdupq_n_u16(0x2800), c = vdupq_n_u16(0x1400);
    // Accumulate as raw .8h registers; fmla reads/writes them as fp16.
    asm volatile("" : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3), "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7));
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < iters; ++i) {
        MNN_F16_FMA8(a0, b, c); MNN_F16_FMA8(a1, b, c);
        MNN_F16_FMA8(a2, b, c); MNN_F16_FMA8(a3, b, c);
        MNN_F16_FMA8(a4, b, c); MNN_F16_FMA8(a5, b, c);
        MNN_F16_FMA8(a6, b, c); MNN_F16_FMA8(a7, b, c);
    }
    const auto t1 = std::chrono::steady_clock::now();
    asm volatile("" : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3), "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7));
    *sink = (double)vaddvq_u16(vaddq_u16(vaddq_u16(a0, a1), vaddq_u16(a2, a3)));
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<double>(iters) * 8 * 8 * 2 / sec / 1e9;
}

static void runFmaMicrobench(std::string& out) {
    char line[256];
    double sink = 0.0;
    const int64_t iters = 100000000; // 8 chains * iters FMAs, ~0.3-1s per pass
    for (int pass = 0; pass < 2; ++pass) {
        const double g32 = microbenchFmaFp32(iters, &sink);
        const double g16 = microbenchFmaFp16(iters, &sink);
        snprintf(line, sizeof(line), "pass%d: fp32 %8.2f GFLOPS   fp16 %8.2f GFLOPS   ratio %.3f (sink %g)\n",
                 pass, g32, g16, g16 / g32, sink);
        out += line;
    }
}
#endif

// Minimal OpenCL ABI (Khronos cl.h constant values) queried via dlopen/dlsym,
// so the app TU needs no CL headers. Dumps platform/device capabilities of the
// GPU actually driving the OpenCL backend (Maleoon on Kirin).
static void runClInfo(std::string& out) {
    typedef void* ClPlatform;
    typedef void* ClDevice;
    typedef int (*GetPlatformIDsFn)(unsigned, ClPlatform*, unsigned*);
    typedef int (*GetPlatformInfoFn)(ClPlatform, unsigned, size_t, void*, size_t*);
    typedef int (*GetDeviceIDsFn)(ClPlatform, unsigned long, unsigned, ClDevice*, unsigned*);
    typedef int (*GetDeviceInfoFn)(ClDevice, unsigned, size_t, void*, size_t*);
    const char* libs[] = {"/vendor/lib64/chipsetsdk/libhvgr_v200.so",
                          "/vendor/lib64/chipsetsdk/libGLES_mali.so",
                          "/system/lib64/libGLES_mali.so",
                          "libGLES_mali.so",
                          "/vendor/lib64/chipsetsdk/libEGL_impl.so",
                          "libOpenCL.so"};
    char line[1024];
    void* handle = nullptr;
    const char* usedLib = nullptr;
    for (const char* lib : libs) {
        handle = dlopen(lib, RTLD_NOW);
        if (handle != nullptr) {
            usedLib = lib;
            break;
        }
    }
    if (handle == nullptr) {
        out += "no OpenCL library could be loaded\n";
        return;
    }
    snprintf(line, sizeof(line), "library: %s\n", usedLib);
    out += line;
    auto getPlatformIDs = reinterpret_cast<GetPlatformIDsFn>(dlsym(handle, "clGetPlatformIDs"));
    auto getPlatformInfo = reinterpret_cast<GetPlatformInfoFn>(dlsym(handle, "clGetPlatformInfo"));
    auto getDeviceIDs = reinterpret_cast<GetDeviceIDsFn>(dlsym(handle, "clGetDeviceIDs"));
    auto getDeviceInfo = reinterpret_cast<GetDeviceInfoFn>(dlsym(handle, "clGetDeviceInfo"));
    if (!getPlatformIDs || !getPlatformInfo || !getDeviceIDs || !getDeviceInfo) {
        out += "missing OpenCL entry symbols\n";
        return;
    }
    ClPlatform platform = nullptr;
    unsigned count = 0;
    if (getPlatformIDs(1, &platform, &count) != 0 || count == 0) {
        out += "no OpenCL platform\n";
        return;
    }
    static char buffer[16384];
    auto queryString = [&](auto queryFn, void* object, unsigned key, const char* label) {
        buffer[0] = 0;
        if (queryFn(object, key, sizeof(buffer) - 1, buffer, nullptr) == 0) {
            snprintf(line, sizeof(line), "%s: %s\n", label, buffer);
            out += line;
        } else {
            snprintf(line, sizeof(line), "%s: <query failed>\n", label);
            out += line;
        }
    };
    queryString(getPlatformInfo, platform, 0x0902, "platform name");
    queryString(getPlatformInfo, platform, 0x0901, "platform version");
    queryString(getPlatformInfo, platform, 0x0904, "platform extensions");
    ClDevice device = nullptr;
    if (getDeviceIDs(platform, 4 /*CL_DEVICE_TYPE_GPU*/, 1, &device, &count) != 0) {
        out += "no GPU device\n";
        return;
    }
    queryString(getDeviceInfo, device, 0x102C, "device vendor");
    queryString(getDeviceInfo, device, 0x102B, "device name");
    queryString(getDeviceInfo, device, 0x103F, "device version");
    queryString(getDeviceInfo, device, 0x1046, "driver version");
    auto queryU32 = [&](unsigned key, const char* label) {
        unsigned value = 0;
        if (getDeviceInfo(device, key, sizeof(value), &value, nullptr) == 0) {
            snprintf(line, sizeof(line), "%s: %u\n", label, value);
            out += line;
        }
    };
    auto queryU64 = [&](unsigned key, const char* label) {
        unsigned long long value = 0;
        if (getDeviceInfo(device, key, sizeof(value), &value, nullptr) == 0) {
            snprintf(line, sizeof(line), "%s: %llu\n", label, value);
            out += line;
        }
    };
    queryU32(0x1002, "max compute units");
    queryU32(0x100C, "max clock frequency MHz");
    queryU64(0x1004, "max work-group size");
    queryU64(0x101F, "global mem bytes");
    queryU64(0x1023, "local mem bytes");
    queryU64(0x1010, "max mem alloc bytes");
    queryU32(0x4043, "wavefront width (qcom ext, 0=unsupported)");
    // CL_DEVICE_QUEUE_PROPERTIES / QUEUE_ON_DEVICE_PROPERTIES: bit0 = out-of-order exec,
    // bit1 = profiling. Device-side queues would allow graph-like indirect launch.
    unsigned long long queueProps = 0, deviceQueueProps = 0;
    if (getDeviceInfo(device, 0x102A, sizeof(queueProps), &queueProps, nullptr) == 0) {
        snprintf(line, sizeof(line), "host queue props: 0x%llx (out-of-order=%d, profiling=%d)\n", queueProps,
                 (int)(queueProps & 1), (int)((queueProps >> 1) & 1));
        out += line;
    }
    if (getDeviceInfo(device, 0x104E, sizeof(deviceQueueProps), &deviceQueueProps, nullptr) == 0) {
        snprintf(line, sizeof(line), "device queue props: 0x%llx (on-device queues %s)\n", deviceQueueProps,
                 deviceQueueProps ? "SUPPORTED" : "absent");
        out += line;
    }
    queryString(getDeviceInfo, device, 0x1042, "extensions");
}

// Experimental "@pin=<cpu,cpu,...> " prefix: bind this (enqueue) thread to the
// given cores before running. All OpenCL submission happens on this thread, so
// pinning tests whether the per-kernel launch gap is CPU-side (the enqueueing
// thread landing on slow/low-frequency cores) or driver/GPU-side.
static bool applyPinPrefix(std::string& args) {
    if (args.rfind("@pin=", 0) != 0) {
        return false;
    }
    const size_t space = args.find(' ');
    if (space == std::string::npos) {
        return false;
    }
    const std::string list = args.substr(5, space - 5);
    cpu_set_t set;
    CPU_ZERO(&set);
    size_t pos = 0;
    while (pos < list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) {
            comma = list.size();
        }
        const int cpu = atoi(list.substr(pos, comma - pos).c_str());
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &set);
        }
        pos = comma + 1;
    }
    args = args.substr(space + 1);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// "@pinall=<cpus> " variant: a watchdog thread re-pins EVERY thread of the
// process (including driver-internal threads created after context init and
// lazily spawned MNN workers) to the given cores for the whole run. Used to
// definitively rule in/out CPU placement as a factor for OpenCL launch gaps.
struct PinWatchdog {
    std::atomic<bool> stop{false};
    std::thread thread;
    void start(const std::vector<int>& cpus) {
        thread = std::thread([this, cpus]() {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int cpu : cpus) {
                if (cpu >= 0 && cpu < CPU_SETSIZE) {
                    CPU_SET(cpu, &set);
                }
            }
            while (!stop.load(std::memory_order_relaxed)) {
                DIR* dir = opendir("/proc/self/task");
                if (dir != nullptr) {
                    struct dirent* entry;
                    while ((entry = readdir(dir)) != nullptr) {
                        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') {
                            sched_setaffinity(atoi(entry->d_name), sizeof(set), &set);
                        }
                    }
                    closedir(dir);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }
    ~PinWatchdog() {
        stop.store(true);
        if (thread.joinable()) {
            thread.join();
        }
    }
};

static std::unique_ptr<PinWatchdog> applyPinAllPrefix(std::string& args) {
    if (args.rfind("@pinall=", 0) != 0) {
        return nullptr;
    }
    const size_t space = args.find(' ');
    if (space == std::string::npos) {
        return nullptr;
    }
    std::vector<int> cpus;
    const std::string list = args.substr(8, space - 8);
    size_t pos = 0;
    while (pos < list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) {
            comma = list.size();
        }
        cpus.push_back(atoi(list.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    args = args.substr(space + 1);
    auto watchdog = std::unique_ptr<PinWatchdog>(new PinWatchdog());
    watchdog->start(cpus);
    return watchdog;
}

// @gpubench: measure the GPU's raw fp16/fp32 FMA peak via a runtime-compiled
// OpenCL kernel. Establishes the physical ceiling for prefill throughput
// (llm_bench pp512 GEMM runs at some fraction of this peak).
// Multiplicative recurrences (a = fma(a, b, c_i)) instead of additive ones:
// the Hisilicon compiler folds constant additive FMA loops into closed form
// even without fast-math (measured "0.000 s"), while a power-like recurrence
// with per-chain distinct addends survives. Host reads back out[0] as proof.
static const char* kGpuBenchSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void minimal(__global float* out, const int iters) {
    out[get_global_id(0)] = 42.0f;
}
__kernel void fma16h(__global float* out, const int iters) {
    out[get_global_id(0)] = 42.0f; // magic: proves the kernel body ran at all
    const half16 b = convert_half16((float16)(1.0000001f));
    half16 a0 = convert_half16((float16)(0.90f));
    half16 a1 = convert_half16((float16)(0.91f));
    half16 a2 = convert_half16((float16)(0.92f));
    half16 a3 = convert_half16((float16)(0.93f));
    half16 a4 = convert_half16((float16)(0.94f));
    half16 a5 = convert_half16((float16)(0.95f));
    half16 a6 = convert_half16((float16)(0.96f));
    half16 a7 = convert_half16((float16)(0.97f));
    for (int i = 0; i < iters; ++i) {
        a0 = fma(a0, b, convert_half16((float16)(0.001f)));
        a1 = fma(a1, b, convert_half16((float16)(0.002f)));
        a2 = fma(a2, b, convert_half16((float16)(0.003f)));
        a3 = fma(a3, b, convert_half16((float16)(0.004f)));
        a4 = fma(a4, b, convert_half16((float16)(0.005f)));
        a5 = fma(a5, b, convert_half16((float16)(0.006f)));
        a6 = fma(a6, b, convert_half16((float16)(0.007f)));
        a7 = fma(a7, b, convert_half16((float16)(0.008f)));
    }
    half s = a0.s0 + a1.s0 + a2.s0 + a3.s0 + a4.s0 + a5.s0 + a6.s0 + a7.s0;
    out[get_global_id(0)] = convert_float(s);
}
__kernel void fma4f(__global float* out, const int iters) {
    out[get_global_id(0)] = 42.0f; // magic
    const float4 b = 1.0000001f;
    float4 a0 = 0.90f, a1 = 0.91f, a2 = 0.92f, a3 = 0.93f;
    float4 a4 = 0.94f, a5 = 0.95f, a6 = 0.96f, a7 = 0.97f;
    for (int i = 0; i < iters; ++i) {
        a0 = fma(a0, b, 0.001f); a1 = fma(a1, b, 0.002f);
        a2 = fma(a2, b, 0.003f); a3 = fma(a3, b, 0.004f);
        a4 = fma(a4, b, 0.005f); a5 = fma(a5, b, 0.006f);
        a6 = fma(a6, b, 0.007f); a7 = fma(a7, b, 0.008f);
    }
    out[get_global_id(0)] = a0.x + a1.x + a2.x + a3.x + a4.x + a5.x + a6.x + a7.x;
}
)CLC";

static void runGpuBench(std::string& out) {
    typedef int (*GetPlatformIDsFn)(unsigned, void**, unsigned*);
    typedef int (*GetDeviceIDsFn)(void*, unsigned long long, unsigned, void**, unsigned*);
    typedef void* (*CreateContextFn)(const void*, unsigned, const void*, void*, void*, int*);
    typedef void* (*CreateQueueFn)(void*, void*, unsigned long long, int*);
    typedef void* (*CreateProgramFn)(void*, unsigned, const char**, const size_t*, int*);
    typedef int (*BuildProgramFn)(void*, unsigned, const void*, const char*, void*, void*);
    typedef int (*GetBuildInfoFn)(void*, void*, unsigned, size_t, void*, size_t*);
    typedef void* (*CreateKernelFn)(void*, const char*, int*);
    typedef void* (*CreateBufferFn)(void*, unsigned long long, size_t, void*, int*);
    typedef int (*SetArgFn)(void*, unsigned, size_t, const void*);
    typedef int (*EnqueueFn)(void*, void*, unsigned, const size_t*, const size_t*, const size_t*, unsigned, const void*, void*);
    typedef int (*FinishFn)(void*);
    // Same candidate list as MNN's OpenCLWrapper (OHOS branch) plus the system
    // ICD loader as last resort; log every attempt because the ICD loader
    // enumerates devices fine but turned out to no-op kernel execution.
    const char* libs[] = {"/vendor/lib64/chipsetsdk/libhvgr_v200.so",
                          "/vendor/lib64/chipsetsdk/libGLES_mali.so",
                          "/system/lib64/libGLES_mali.so",
                          "libGLES_mali.so",
                          "/vendor/lib64/chipsetsdk/libEGL_impl.so",
                          "libOpenCL.so"};
    void* lib = nullptr;
    char line[512];
    for (const char* name : libs) {
        lib = dlopen(name, RTLD_NOW);
        snprintf(line, sizeof(line), "gpubench dlopen %s: %s\n", name, lib ? "OK" : dlerror());
        out += line;
        if (lib) {
            break;
        }
    }
    if (!lib) {
        out += "gpubench: no OpenCL library\n";
        return;
    }
    auto getPlatformIDs = (GetPlatformIDsFn)dlsym(lib, "clGetPlatformIDs");
    auto getDeviceIDs = (GetDeviceIDsFn)dlsym(lib, "clGetDeviceIDs");
    auto createContext = (CreateContextFn)dlsym(lib, "clCreateContext");
    auto createQueue = (CreateQueueFn)dlsym(lib, "clCreateCommandQueue");
    auto createProgram = (CreateProgramFn)dlsym(lib, "clCreateProgramWithSource");
    auto buildProgram = (BuildProgramFn)dlsym(lib, "clBuildProgram");
    auto getBuildInfo = (GetBuildInfoFn)dlsym(lib, "clGetProgramBuildInfo");
    auto createKernel = (CreateKernelFn)dlsym(lib, "clCreateKernel");
    auto createBuffer = (CreateBufferFn)dlsym(lib, "clCreateBuffer");
    auto setArg = (SetArgFn)dlsym(lib, "clSetKernelArg");
    auto enqueue = (EnqueueFn)dlsym(lib, "clEnqueueNDRangeKernel");
    auto finish = (FinishFn)dlsym(lib, "clFinish");
    if (!getPlatformIDs || !getDeviceIDs || !createContext || !createQueue || !createProgram || !buildProgram ||
        !createKernel || !createBuffer || !setArg || !enqueue || !finish) {
        out += "gpubench: missing OpenCL symbols\n";
        return;
    }
    void* platform = nullptr;
    unsigned count = 0;
    int err = getPlatformIDs(1, &platform, &count);
    void* device = nullptr;
    if (err != 0 || getDeviceIDs(platform, 4 /*GPU*/, 1, &device, &count) != 0) {
        out += "gpubench: no GPU device\n";
        return;
    }
    void* context = createContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != 0 || !context) {
        snprintf(line, sizeof(line), "gpubench: context failed %d\n", err);
        out += line;
        return;
    }
    void* queue = createQueue(context, device, 2 /*CL_QUEUE_PROFILING_ENABLE*/, &err);
    const char* src = kGpuBenchSource;
    size_t srcLen = strlen(src);
    void* program = createProgram(context, 1, &src, &srcLen, &err);
    // NOTE: no -cl-fast-relaxed-math — it allows reassociation, letting the
    // compiler collapse the FMA loops into closed form (measured "0.000 s").
    int buildErr = buildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (getBuildInfo) {
        char log[2048] = {0};
        unsigned buildStatus = 0;
        getBuildInfo(program, device, 0x1181 /*BUILD_STATUS*/, sizeof(buildStatus), &buildStatus, nullptr);
        getBuildInfo(program, device, 0x1183 /*BUILD_LOG*/, sizeof(log) - 1, log, nullptr);
        snprintf(line, sizeof(line), "gpubench build: err=%d status=%d log=%.800s\n", buildErr, (int)buildStatus, log);
        out += line;
    }
    if (buildErr != 0) {
        return;
    }
    void* buffer = createBuffer(context, 1 /*READ_WRITE*/, 65536 * sizeof(float), nullptr, &err);
    snprintf(line, sizeof(line), "gpubench diag: ctx=%p queue=%p prog=%p buf=%p bufErr=%d\n", context, queue, program,
             buffer, err);
    out += line;
    const size_t global = 4096;
    const size_t local = 256;
    const int iters = 5000;
    struct Case {
        const char* name;
        double flopPerIterPerWi; // 8 chains * lanes * 2
    };
    // fma4f before fma16h: if the fp16 kernel wedges the queue we still get the
    // fp32 number. Each case gets a fresh queue for the same isolation reason.
    const Case cases[] = {{"minimal", 0.0}, {"fma4f", 8.0 * 4 * 2}, {"fma16h", 8.0 * 16 * 2}};
    for (const Case& item : cases) {
        void* caseQueue = createQueue(context, device, 2 /*PROFILING*/, &err);
        if (!caseQueue) {
            snprintf(line, sizeof(line), "gpubench %s: queue recreate failed %d\n", item.name, err);
            out += line;
            continue;
        }
        void* kernel = createKernel(program, item.name, &err);
        if (err != 0 || !kernel) {
            snprintf(line, sizeof(line), "gpubench: kernel %s failed %d\n", item.name, err);
            out += line;
            continue;
        }
        int argErr0 = setArg(kernel, 0, sizeof(void*), &buffer);
        int argErr1 = setArg(kernel, 1, sizeof(int), &iters);
        int enqErr = enqueue(caseQueue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr); // warmup
        int finErr = finish(caseQueue);
        snprintf(line, sizeof(line), "gpubench %s diag: kernel=%p argErr=%d/%d warmEnq=%d warmFin=%d\n", item.name,
                 kernel, argErr0, argErr1, enqErr, finErr);
        out += line;
        if (enqErr != 0 || finErr != 0) {
            snprintf(line, sizeof(line), "gpubench %s: enqueue=%d finish=%d\n", item.name, enqErr, finErr);
            out += line;
            continue;
        }
        typedef int (*GetEventProfilingFn)(void*, unsigned, size_t, void*, size_t*);
        auto getEventProfiling = (GetEventProfilingFn)dlsym(lib, "clGetEventProfilingInfo");
        const auto t0 = std::chrono::steady_clock::now();
        void* event = nullptr;
        enqErr = enqueue(caseQueue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event);
        finErr = finish(caseQueue);
        const auto t1 = std::chrono::steady_clock::now();
        unsigned long long gpuStart = 0, gpuEnd = 0;
        if (event && getEventProfiling) {
            getEventProfiling(event, 0x1191 /*COMMAND_START*/, sizeof(gpuStart), &gpuStart, nullptr);
            getEventProfiling(event, 0x1192 /*COMMAND_END*/, sizeof(gpuEnd), &gpuEnd, nullptr);
        }
        if (enqErr != 0 || finErr != 0) {
            snprintf(line, sizeof(line), "gpubench %s: timed run enqueue=%d finish=%d\n", item.name, enqErr, finErr);
            out += line;
            continue;
        }
        snprintf(line, sizeof(line), "gpubench %s: gpu-event %.3f ms, host-wall %.3f ms\n", item.name,
                 (gpuEnd - gpuStart) / 1e6,
                 std::chrono::duration<double, std::milli>(t1 - t0).count());
        out += line;
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double gflops = static_cast<double>(global) * iters * item.flopPerIterPerWi / sec / 1e9;
        // Read back one element as proof the kernel really executed.
        float sample = 0.0f;
        typedef void* (*MapFn)(void*, void*, unsigned long long, unsigned, unsigned long long, size_t, unsigned,
                               const void*, void*, int*);
        typedef int (*UnmapFn)(void*, void*, void*, unsigned, const void*, void*);
        auto mapBuffer = (MapFn)dlsym(lib, "clEnqueueMapBuffer");
        auto unmapBuffer = (UnmapFn)dlsym(lib, "clEnqueueUnmapMemObject");
        if (mapBuffer && unmapBuffer) {
            int mapErr = 0;
            void* mapped = mapBuffer(caseQueue, buffer, 1 /*BLOCKING*/, 2 /*READ*/, 0, sizeof(float), 0, nullptr, nullptr,
                                     &mapErr);
            if (mapped != nullptr && mapErr == 0) {
                sample = *static_cast<float*>(mapped);
                unmapBuffer(caseQueue, buffer, mapped, 0, nullptr, nullptr);
            } else {
                snprintf(line, sizeof(line), "gpubench %s: map failed err=%d ptr=%p\n", item.name, mapErr, mapped);
                out += line;
            }
        }
        snprintf(line, sizeof(line), "gpubench %s: %8.1f GFLOPS (%.4f s, out[0]=%g)\n", item.name, gflops, sec,
                 (double)sample);
        out += line;
    }
}

// @vkinfo: probe the Vulkan physical device through the loader (dlopen, no
// headers needed): API/driver version, subgroup size/ops, extension list and
// VK_KHR_cooperative_matrix feature/property enumeration — decides whether the
// CoopMat GEMM path (currently gated to Adreno in VulkanConvolution.cpp) can
// be opened for Maleoon.
static void runVkInfo(std::string& out) {
    char line[512];
    void* lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!lib) {
        out += "vkinfo: libvulkan.so not loadable\n";
        return;
    }
    typedef int (*CreateInstanceFn)(const void*, const void*, void**);
    typedef int (*EnumPhysFn)(void*, unsigned*, void**);
    typedef void (*GetPropsFn)(void*, void*);
    typedef void (*GetProps2Fn)(void*, void*);
    typedef void (*GetFeats2Fn)(void*, void*);
    typedef int (*EnumDevExtFn)(void*, const char*, unsigned*, void*);
    typedef void* (*GetInstProcFn)(void*, const char*);
    typedef int (*CoopPropsFn)(void*, unsigned*, void*);
    auto createInstance = (CreateInstanceFn)dlsym(lib, "vkCreateInstance");
    auto enumPhys = (EnumPhysFn)dlsym(lib, "vkEnumeratePhysicalDevices");
    auto getProps = (GetPropsFn)dlsym(lib, "vkGetPhysicalDeviceProperties");
    auto enumDevExt = (EnumDevExtFn)dlsym(lib, "vkEnumerateDeviceExtensionProperties");
    auto getInstProc = (GetInstProcFn)dlsym(lib, "vkGetInstanceProcAddr");
    if (!createInstance || !enumPhys || !getProps || !enumDevExt || !getInstProc) {
        out += "vkinfo: missing loader symbols\n";
        return;
    }
    struct InstanceCreateInfo {
        unsigned sType;
        void* pNext;
        unsigned flags;
        const void* appInfo;
        unsigned layerCount;
        const char* const* layers;
        unsigned extCount;
        const char* const* exts;
    };
    // The pNext chain of vkGetPhysicalDeviceProperties2 (subgroup properties!)
    // may only be filled when the instance enables the KHR extension — verify
    // by enabling it here; MNN's VulkanInstance enables none, which is the
    // suspected reason Maleoon reports subgroupSize=0.
    const char* instExts[] = {"VK_KHR_get_physical_device_properties2"};
    InstanceCreateInfo ici = {1 /*INSTANCE_CREATE_INFO*/, nullptr, 0, nullptr, 0, nullptr, 1, instExts};
    void* instance = nullptr;
    if (createInstance(&ici, nullptr, &instance) != 0) {
        // retry without extensions
        InstanceCreateInfo fallback = {1, nullptr, 0, nullptr, 0, nullptr, 0, nullptr};
        if (createInstance(&fallback, nullptr, &instance) != 0 || !instance) {
            out += "vkinfo: vkCreateInstance failed\n";
            return;
        }
        out += "vkinfo: instance created WITHOUT properties2 extension\n";
    }
    unsigned count = 0;
    enumPhys(instance, &count, nullptr);
    if (count == 0) {
        out += "vkinfo: no physical device\n";
        return;
    }
    std::vector<void*> devices(count, nullptr);
    enumPhys(instance, &count, devices.data());
    void* device = devices[0];

    char props[1200] = {0};
    getProps(device, props);
    unsigned apiVersion = 0, driverVersion = 0, vendorID = 0, deviceID = 0;
    memcpy(&apiVersion, props + 0, 4);
    memcpy(&driverVersion, props + 4, 4);
    memcpy(&vendorID, props + 8, 4);
    memcpy(&deviceID, props + 12, 4);
    snprintf(line, sizeof(line), "apiVersion %u.%u.%u  driverVersion 0x%x  vendorID 0x%x  deviceID 0x%x\n",
             apiVersion >> 22, (apiVersion >> 12) & 0x3ff, apiVersion & 0xfff, driverVersion, vendorID, deviceID);
    out += line;
    snprintf(line, sizeof(line), "deviceName: %.200s\n", props + 20);
    out += line;

    // Subgroup properties via the properties2 chain.
    struct SubgroupProps {
        unsigned sType;
        void* pNext;
        unsigned subgroupSize;
        unsigned long long supportedStages;
        unsigned long long supportedOperations;
        unsigned quadOperationsInAllStages;
    } subgroup = {1000094000, nullptr, 0, 0, 0, 0};
    struct Props2 {
        unsigned sType;
        void* pNext;
        char base[1200];
    } props2 = {1000059001, &subgroup, {0}};
    auto getProps2 = (GetProps2Fn)getInstProc(instance, "vkGetPhysicalDeviceProperties2");
    if (!getProps2) {
        getProps2 = (GetProps2Fn)getInstProc(instance, "vkGetPhysicalDeviceProperties2KHR");
        if (getProps2) {
            out += "vkinfo: using KHR entry for properties2\n";
        }
    }
    if (getProps2) {
        getProps2(device, &props2);
        snprintf(line, sizeof(line), "subgroupSize=%u stages=0x%llx ops=0x%llx quadAllStages=%u\n",
                 subgroup.subgroupSize, subgroup.supportedStages, subgroup.supportedOperations,
                 subgroup.quadOperationsInAllStages);
        out += line;
    } else {
        out += "no vkGetPhysicalDeviceProperties2\n";
    }

    // Device extensions.
    unsigned extCount = 0;
    enumDevExt(device, nullptr, &extCount, nullptr);
    std::vector<char> extBuffer(static_cast<size_t>(extCount) * 260, 0);
    bool hasCoopMat = false;
    if (extCount > 0 && enumDevExt(device, nullptr, &extCount, extBuffer.data()) == 0) {
        out += "device extensions:\n";
        for (unsigned i = 0; i < extCount; ++i) {
            const char* name = extBuffer.data() + static_cast<size_t>(i) * 260;
            snprintf(line, sizeof(line), "  %s\n", name);
            out += line;
            if (strstr(name, "cooperative_matrix")) {
                hasCoopMat = true;
            }
        }
    }

    // Cooperative matrix features + supported M/N/K combinations.
    struct CoopMatFeatures {
        unsigned sType;
        void* pNext;
        unsigned cooperativeMatrix;
        unsigned robustBufferAccess;
    } coopFeatures = {1000506000, nullptr, 0, 0};
    struct Features2 {
        unsigned sType;
        void* pNext;
        char base[512];
    } features2 = {1000059000, &coopFeatures, {0}};
    auto getFeats2 = (GetFeats2Fn)getInstProc(instance, "vkGetPhysicalDeviceFeatures2");
    if (!getFeats2) {
        getFeats2 = (GetFeats2Fn)getInstProc(instance, "vkGetPhysicalDeviceFeatures2KHR");
    }
    if (getFeats2) {
        getFeats2(device, &features2);
        snprintf(line, sizeof(line), "coopmat feature: supported=%u robustBufferAccess=%u\n",
                 coopFeatures.cooperativeMatrix, coopFeatures.robustBufferAccess);
        out += line;
    }
    if (hasCoopMat) {
        auto coopProps = (CoopPropsFn)getInstProc(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (!coopProps) {
            out += "coopmat ext present but entry point missing\n";
        } else {
            unsigned propCount = 0;
            coopProps(device, &propCount, nullptr);
            snprintf(line, sizeof(line), "coopmat combos: %u\n", propCount);
            out += line;
            // sizeof(VkCooperativeMatrixPropertiesKHR) == 56 (11 fields, align 8).
            std::vector<unsigned long long> propBuffer(static_cast<size_t>(propCount) * 7, 0);
            unsigned shown = 0;
            if (propCount > 0 && coopProps(device, &propCount, propBuffer.data()) == 0) {
                for (unsigned i = 0; i < propCount && shown < 24; ++i) {
                    const unsigned* p = reinterpret_cast<const unsigned*>(propBuffer.data() + static_cast<size_t>(i) * 7);
                    // sType,pNext(2 words skip), MSize,NSize,KSize,AType,BType,CType,ResultType,sat,scope
                    snprintf(line, sizeof(line), "  M%u N%u K%u A=%u B=%u C=%u R=%u sat=%u scope=%u\n", p[4], p[5],
                             p[6], p[7], p[8], p[9], p[10], p[11], p[12]);
                    out += line;
                    ++shown;
                }
            }
        }
    }
}

// @vksubgroup: run a minimal compute dispatch and read gl_SubgroupSize from the
// GPU itself — Maleoon's vkGetPhysicalDeviceProperties2 chain returns zeros for
// VkPhysicalDeviceSubgroupProperties, so the real value can only be measured.
#include "probe_spv.h"

namespace vkprobe {
struct DeviceQueueCI { unsigned sType; const void* pNext; unsigned flags; unsigned queueFamily; unsigned queueCount;
                       const float* pPriorities; };
struct DeviceCI { unsigned sType; const void* pNext; unsigned flags; unsigned queueCount; const DeviceQueueCI* queues;
                  unsigned layerCount; const char* const* layers; unsigned extCount; const char* const* exts;
                  const void* features; };
struct BufferCI { unsigned sType; const void* pNext; unsigned flags; unsigned long long size; unsigned usage;
                  unsigned sharingMode; unsigned qfCount; const unsigned* qf; };
struct MemReqs { unsigned long long size; unsigned long long alignment; unsigned memoryTypeBits; };
struct MemAllocInfo { unsigned sType; const void* pNext; unsigned long long size; unsigned memoryTypeIndex; };
struct ShaderModuleCI { unsigned sType; const void* pNext; unsigned flags; unsigned long long codeSize;
                        const void* pCode; };
struct DsLayoutBinding { unsigned binding; unsigned descriptorType; unsigned descriptorCount; unsigned stageFlags;
                         const void* pImmutableSamplers; };
struct DsLayoutCI { unsigned sType; const void* pNext; unsigned flags; unsigned bindingCount;
                    const DsLayoutBinding* pBindings; };
struct PipelineLayoutCI { unsigned sType; const void* pNext; unsigned flags; unsigned setLayoutCount;
                          const void* pSetLayouts; unsigned pushCount; const void* pPush; };
struct StageCI { unsigned sType; const void* pNext; unsigned flags; unsigned stage; void* module; const char* pName;
                 const void* pSpec; };
struct ComputePipelineCI { unsigned sType; const void* pNext; unsigned long long flags; StageCI stage; void* layout;
                           void* baseHandle; int baseIndex; };
struct DescPoolSize { unsigned type; unsigned count; };
struct DescPoolCI { unsigned sType; const void* pNext; unsigned flags; unsigned maxSets; unsigned sizeCount;
                    const DescPoolSize* pSizes; };
struct DescSetAllocInfo { unsigned sType; const void* pNext; void* pool; unsigned setCount; const void* pSetLayouts; };
struct DescBufferInfo { void* buffer; unsigned long long offset; unsigned long long range; };
struct WriteDescSet { unsigned sType; const void* pNext; void* dstSet; unsigned binding; unsigned arrayElement;
                      unsigned count; unsigned type; const void* imageInfo; const DescBufferInfo* bufferInfo;
                      const void* texelView; };
struct CmdPoolCI { unsigned sType; const void* pNext; unsigned flags; unsigned queueFamily; };
struct CmdAllocInfo { unsigned sType; const void* pNext; void* pool; unsigned level; unsigned count; };
struct CmdBeginInfo { unsigned sType; const void* pNext; unsigned flags; const void* pInheritance; };
struct SubmitInfo { unsigned sType; const void* pNext; unsigned waitSemCount; const void* waitSems;
                    const void* waitMasks; unsigned cmdCount; const void* cmdBufs; unsigned sigSemCount;
                    const void* sigSems; };
struct FenceCI { unsigned sType; const void* pNext; unsigned flags; };
struct QueueFamilyProps { unsigned queueFlags; unsigned queueCount; unsigned timestampValidBits;
                          unsigned granularity[3]; };
} // namespace vkprobe

static void runVkSubgroupProbe(std::string& out) {
    using namespace vkprobe;
    char line[256];
    void* lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!lib) { out += "probe: no libvulkan\n"; return; }
    // dlsym each entry point with an explicit typedef to avoid header deps.
    typedef int (*CreateInstanceFn)(const void*, const void*, void**);
    typedef int (*EnumPhysFn)(void*, unsigned*, void**);
    typedef void (*GetQueueFamFn)(void*, unsigned*, void*);
    typedef int (*CreateDeviceFn)(void*, const DeviceCI*, const void*, void**);
    typedef void (*GetDeviceQueueFn)(void*, unsigned, unsigned, void**);
    typedef int (*CreateBufferFn)(void*, const BufferCI*, const void*, void**);
    typedef void (*GetBufMemReqFn)(void*, void*, void*);
    typedef int (*AllocMemFn)(void*, const MemAllocInfo*, const void*, void**);
    typedef int (*BindBufMemFn)(void*, void*, void*, unsigned long long);
    typedef int (*MapMemFn)(void*, void*, unsigned long long, unsigned long long, unsigned, void**);
    typedef int (*CreateShaderFn)(void*, const ShaderModuleCI*, const void*, void**);
    typedef int (*CreateDsLayoutFn)(void*, const DsLayoutCI*, const void*, void**);
    typedef int (*CreatePipeLayoutFn)(void*, const PipelineLayoutCI*, const void*, void**);
    typedef int (*CreateDescPoolFn)(void*, const DescPoolCI*, const void*, void**);
    typedef int (*AllocDescSetsFn)(void*, const DescSetAllocInfo*, void**);
    typedef void (*UpdateDescSetsFn)(void*, unsigned, const WriteDescSet*, unsigned, const void*);
    typedef int (*CreatePipelinesFn)(void*, void*, unsigned, const ComputePipelineCI*, const void*, void**);
    typedef int (*CreateCmdPoolFn)(void*, const CmdPoolCI*, const void*, void**);
    typedef int (*AllocCmdBufsFn)(void*, const CmdAllocInfo*, void**);
    typedef int (*BeginCmdFn)(void*, const CmdBeginInfo*);
    typedef void (*CmdBindPipelineFn)(void*, unsigned, void*);
    typedef void (*CmdBindDescFn)(void*, unsigned, void*, unsigned, unsigned, const void*, unsigned, const unsigned*);
    typedef void (*CmdDispatchFn)(void*, unsigned, unsigned, unsigned);
    typedef int (*EndCmdFn)(void*);
    typedef int (*CreateFenceFn)(void*, const FenceCI*, const void*, void**);
    typedef int (*QueueSubmitFn)(void*, unsigned, const SubmitInfo*, void*);
    typedef int (*WaitFencesFn)(void*, unsigned, const void*, unsigned, unsigned long long);
    auto createInstance = (CreateInstanceFn)dlsym(lib, "vkCreateInstance");
    auto enumPhys = (EnumPhysFn)dlsym(lib, "vkEnumeratePhysicalDevices");
    auto getQueueFam = (GetQueueFamFn)dlsym(lib, "vkGetPhysicalDeviceQueueFamilyProperties");
    auto createDevice = (CreateDeviceFn)dlsym(lib, "vkCreateDevice");
    auto getDeviceQueue = (GetDeviceQueueFn)dlsym(lib, "vkGetDeviceQueue");
    auto createBuffer = (CreateBufferFn)dlsym(lib, "vkCreateBuffer");
    auto getBufMemReq = (GetBufMemReqFn)dlsym(lib, "vkGetBufferMemoryRequirements");
    auto allocMem = (AllocMemFn)dlsym(lib, "vkAllocateMemory");
    auto bindBufMem = (BindBufMemFn)dlsym(lib, "vkBindBufferMemory");
    auto mapMem = (MapMemFn)dlsym(lib, "vkMapMemory");
    auto createShader = (CreateShaderFn)dlsym(lib, "vkCreateShaderModule");
    auto createDsLayout = (CreateDsLayoutFn)dlsym(lib, "vkCreateDescriptorSetLayout");
    auto createPipeLayout = (CreatePipeLayoutFn)dlsym(lib, "vkCreatePipelineLayout");
    auto createDescPool = (CreateDescPoolFn)dlsym(lib, "vkCreateDescriptorPool");
    auto allocDescSets = (AllocDescSetsFn)dlsym(lib, "vkAllocateDescriptorSets");
    auto updateDescSets = (UpdateDescSetsFn)dlsym(lib, "vkUpdateDescriptorSets");
    auto createPipelines = (CreatePipelinesFn)dlsym(lib, "vkCreateComputePipelines");
    auto createCmdPool = (CreateCmdPoolFn)dlsym(lib, "vkCreateCommandPool");
    auto allocCmdBufs = (AllocCmdBufsFn)dlsym(lib, "vkAllocateCommandBuffers");
    auto beginCmd = (BeginCmdFn)dlsym(lib, "vkBeginCommandBuffer");
    auto cmdBindPipeline = (CmdBindPipelineFn)dlsym(lib, "vkCmdBindPipeline");
    auto cmdBindDesc = (CmdBindDescFn)dlsym(lib, "vkCmdBindDescriptorSets");
    auto cmdDispatch = (CmdDispatchFn)dlsym(lib, "vkCmdDispatch");
    auto endCmd = (EndCmdFn)dlsym(lib, "vkEndCommandBuffer");
    auto createFence = (CreateFenceFn)dlsym(lib, "vkCreateFence");
    auto queueSubmit = (QueueSubmitFn)dlsym(lib, "vkQueueSubmit");
    auto waitFences = (WaitFencesFn)dlsym(lib, "vkWaitForFences");
    if (!createInstance || !createDevice || !createPipelines || !queueSubmit || !waitFences) {
        out += "probe: missing symbols\n";
        return;
    }
    struct InstanceCI { unsigned sType; const void* pNext; unsigned flags; const void* appInfo; unsigned layerCount;
                        const char* const* layers; unsigned extCount; const char* const* exts; };
    InstanceCI ici = {1, nullptr, 0, nullptr, 0, nullptr, 0, nullptr};
    void* instance = nullptr;
    if (createInstance(&ici, nullptr, &instance) != 0) { out += "probe: instance failed\n"; return; }
    unsigned count = 0;
    enumPhys(instance, &count, nullptr);
    std::vector<void*> phys(count, nullptr);
    enumPhys(instance, &count, phys.data());
    void* gpu = phys[0];
    // Pick a compute-capable queue family.
    unsigned famCount = 0;
    getQueueFam(gpu, &famCount, nullptr);
    std::vector<QueueFamilyProps> fams(famCount);
    getQueueFam(gpu, &famCount, fams.data());
    unsigned family = 0xffffffff;
    for (unsigned i = 0; i < famCount; ++i) {
        if ((fams[i].queueFlags & 2) && fams[i].queueCount > 0) { family = i; break; }
    }
    if (family == 0xffffffff) { out += "probe: no compute queue family\n"; return; }
    float priority = 1.0f;
    DeviceQueueCI qci = {2, nullptr, 0, family, 1, &priority};
    DeviceCI dci = {3, nullptr, 0, 1, &qci, 0, nullptr, 0, nullptr, nullptr};
    void* device = nullptr;
    if (createDevice(gpu, &dci, nullptr, &device) != 0) { out += "probe: device failed\n"; return; }
    void* queue = nullptr;
    getDeviceQueue(device, family, 0, &queue);

    BufferCI bci = {12, nullptr, 0, 256, 0x20 /*STORAGE*/, 0, 0, nullptr};
    void* buffer = nullptr;
    createBuffer(device, &bci, nullptr, &buffer);
    MemReqs reqs = {0, 0, 0};
    getBufMemReq(device, buffer, &reqs);
    // Brute-force a host-visible+mappable memory type among the allowed bits.
    void* memory = nullptr;
    unsigned chosenType = 0xffffffff;
    for (unsigned t = 0; t < 32 && memory == nullptr; ++t) {
        if (!(reqs.memoryTypeBits & (1u << t))) continue;
        MemAllocInfo mai = {5, nullptr, reqs.size, t};
        if (allocMem(device, &mai, nullptr, &memory) != 0) { memory = nullptr; continue; }
        void* mapped = nullptr;
        if (mapMem(device, memory, 0, reqs.size, 0, &mapped) != 0 || !mapped) {
            memory = nullptr;
            continue;
        }
        chosenType = t;
    }
    if (!memory) { out += "probe: no host-visible memory type\n"; return; }
    bindBufMem(device, buffer, memory, 0);
    void* mapped = nullptr;
    mapMem(device, memory, 0, 256, 0, &mapped);
    memset(mapped, 0, 256);

    ShaderModuleCI smci = {16, nullptr, 0, kProbeSpvLen, kProbeSpv};
    void* shaderModule = nullptr;
    if (createShader(device, &smci, nullptr, &shaderModule) != 0) { out += "probe: shader module failed\n"; return; }
    DsLayoutBinding binding = {0, 6 /*STORAGE_BUFFER*/, 1, 0x20 /*COMPUTE*/, nullptr};
    DsLayoutCI dslci = {32, nullptr, 0, 1, &binding};
    void* dsLayout = nullptr;
    createDsLayout(device, &dslci, nullptr, &dsLayout);
    PipelineLayoutCI plci = {30, nullptr, 0, 1, &dsLayout, 0, nullptr};
    void* pipeLayout = nullptr;
    createPipeLayout(device, &plci, nullptr, &pipeLayout);
    DescPoolSize poolSize = {6, 1};
    DescPoolCI dpci = {33, nullptr, 0, 1, 1, &poolSize};
    void* descPool = nullptr;
    createDescPool(device, &dpci, nullptr, &descPool);
    DescSetAllocInfo dsai = {34, nullptr, descPool, 1, &dsLayout};
    void* descSet = nullptr;
    allocDescSets(device, &dsai, &descSet);
    DescBufferInfo dbi = {buffer, 0, 256};
    WriteDescSet wds = {35, nullptr, descSet, 0, 0, 1, 6, nullptr, &dbi, nullptr};
    updateDescSets(device, 1, &wds, 0, nullptr);
    StageCI stage = {18, nullptr, 0, 0x20 /*COMPUTE*/, shaderModule, "main", nullptr};
    ComputePipelineCI cpci = {29, nullptr, 0, stage, pipeLayout, nullptr, -1};
    void* pipeline = nullptr;
    if (createPipelines(device, nullptr, 1, &cpci, nullptr, &pipeline) != 0) {
        out += "probe: pipeline failed\n";
        return;
    }
    CmdPoolCI cpool = {39, nullptr, 0, family};
    void* cmdPool = nullptr;
    createCmdPool(device, &cpool, nullptr, &cmdPool);
    CmdAllocInfo cai = {40, nullptr, cmdPool, 0, 1};
    void* cmd = nullptr;
    allocCmdBufs(device, &cai, &cmd);
    CmdBeginInfo cbi = {42, nullptr, 1 /*ONE_TIME_SUBMIT*/, nullptr};
    beginCmd(cmd, &cbi);
    cmdBindPipeline(cmd, 1 /*COMPUTE*/, pipeline);
    cmdBindDesc(cmd, 1 /*COMPUTE*/, pipeLayout, 0, 1, &descSet, 0, nullptr);
    cmdDispatch(cmd, 8, 1, 1); // 8 workgroups * 128 lanes = 1024 invocations
    endCmd(cmd);
    FenceCI fci = {8, nullptr, 0};
    void* fence = nullptr;
    createFence(device, &fci, nullptr, &fence);
    SubmitInfo si = {4, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
    int submitErr = queueSubmit(queue, 1, &si, fence);
    int waitErr = submitErr == 0 ? waitFences(device, 1, &fence, 1, 0xFFFFFFFFFFFFFFFFull) : -1;
    unsigned* result = static_cast<unsigned*>(mapped);
    snprintf(line, sizeof(line),
             "probe: submit=%d wait=%d memType=%u gl_SubgroupSize=%u numSubgroups=%u (expect %u for 1024 lanes)\n",
             submitErr, waitErr, chosenType, result[0], result[1],
             result[0] ? 1024u / result[0] : 0u);
    out += line;
}

// @vkapi / @ocapi: driver API cost quantification with EMPTY kernels.
// Separates (a) CPU-side record/enqueue cost per command and (b) GPU-side
// front-end cost per dispatch/kernel, isolated from any real shader work.
#include "empty_spv.h"
#include "empty2_spv.h"

namespace vkapi {
typedef int (*CreateInstanceFn)(const void*, const void*, void**);
typedef int (*EnumPhysFn)(void*, unsigned*, void**);
typedef void (*GetQueueFamFn)(void*, unsigned*, void*);
typedef void (*GetMemPropsFn)(void*, void*);
typedef int (*CreateDeviceFn)(void*, const vkprobe::DeviceCI*, const void*, void**);
typedef void (*GetDeviceQueueFn)(void*, unsigned, unsigned, void**);
typedef int (*CreateBufferFn)(void*, const vkprobe::BufferCI*, const void*, void**);
typedef void (*GetBufMemReqFn)(void*, void*, void*);
typedef int (*AllocMemFn)(void*, const vkprobe::MemAllocInfo*, const void*, void**);
typedef int (*BindBufMemFn)(void*, void*, void*, unsigned long long);
typedef int (*CreateShaderFn)(void*, const vkprobe::ShaderModuleCI*, const void*, void**);
typedef int (*CreateDsLayoutFn)(void*, const vkprobe::DsLayoutCI*, const void*, void**);
typedef int (*CreatePipeLayoutFn)(void*, const vkprobe::PipelineLayoutCI*, const void*, void**);
typedef int (*CreateDescPoolFn)(void*, const vkprobe::DescPoolCI*, const void*, void**);
typedef int (*AllocDescSetsFn)(void*, const vkprobe::DescSetAllocInfo*, void**);
typedef void (*UpdateDescSetsFn)(void*, unsigned, const vkprobe::WriteDescSet*, unsigned, const void*);
typedef int (*CreatePipelinesFn)(void*, void*, unsigned, const vkprobe::ComputePipelineCI*, const void*, void**);
typedef int (*CreateCmdPoolFn)(void*, const vkprobe::CmdPoolCI*, const void*, void**);
typedef int (*AllocCmdBufsFn)(void*, const vkprobe::CmdAllocInfo*, void**);
typedef int (*BeginCmdFn)(void*, const vkprobe::CmdBeginInfo*);
typedef void (*CmdBindPipelineFn)(void*, unsigned, void*);
typedef void (*CmdBindDescFn)(void*, unsigned, void*, unsigned, unsigned, const void*, unsigned, const unsigned*);
typedef void (*CmdDispatchFn)(void*, unsigned, unsigned, unsigned);
typedef int (*EndCmdFn)(void*);
typedef int (*CreateFenceFn)(void*, const vkprobe::FenceCI*, const void*, void**);
typedef int (*ResetFencesFn)(void*, unsigned, const void*);
typedef int (*QueueSubmitFn)(void*, unsigned, const vkprobe::SubmitInfo*, void*);
typedef int (*WaitFencesFn)(void*, unsigned, const void*, unsigned, unsigned long long);
} // namespace vkapi

static void runVkApiBench(std::string& out) {
    using namespace vkprobe;
    using namespace vkapi;
    auto emit = [&out](const std::string& s) {
        out += s;
        MNN_PRINT("%s", s.c_str()); // live trace: survives a later crash
    };
    char line[256];
    auto tms = [](const std::chrono::steady_clock::time_point& a, const std::chrono::steady_clock::time_point& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    void* lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!lib) { emit("vkapi: no libvulkan\n"); return; }
#define VKAPI_SYM(type, name) auto name = (type)dlsym(lib, #name); \
    if (!name) { emit("vkapi: missing " #name "\n"); return; }
    VKAPI_SYM(CreateInstanceFn, vkCreateInstance)
    VKAPI_SYM(EnumPhysFn, vkEnumeratePhysicalDevices)
    VKAPI_SYM(GetQueueFamFn, vkGetPhysicalDeviceQueueFamilyProperties)
    VKAPI_SYM(GetMemPropsFn, vkGetPhysicalDeviceMemoryProperties)
    VKAPI_SYM(CreateDeviceFn, vkCreateDevice)
    VKAPI_SYM(GetDeviceQueueFn, vkGetDeviceQueue)
    VKAPI_SYM(CreateBufferFn, vkCreateBuffer)
    VKAPI_SYM(GetBufMemReqFn, vkGetBufferMemoryRequirements)
    VKAPI_SYM(AllocMemFn, vkAllocateMemory)
    VKAPI_SYM(BindBufMemFn, vkBindBufferMemory)
    VKAPI_SYM(CreateShaderFn, vkCreateShaderModule)
    VKAPI_SYM(CreateDsLayoutFn, vkCreateDescriptorSetLayout)
    VKAPI_SYM(CreatePipeLayoutFn, vkCreatePipelineLayout)
    VKAPI_SYM(CreateDescPoolFn, vkCreateDescriptorPool)
    VKAPI_SYM(AllocDescSetsFn, vkAllocateDescriptorSets)
    VKAPI_SYM(UpdateDescSetsFn, vkUpdateDescriptorSets)
    VKAPI_SYM(CreatePipelinesFn, vkCreateComputePipelines)
    VKAPI_SYM(CreateCmdPoolFn, vkCreateCommandPool)
    VKAPI_SYM(AllocCmdBufsFn, vkAllocateCommandBuffers)
    VKAPI_SYM(BeginCmdFn, vkBeginCommandBuffer)
    VKAPI_SYM(CmdBindPipelineFn, vkCmdBindPipeline)
    VKAPI_SYM(CmdBindDescFn, vkCmdBindDescriptorSets)
    VKAPI_SYM(CmdDispatchFn, vkCmdDispatch)
    VKAPI_SYM(EndCmdFn, vkEndCommandBuffer)
    VKAPI_SYM(CreateFenceFn, vkCreateFence)
    VKAPI_SYM(ResetFencesFn, vkResetFences)
    VKAPI_SYM(QueueSubmitFn, vkQueueSubmit)
    VKAPI_SYM(WaitFencesFn, vkWaitForFences)
#undef VKAPI_SYM

    struct InstanceCI { unsigned sType; const void* pNext; unsigned flags; const void* appInfo;
                        unsigned layerCount; const char* const* layers; unsigned extCount; const char* const* exts; };
    const char* instExts[] = {"VK_KHR_get_physical_device_properties2"};
    InstanceCI ici = {1, nullptr, 0, nullptr, 0, nullptr, 1, instExts};
    void* instance = nullptr;
    if (vkCreateInstance(&ici, nullptr, &instance) != 0) { emit("vkapi: instance failed\n"); return; }
    unsigned count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<void*> phys(count, nullptr);
    vkEnumeratePhysicalDevices(instance, &count, phys.data());
    if (count == 0) { emit("vkapi: no gpu\n"); return; }
    void* gpu = phys[0];
    unsigned famCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &famCount, nullptr);
    std::vector<QueueFamilyProps> fams(famCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &famCount, fams.data());
    unsigned family = 0xffffffff;
    for (unsigned i = 0; i < famCount; ++i) {
        if ((fams[i].queueFlags & 2) && fams[i].queueCount > 0) { family = i; break; }
    }
    if (family == 0xffffffff) { emit("vkapi: no compute family\n"); return; }
    float priority = 1.0f;
    DeviceQueueCI qci = {2, nullptr, 0, family, 1, &priority};
    DeviceCI dci = {3, nullptr, 0, 1, &qci, 0, nullptr, 0, nullptr, nullptr};
    void* device = nullptr;
    if (vkCreateDevice(gpu, &dci, nullptr, &device) != 0) { emit("vkapi: device failed\n"); return; }
    void* queue = nullptr;
    vkGetDeviceQueue(device, family, 0, &queue);

    // Host-visible memory type via the proper memory-properties query.
    char memProps[1024] = {0};
    vkGetPhysicalDeviceMemoryProperties(gpu, memProps);
    unsigned typeCount = 0;
    memcpy(&typeCount, memProps, 4);
    unsigned heapCount = 0;
    memcpy(&heapCount, memProps + 4 + 32 * 8, 4);
    for (unsigned i = 0; i < typeCount && i < 32; ++i) {
        unsigned flags = 0, heapIdx = 0;
        memcpy(&flags, memProps + 4 + i * 8, 4);
        memcpy(&heapIdx, memProps + 4 + i * 8 + 4, 4);
        snprintf(line, sizeof(line), "vkapi memtype %2u: flags=0x%x [%s%s%s%s%s] heap=%u\n", i, flags,
                 (flags & 1) ? "DEVICE_LOCAL " : "", (flags & 2) ? "HOST_VISIBLE " : "",
                 (flags & 4) ? "HOST_COHERENT " : "", (flags & 8) ? "HOST_CACHED " : "",
                 (flags & 16) ? "LAZILY_ALLOCATED " : "", heapIdx);
        emit(line);
    }
    for (unsigned h = 0; h < heapCount && h < 16; ++h) {
        unsigned long long hsize = 0;
        unsigned hflags = 0;
        memcpy(&hsize, memProps + 4 + 32 * 8 + 4 + h * 16, 8);
        memcpy(&hflags, memProps + 4 + 32 * 8 + 4 + h * 16 + 8, 4);
        snprintf(line, sizeof(line), "vkapi heap %u: %llu MB flags=0x%x\n", h, hsize >> 20, hflags);
        emit(line);
    }
    unsigned hostType = 0xffffffff;
    for (unsigned i = 0; i < typeCount && i < 32; ++i) {
        unsigned flags = 0;
        memcpy(&flags, memProps + 4 + i * 8, 4);
        if ((flags & 3) == 3) { hostType = i; break; } // HOST_VISIBLE | HOST_COHERENT
    }
    if (hostType == 0xffffffff) { emit("vkapi: no host-visible memory\n"); return; }

    BufferCI bci = {12, nullptr, 0, 256, 0x20 /*STORAGE*/, 0, 0, nullptr};
    void* buffer = nullptr;
    vkCreateBuffer(device, &bci, nullptr, &buffer);
    MemReqs reqs = {0, 0, 0};
    vkGetBufferMemoryRequirements(device, buffer, &reqs);
    MemAllocInfo mai = {5, nullptr, reqs.size, hostType};
    void* memory = nullptr;
    if (vkAllocateMemory(device, &mai, nullptr, &memory) != 0) { emit("vkapi: alloc failed\n"); return; }
    vkBindBufferMemory(device, buffer, memory, 0);

    ShaderModuleCI smci = {16, nullptr, 0, kEmptySpvLen, kEmptySpv};
    void* shaderModule = nullptr;
    if (vkCreateShaderModule(device, &smci, nullptr, &shaderModule) != 0) { emit("vkapi: shader failed\n"); return; }
    DsLayoutBinding binding = {0, 6, 1, 0x20, nullptr};
    DsLayoutCI dslci = {32, nullptr, 0, 1, &binding};
    void* dsLayout = nullptr;
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsLayout);
    PipelineLayoutCI plci = {30, nullptr, 0, 1, &dsLayout, 0, nullptr};
    void* pipeLayout = nullptr;
    vkCreatePipelineLayout(device, &plci, nullptr, &pipeLayout);
    DescPoolSize poolSize = {6, 4};
    DescPoolCI dpci = {33, nullptr, 0, 4, 1, &poolSize};
    void* descPool = nullptr;
    vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);
    DescSetAllocInfo dsai = {34, nullptr, descPool, 1, &dsLayout};
    void* descSet = nullptr;
    vkAllocateDescriptorSets(device, &dsai, &descSet);
    DescBufferInfo dbi = {buffer, 0, 256};
    WriteDescSet wds = {35, nullptr, descSet, 0, 0, 1, 6, nullptr, &dbi, nullptr};
    vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    StageCI stage = {18, nullptr, 0, 0x20, shaderModule, "main", nullptr};
    ComputePipelineCI cpci = {29, nullptr, 0, stage, pipeLayout, nullptr, -1};
    void* pipeline = nullptr;
    if (vkCreateComputePipelines(device, nullptr, 1, &cpci, nullptr, &pipeline) != 0) {
        emit("vkapi: pipeline failed\n");
        return;
    }
    CmdPoolCI cpoolCI = {39, nullptr, 0, family};
    void* cmdPool = nullptr;
    vkCreateCommandPool(device, &cpoolCI, nullptr, &cmdPool);
    FenceCI fci = {8, nullptr, 0};
    void* fence = nullptr;
    vkCreateFence(device, &fci, nullptr, &fence);
    emit("vkapi: setup done\n");

    // --- A. CPU record cost: N x (bindPipeline + bindDescSets + dispatch) ---
    CmdAllocInfo cai = {40, nullptr, cmdPool, 0, 1};
    void* cmd = nullptr;
    vkAllocateCommandBuffers(device, &cai, &cmd);
    CmdBeginInfo cbi = {42, nullptr, 1, nullptr};
    for (int dispCount : {1, 180}) {
        double best = 1e9;
        for (int rep = 0; rep < 20; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            vkBeginCommandBuffer(cmd, &cbi);
            for (int i = 0; i < dispCount; ++i) {
                vkCmdBindPipeline(cmd, 1, pipeline);
                vkCmdBindDescriptorSets(cmd, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
                vkCmdDispatch(cmd, 1, 1, 1);
            }
            vkEndCommandBuffer(cmd);
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, tms(t0, t1));
        }
        snprintf(line, sizeof(line), "vkapi record %3d full-bind dispatch: %8.3f ms total, %7.1f us/cmd\n", dispCount,
                 best, best * 1000 / dispCount);
        emit(line);
    }
    // dispatch-only (no rebind) to separate bind cost from dispatch record cost
    {
        double best = 1e9;
        for (int rep = 0; rep < 20; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            vkBeginCommandBuffer(cmd, &cbi);
            vkCmdBindPipeline(cmd, 1, pipeline);
            vkCmdBindDescriptorSets(cmd, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
            for (int i = 0; i < 180; ++i) {
                vkCmdDispatch(cmd, 1, 1, 1);
            }
            vkEndCommandBuffer(cmd);
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, tms(t0, t1));
        }
        snprintf(line, sizeof(line), "vkapi record 180 dispatch-only:         %8.3f ms total, %7.1f us/cmd\n", best,
                 best * 1000 / 180);
        emit(line);
    }

    // --- B. GPU front-end cost: submit+fence for pre-recorded 1 vs 180 dispatches ---
    for (int dispCount : {1, 180}) {
        vkBeginCommandBuffer(cmd, &cbi);
        for (int i = 0; i < dispCount; ++i) {
            vkCmdBindPipeline(cmd, 1, pipeline);
            vkCmdBindDescriptorSets(cmd, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
            vkCmdDispatch(cmd, 1, 1, 1);
        }
        vkEndCommandBuffer(cmd);
        SubmitInfo si = {4, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
        double bestSubmit = 1e9, bestWait = 1e9;
        for (int rep = 0; rep < 12; ++rep) {
            vkResetFences(device, 1, &fence);
            auto t0 = std::chrono::steady_clock::now();
            vkQueueSubmit(queue, 1, &si, fence);
            auto t1 = std::chrono::steady_clock::now();
            vkWaitForFences(device, 1, &fence, 1, 0xFFFFFFFFFFFFFFFFull);
            auto t2 = std::chrono::steady_clock::now();
            bestSubmit = std::min(bestSubmit, tms(t0, t1));
            bestWait = std::min(bestWait, tms(t1, t2));
        }
        snprintf(line, sizeof(line),
                 "vkapi submit+fence %3d empty dispatch: submit %7.3f ms, gpuWait %8.3f ms (%7.1f us/dispatch)\n",
                 dispCount, bestSubmit, bestWait, bestWait * 1000 / dispCount);
        emit(line);
    }

    // --- C. pipeline-switch cost: alternate between two distinct pipelines ---
    ShaderModuleCI smci2 = {16, nullptr, 0, kEmpty2SpvLen, kEmpty2Spv};
    void* shaderModule2 = nullptr;
    if (vkCreateShaderModule(device, &smci2, nullptr, &shaderModule2) == 0) {
        StageCI stage2 = {18, nullptr, 0, 0x20, shaderModule2, "main", nullptr};
        ComputePipelineCI cpci2 = {29, nullptr, 0, stage2, pipeLayout, nullptr, -1};
        void* pipeline2 = nullptr;
        if (vkCreateComputePipelines(device, nullptr, 1, &cpci2, nullptr, &pipeline2) == 0) {
            for (int mode = 0; mode < 2; ++mode) {
                vkBeginCommandBuffer(cmd, &cbi);
                for (int i = 0; i < 180; ++i) {
                    void* p = (mode == 0 || (i & 1) == 0) ? pipeline : pipeline2;
                    vkCmdBindPipeline(cmd, 1, p);
                    vkCmdBindDescriptorSets(cmd, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
                    vkCmdDispatch(cmd, 1, 1, 1);
                }
                vkEndCommandBuffer(cmd);
                SubmitInfo si = {4, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
                double bestWait = 1e9;
                for (int rep = 0; rep < 12; ++rep) {
                    vkResetFences(device, 1, &fence);
                    vkQueueSubmit(queue, 1, &si, fence);
                    auto t1 = std::chrono::steady_clock::now();
                    vkWaitForFences(device, 1, &fence, 1, 0xFFFFFFFFFFFFFFFFull);
                    auto t2 = std::chrono::steady_clock::now();
                    bestWait = std::min(bestWait, tms(t1, t2));
                }
                snprintf(line, sizeof(line),
                         "vkapi 180 dispatch %s pipeline: gpuWait %8.3f ms (%7.1f us/dispatch)\n",
                         mode == 0 ? "alternating" : "single     ", bestWait, bestWait * 1000 / 180);
                emit(line);
            }
        }
    }
    emit("vkapi: done\n");
}

// @vkbw: read-bandwidth microbench comparing Maleoon's two memory heaps:
// heap0 (unified, DEVICE_LOCAL|HOST_VISIBLE) — what MNN currently allocates
// tensors/weights in — versus heap1 (800MB pure DEVICE_LOCAL). If heap1 is
// substantially faster, MNN's HOST_VISIBLE allocation policy is the reason
// Vulkan GEMV runs at 1-3 GB/s while OpenCL (device-side allocs) hits 13-28.
#include "bw_spv.h"

static void runVkBw(std::string& out) {
    using namespace vkprobe;
    auto emit = [&out](const std::string& s) {
        out += s;
        MNN_PRINT("%s", s.c_str());
    };
    char line[256];
    auto tms = [](const std::chrono::steady_clock::time_point& a, const std::chrono::steady_clock::time_point& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    typedef int (*CreateInstanceFn)(const void*, const void*, void**);
    typedef int (*EnumPhysFn)(void*, unsigned*, void**);
    typedef void (*GetQueueFamFn)(void*, unsigned*, void*);
    typedef void (*GetMemPropsFn)(void*, void*);
    typedef int (*CreateDeviceFn)(void*, const DeviceCI*, const void*, void**);
    typedef void (*GetDeviceQueueFn)(void*, unsigned, unsigned, void**);
    typedef int (*CreateBufferFn)(void*, const BufferCI*, const void*, void**);
    typedef void (*GetBufMemReqFn)(void*, void*, void*);
    typedef int (*AllocMemFn)(void*, const MemAllocInfo*, const void*, void**);
    typedef int (*BindBufMemFn)(void*, void*, void*, unsigned long long);
    typedef int (*CreateShaderFn)(void*, const ShaderModuleCI*, const void*, void**);
    typedef int (*CreateDsLayoutFn)(void*, const DsLayoutCI*, const void*, void**);
    typedef int (*CreatePipeLayoutFn)(void*, const PipelineLayoutCI*, const void*, void**);
    typedef int (*CreateDescPoolFn)(void*, const DescPoolCI*, const void*, void**);
    typedef int (*AllocDescSetsFn)(void*, const DescSetAllocInfo*, void**);
    typedef void (*UpdateDescSetsFn)(void*, unsigned, const WriteDescSet*, unsigned, const void*);
    typedef int (*CreatePipelinesFn)(void*, void*, unsigned, const ComputePipelineCI*, const void*, void**);
    typedef int (*CreateCmdPoolFn)(void*, const CmdPoolCI*, const void*, void**);
    typedef int (*AllocCmdBufsFn)(void*, const CmdAllocInfo*, void**);
    typedef int (*BeginCmdFn)(void*, const CmdBeginInfo*);
    typedef void (*CmdBindPipelineFn)(void*, unsigned, void*);
    typedef void (*CmdBindDescFn)(void*, unsigned, void*, unsigned, unsigned, const void*, unsigned, const unsigned*);
    typedef void (*CmdPushConstFn)(void*, void*, unsigned, unsigned, unsigned, const void*);
    typedef void (*CmdDispatchFn)(void*, unsigned, unsigned, unsigned);
    typedef int (*EndCmdFn)(void*);
    typedef int (*CreateFenceFn)(void*, const FenceCI*, const void*, void**);
    typedef int (*ResetFencesFn)(void*, unsigned, const void*);
    typedef int (*QueueSubmitFn)(void*, unsigned, const SubmitInfo*, void*);
    typedef int (*WaitFencesFn)(void*, unsigned, const void*, unsigned, unsigned long long);
    void* lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!lib) { emit("vkbw: no libvulkan\n"); return; }
#define BW_SYM(type, name) auto name = (type)dlsym(lib, #name); \
    if (!name) { emit("vkbw: missing " #name "\n"); return; }
    BW_SYM(CreateInstanceFn, vkCreateInstance)
    BW_SYM(EnumPhysFn, vkEnumeratePhysicalDevices)
    BW_SYM(GetQueueFamFn, vkGetPhysicalDeviceQueueFamilyProperties)
    BW_SYM(GetMemPropsFn, vkGetPhysicalDeviceMemoryProperties)
    BW_SYM(CreateDeviceFn, vkCreateDevice)
    BW_SYM(GetDeviceQueueFn, vkGetDeviceQueue)
    BW_SYM(CreateBufferFn, vkCreateBuffer)
    BW_SYM(GetBufMemReqFn, vkGetBufferMemoryRequirements)
    BW_SYM(AllocMemFn, vkAllocateMemory)
    BW_SYM(BindBufMemFn, vkBindBufferMemory)
    BW_SYM(CreateShaderFn, vkCreateShaderModule)
    BW_SYM(CreateDsLayoutFn, vkCreateDescriptorSetLayout)
    BW_SYM(CreatePipeLayoutFn, vkCreatePipelineLayout)
    BW_SYM(CreateDescPoolFn, vkCreateDescriptorPool)
    BW_SYM(AllocDescSetsFn, vkAllocateDescriptorSets)
    BW_SYM(UpdateDescSetsFn, vkUpdateDescriptorSets)
    BW_SYM(CreatePipelinesFn, vkCreateComputePipelines)
    BW_SYM(CreateCmdPoolFn, vkCreateCommandPool)
    BW_SYM(AllocCmdBufsFn, vkAllocateCommandBuffers)
    BW_SYM(BeginCmdFn, vkBeginCommandBuffer)
    BW_SYM(CmdBindPipelineFn, vkCmdBindPipeline)
    BW_SYM(CmdBindDescFn, vkCmdBindDescriptorSets)
    BW_SYM(CmdPushConstFn, vkCmdPushConstants)
    BW_SYM(CmdDispatchFn, vkCmdDispatch)
    BW_SYM(EndCmdFn, vkEndCommandBuffer)
    BW_SYM(CreateFenceFn, vkCreateFence)
    BW_SYM(ResetFencesFn, vkResetFences)
    BW_SYM(QueueSubmitFn, vkQueueSubmit)
    BW_SYM(WaitFencesFn, vkWaitForFences)
#undef BW_SYM

    struct InstanceCI2 { unsigned sType; const void* pNext; unsigned flags; const void* appInfo;
                         unsigned layerCount; const char* const* layers; unsigned extCount; const char* const* exts; };
    const char* instExts[] = {"VK_KHR_get_physical_device_properties2"};
    InstanceCI2 ici = {1, nullptr, 0, nullptr, 0, nullptr, 1, instExts};
    void* instance = nullptr;
    if (vkCreateInstance(&ici, nullptr, &instance) != 0) { emit("vkbw: instance failed\n"); return; }
    unsigned count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<void*> phys(count, nullptr);
    vkEnumeratePhysicalDevices(instance, &count, phys.data());
    void* gpu = phys[0];
    unsigned famCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &famCount, nullptr);
    std::vector<QueueFamilyProps> fams(famCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &famCount, fams.data());
    unsigned family = 0xffffffff;
    for (unsigned i = 0; i < famCount; ++i) {
        if ((fams[i].queueFlags & 2) && fams[i].queueCount > 0) { family = i; break; }
    }
    float qprio = 1.0f;
    DeviceQueueCI qci = {2, nullptr, 0, family, 1, &qprio};
    DeviceCI dci = {3, nullptr, 0, 1, &qci, 0, nullptr, 0, nullptr, nullptr};
    void* device = nullptr;
    if (vkCreateDevice(gpu, &dci, nullptr, &device) != 0) { emit("vkbw: device failed\n"); return; }
    void* queue = nullptr;
    vkGetDeviceQueue(device, family, 0, &queue);

    char memProps[1024] = {0};
    vkGetPhysicalDeviceMemoryProperties(gpu, memProps);
    unsigned typeCount = 0;
    memcpy(&typeCount, memProps, 4);
    int unifiedType = -1, deviceOnlyType = -1;
    for (unsigned i = 0; i < typeCount && i < 32; ++i) {
        unsigned flags = 0;
        memcpy(&flags, memProps + 4 + i * 8, 4);
        const bool devLocal = flags & 1, hostVis = flags & 2, lazy = flags & 16;
        if (devLocal && hostVis && (flags & 4) && unifiedType < 0) unifiedType = (int)i;
        if (devLocal && !hostVis && !lazy && deviceOnlyType < 0) deviceOnlyType = (int)i;
    }
    snprintf(line, sizeof(line), "vkbw: unifiedType=%d deviceOnlyType=%d\n", unifiedType, deviceOnlyType);
    emit(line);

    // 256 MB source, 16 bytes per uvec4 → 16M vectors; each of 1M threads reads 16.
    const unsigned long long kBufSize = 256ull * 1024 * 1024;
    const unsigned nVec = (unsigned)(kBufSize / 16);
    const unsigned stride = 16;
    const unsigned threads = nVec / stride; // 1M
    const unsigned wg = threads / 256;      // 4096

    ShaderModuleCI smci = {16, nullptr, 0, kBwSpvLen, kBwSpv};
    void* shaderModule = nullptr;
    if (vkCreateShaderModule(device, &smci, nullptr, &shaderModule) != 0) { emit("vkbw: shader failed\n"); return; }
    DsLayoutBinding bindings[2] = {{0, 6, 1, 0x20, nullptr}, {1, 6, 1, 0x20, nullptr}};
    DsLayoutCI dslci = {32, nullptr, 0, 2, bindings};
    void* dsLayout = nullptr;
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsLayout);
    struct PushRange { unsigned stage; unsigned offset; unsigned size; } prange = {0x20, 0, 8};
    PipelineLayoutCI plci = {30, nullptr, 0, 1, &dsLayout, 1, &prange};
    void* pipeLayout = nullptr;
    vkCreatePipelineLayout(device, &plci, nullptr, &pipeLayout);
    StageCI stage = {18, nullptr, 0, 0x20, shaderModule, "main", nullptr};
    ComputePipelineCI cpci = {29, nullptr, 0, stage, pipeLayout, nullptr, -1};
    void* pipeline = nullptr;
    if (vkCreateComputePipelines(device, nullptr, 1, &cpci, nullptr, &pipeline) != 0) {
        emit("vkbw: pipeline failed\n");
        return;
    }
    CmdPoolCI cpoolCI = {39, nullptr, 0, family};
    void* cmdPool = nullptr;
    vkCreateCommandPool(device, &cpoolCI, nullptr, &cmdPool);
    CmdAllocInfo cai = {40, nullptr, cmdPool, 0, 1};
    void* cmd = nullptr;
    vkAllocateCommandBuffers(device, &cai, &cmd);
    CmdBeginInfo cbi = {42, nullptr, 1, nullptr};
    FenceCI fci = {8, nullptr, 0};
    void* fence = nullptr;
    vkCreateFence(device, &fci, nullptr, &fence);
    DescPoolSize poolSize = {6, 8};
    DescPoolCI dpci = {33, nullptr, 0, 4, 1, &poolSize};
    void* descPool = nullptr;
    vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);

    for (int pass = 0; pass < 2; ++pass) {
        const int memType = (pass == 0) ? unifiedType : deviceOnlyType;
        if (memType < 0) { emit("vkbw: skip pass (type missing)\n"); continue; }
        BufferCI srcCI = {12, nullptr, 0, kBufSize, 0x20, 0, 0, nullptr};
        void* srcBuf = nullptr;
        if (vkCreateBuffer(device, &srcCI, nullptr, &srcBuf) != 0) { emit("vkbw: src buffer failed\n"); continue; }
        MemReqs srcReq = {0, 0, 0};
        vkGetBufferMemoryRequirements(device, srcBuf, &srcReq);
        if ((srcReq.memoryTypeBits & (1u << memType)) == 0) {
            emit("vkbw: src type unsupported by buffer\n");
            continue;
        }
        MemAllocInfo srcMai = {5, nullptr, srcReq.size, (unsigned)memType};
        void* srcMem = nullptr;
        if (vkAllocateMemory(device, &srcMai, nullptr, &srcMem) != 0) { emit("vkbw: src alloc failed\n"); continue; }
        vkBindBufferMemory(device, srcBuf, srcMem, 0);
        BufferCI outCI = {12, nullptr, 0, 256, 0x20, 0, 0, nullptr};
        void* outBuf = nullptr;
        vkCreateBuffer(device, &outCI, nullptr, &outBuf);
        MemReqs outReq = {0, 0, 0};
        vkGetBufferMemoryRequirements(device, outBuf, &outReq);
        MemAllocInfo outMai = {5, nullptr, outReq.size, (unsigned)memType};
        void* outMem = nullptr;
        if ((outReq.memoryTypeBits & (1u << memType)) == 0 || vkAllocateMemory(device, &outMai, nullptr, &outMem) != 0) {
            // small out buffer: fall back to any host type
            outMai.memoryTypeIndex = (unsigned)unifiedType;
            if (vkAllocateMemory(device, &outMai, nullptr, &outMem) != 0) { emit("vkbw: out alloc failed\n"); continue; }
        }
        vkBindBufferMemory(device, outBuf, outMem, 0);
        DescSetAllocInfo dsai = {34, nullptr, descPool, 1, &dsLayout};
        void* descSet = nullptr;
        vkAllocateDescriptorSets(device, &dsai, &descSet);
        DescBufferInfo dbi0 = {srcBuf, 0, kBufSize};
        DescBufferInfo dbi1 = {outBuf, 0, 256};
        WriteDescSet wds[2] = {{35, nullptr, descSet, 0, 0, 1, 6, nullptr, &dbi0, nullptr},
                               {35, nullptr, descSet, 1, 0, 1, 6, nullptr, &dbi1, nullptr}};
        vkUpdateDescriptorSets(device, 2, wds, 0, nullptr);

        vkBeginCommandBuffer(cmd, &cbi);
        vkCmdBindPipeline(cmd, 1, pipeline);
        vkCmdBindDescriptorSets(cmd, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
        unsigned pcData[2] = {nVec, stride};
        vkCmdPushConstants(cmd, pipeLayout, 0x20, 0, 8, pcData);
        vkCmdDispatch(cmd, wg, 1, 1);
        vkEndCommandBuffer(cmd);
        SubmitInfo si = {4, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
        // warmup twice (also commits device-local pages), then best of 5
        double best = 1e9;
        for (int rep = 0; rep < 7; ++rep) {
            vkResetFences(device, 1, &fence);
            vkQueueSubmit(queue, 1, &si, fence);
            auto t0 = std::chrono::steady_clock::now();
            vkWaitForFences(device, 1, &fence, 1, 0xFFFFFFFFFFFFFFFFull);
            auto t1 = std::chrono::steady_clock::now();
            if (rep >= 2) best = std::min(best, tms(t0, t1));
        }
        const double gbps = static_cast<double>(kBufSize) / (best / 1000.0) / 1e9;
        snprintf(line, sizeof(line), "vkbw memtype %d (%s): 256MB read in %.3f ms => %.1f GB/s\n", memType,
                 pass == 0 ? "heap0 unified, MNN default" : "heap1 device-only", best, gbps);
        emit(line);
    }
    emit("vkbw: done\n");
}

// @vkgemv: isolate the GEMV decode access pattern. Sequential big-block reads
// hit 156 GB/s (@vkbw) while MNN's GEMV runs at 1-3 GB/s; this probe sweeps
// the pattern dimensions (rows per workgroup x consecutive uvec4 reads per
// lane) on a fixed row count to find which decomposition Maleoon likes.
#include "gemvpat_spv.h"

static void runVkGemvPattern(std::string& out, bool hostCached = false) {
    using namespace vkprobe;
    auto emit = [&out](const std::string& s) {
        out += s;
        MNN_PRINT("%s", s.c_str());
    };
    char line[256];
    auto tms = [](const std::chrono::steady_clock::time_point& a, const std::chrono::steady_clock::time_point& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    typedef int (*CreateInstanceFn)(const void*, const void*, void**);
    typedef int (*EnumPhysFn)(void*, unsigned*, void**);
    typedef void (*GetQueueFamFn)(void*, unsigned*, void*);
    typedef void (*GetMemPropsFn)(void*, void*);
    typedef int (*CreateDeviceFn)(void*, const DeviceCI*, const void*, void**);
    typedef void (*GetDeviceQueueFn)(void*, unsigned, unsigned, void**);
    typedef int (*CreateBufferFn)(void*, const BufferCI*, const void*, void**);
    typedef void (*GetBufMemReqFn)(void*, void*, void*);
    typedef int (*AllocMemFn)(void*, const MemAllocInfo*, const void*, void**);
    typedef int (*BindBufMemFn)(void*, void*, void*, unsigned long long);
    typedef int (*CreateShaderFn)(void*, const ShaderModuleCI*, const void*, void**);
    typedef int (*CreateDsLayoutFn)(void*, const DsLayoutCI*, const void*, void**);
    typedef int (*CreatePipeLayoutFn)(void*, const PipelineLayoutCI*, const void*, void**);
    typedef int (*CreateDescPoolFn)(void*, const DescPoolCI*, const void*, void**);
    typedef int (*AllocDescSetsFn)(void*, const DescSetAllocInfo*, void**);
    typedef void (*UpdateDescSetsFn)(void*, unsigned, const WriteDescSet*, unsigned, const void*);
    typedef int (*CreatePipelinesFn)(void*, void*, unsigned, const ComputePipelineCI*, const void*, void**);
    typedef int (*CreateCmdPoolFn)(void*, const CmdPoolCI*, const void*, void**);
    typedef int (*AllocCmdBufsFn)(void*, const CmdAllocInfo*, void**);
    typedef int (*BeginCmdFn)(void*, const CmdBeginInfo*);
    typedef void (*CmdBindPipelineFn)(void*, unsigned, void*);
    typedef void (*CmdBindDescFn)(void*, unsigned, void*, unsigned, unsigned, const void*, unsigned, const unsigned*);
    typedef void (*CmdPushConstFn)(void*, void*, unsigned, unsigned, unsigned, const void*);
    typedef void (*CmdDispatchFn)(void*, unsigned, unsigned, unsigned);
    typedef int (*EndCmdFn)(void*);
    typedef int (*CreateFenceFn)(void*, const FenceCI*, const void*, void**);
    typedef int (*ResetFencesFn)(void*, unsigned, const void*);
    typedef int (*QueueSubmitFn)(void*, unsigned, const SubmitInfo*, void*);
    typedef int (*WaitFencesFn)(void*, unsigned, const void*, unsigned, unsigned long long);
    typedef int (*CreateBufferViewFn)(void*, const void*, const void*, void**);
    void* lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!lib) { emit("vkgemv: no libvulkan\n"); return; }
#define GP_SYM(type, name) auto name = (type)dlsym(lib, #name); \
    if (!name) { emit("vkgemv: missing " #name "\n"); return; }
    GP_SYM(CreateInstanceFn, vkCreateInstance)
    GP_SYM(EnumPhysFn, vkEnumeratePhysicalDevices)
    GP_SYM(GetQueueFamFn, vkGetPhysicalDeviceQueueFamilyProperties)
    GP_SYM(GetMemPropsFn, vkGetPhysicalDeviceMemoryProperties)
    GP_SYM(CreateDeviceFn, vkCreateDevice)
    GP_SYM(GetDeviceQueueFn, vkGetDeviceQueue)
    GP_SYM(CreateBufferFn, vkCreateBuffer)
    GP_SYM(GetBufMemReqFn, vkGetBufferMemoryRequirements)
    GP_SYM(AllocMemFn, vkAllocateMemory)
    GP_SYM(BindBufMemFn, vkBindBufferMemory)
    GP_SYM(CreateShaderFn, vkCreateShaderModule)
    GP_SYM(CreateDsLayoutFn, vkCreateDescriptorSetLayout)
    GP_SYM(CreatePipeLayoutFn, vkCreatePipelineLayout)
    GP_SYM(CreateDescPoolFn, vkCreateDescriptorPool)
    GP_SYM(AllocDescSetsFn, vkAllocateDescriptorSets)
    GP_SYM(UpdateDescSetsFn, vkUpdateDescriptorSets)
    GP_SYM(CreatePipelinesFn, vkCreateComputePipelines)
    GP_SYM(CreateCmdPoolFn, vkCreateCommandPool)
    GP_SYM(AllocCmdBufsFn, vkAllocateCommandBuffers)
    GP_SYM(BeginCmdFn, vkBeginCommandBuffer)
    GP_SYM(CmdBindPipelineFn, vkCmdBindPipeline)
    GP_SYM(CmdBindDescFn, vkCmdBindDescriptorSets)
    GP_SYM(CmdPushConstFn, vkCmdPushConstants)
    GP_SYM(CmdDispatchFn, vkCmdDispatch)
    GP_SYM(EndCmdFn, vkEndCommandBuffer)
    GP_SYM(CreateFenceFn, vkCreateFence)
    GP_SYM(ResetFencesFn, vkResetFences)
    GP_SYM(QueueSubmitFn, vkQueueSubmit)
    GP_SYM(WaitFencesFn, vkWaitForFences)
    GP_SYM(CreateBufferViewFn, vkCreateBufferView)
#undef GP_SYM

    struct InstanceCI3 { unsigned sType; const void* pNext; unsigned flags; const void* appInfo;
                         unsigned layerCount; const char* const* layers; unsigned extCount; const char* const* exts; };
    const char* instExts[] = {"VK_KHR_get_physical_device_properties2"};
    InstanceCI3 ici = {1, nullptr, 0, nullptr, 0, nullptr, 1, instExts};
    void* instance = nullptr;
    if (vkCreateInstance(&ici, nullptr, &instance) != 0) { emit("vkgemv: instance failed\n"); return; }
    unsigned count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<void*> phys(count, nullptr);
    vkEnumeratePhysicalDevices(instance, &count, phys.data());
    void* gpu = phys[0];
    unsigned famCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &famCount, nullptr);
    std::vector<QueueFamilyProps> fams(famCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &famCount, fams.data());
    unsigned family = 0xffffffff;
    for (unsigned i = 0; i < famCount; ++i) {
        if ((fams[i].queueFlags & 2) && fams[i].queueCount > 0) { family = i; break; }
    }
    float qprio = 1.0f;
    DeviceQueueCI qci = {2, nullptr, 0, family, 1, &qprio};
    DeviceCI dci = {3, nullptr, 0, 1, &qci, 0, nullptr, 0, nullptr, nullptr};
    void* device = nullptr;
    if (vkCreateDevice(gpu, &dci, nullptr, &device) != 0) { emit("vkgemv: device failed\n"); return; }
    void* queue = nullptr;
    vkGetDeviceQueue(device, family, 0, &queue);
    char memProps[1024] = {0};
    vkGetPhysicalDeviceMemoryProperties(gpu, memProps);
    unsigned typeCount = 0;
    memcpy(&typeCount, memProps, 4);
    unsigned hostType = 0xffffffff;
    for (unsigned i = 0; i < typeCount && i < 32; ++i) {
        unsigned flags = 0;
        memcpy(&flags, memProps + 4 + i * 8, 4);
        // HOST_VISIBLE=0x2, HOST_CACHED=0x8, HOST_COHERENT=0x4
        const bool ok = hostCached ? ((flags & 0xA) == 0xA) : ((flags & 3) == 3);
        if (ok) { hostType = i; break; }
    }
    snprintf(line, sizeof(line), "vkgemv: memory type %u (%s)\n", hostType, hostCached ? "HOST_CACHED" : "HOST_COHERENT");
    emit(line);

    const unsigned long long kBufSize = 16ull * 1024 * 1024;
    BufferCI srcCI = {12, nullptr, 0, kBufSize, 0x20, 0, 0, nullptr};
    void* srcBuf = nullptr;
    vkCreateBuffer(device, &srcCI, nullptr, &srcBuf);
    MemReqs srcReq = {0, 0, 0};
    vkGetBufferMemoryRequirements(device, srcBuf, &srcReq);
    MemAllocInfo srcMai = {5, nullptr, srcReq.size, hostType};
    void* srcMem = nullptr;
    if (vkAllocateMemory(device, &srcMai, nullptr, &srcMem) != 0) { emit("vkgemv: alloc failed\n"); return; }
    vkBindBufferMemory(device, srcBuf, srcMem, 0);
    BufferCI outCI = {12, nullptr, 0, 256, 0x20, 0, 0, nullptr};
    void* outBuf = nullptr;
    vkCreateBuffer(device, &outCI, nullptr, &outBuf);
    MemReqs outReq = {0, 0, 0};
    vkGetBufferMemoryRequirements(device, outBuf, &outReq);
    MemAllocInfo outMai = {5, nullptr, outReq.size, hostType};
    void* outMem = nullptr;
    vkAllocateMemory(device, &outMai, nullptr, &outMem);
    vkBindBufferMemory(device, outBuf, outMem, 0);

    // RGBA32_UINT texel view over the weight buffer: same bytes, texture path.
    struct BufferViewCI { unsigned sType; const void* pNext; unsigned flags; void* buffer; unsigned format;
                          unsigned long long offset; unsigned long long range; };
    BufferViewCI bvci = {13, nullptr, 0, srcBuf, 109 /*R32G32B32A32_UINT*/, 0, kBufSize};
    void* srcView = nullptr;
    if (vkCreateBufferView(device, &bvci, nullptr, &srcView) != 0) { emit("vkgemv: buffer view failed\n"); return; }

    ShaderModuleCI smci = {16, nullptr, 0, kGemvPatSpvLen, kGemvPatSpv};
    void* shaderModule = nullptr;
    if (vkCreateShaderModule(device, &smci, nullptr, &shaderModule) != 0) { emit("vkgemv: shader failed\n"); return; }
    DsLayoutBinding bindings[3] = {{0, 6, 1, 0x20, nullptr}, {1, 6, 1, 0x20, nullptr}, {2, 1, 1, 0x20, nullptr}};
    DsLayoutCI dslci = {32, nullptr, 0, 3, bindings};
    void* dsLayout = nullptr;
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsLayout);
    struct PushRange { unsigned stage; unsigned offset; unsigned size; } prange = {0x20, 0, 12};
    PipelineLayoutCI plci = {30, nullptr, 0, 1, &dsLayout, 1, &prange};
    void* pipeLayout = nullptr;
    vkCreatePipelineLayout(device, &plci, nullptr, &pipeLayout);
    DescPoolSize poolSizes[2] = {{6, 16}, {1, 16}};
    DescPoolCI dpci = {33, nullptr, 0, 8, 2, poolSizes};
    void* descPool = nullptr;
    vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);
    DescSetAllocInfo dsai = {34, nullptr, descPool, 1, &dsLayout};
    void* descSet = nullptr;
    vkAllocateDescriptorSets(device, &dsai, &descSet);
    DescBufferInfo dbi0 = {srcBuf, 0, kBufSize};
    DescBufferInfo dbi1 = {outBuf, 0, 256};
    WriteDescSet wds[3] = {{35, nullptr, descSet, 0, 0, 1, 6, nullptr, &dbi0, nullptr},
                           {35, nullptr, descSet, 1, 0, 1, 6, nullptr, &dbi1, nullptr},
                           {35, nullptr, descSet, 2, 0, 1, 1, nullptr, nullptr, &srcView}};
    vkUpdateDescriptorSets(device, 3, wds, 0, nullptr);

    CmdPoolCI cpoolCI = {39, nullptr, 0, family};
    void* cmdPool = nullptr;
    vkCreateCommandPool(device, &cpoolCI, nullptr, &cmdPool);
    CmdAllocInfo cai = {40, nullptr, cmdPool, 0, 1};
    void* cmd = nullptr;
    vkAllocateCommandBuffers(device, &cai, &cmd);
    CmdBeginInfo cbi = {42, nullptr, 1, nullptr};
    FenceCI fci = {8, nullptr, 0};
    void* fence = nullptr;
    vkCreateFence(device, &fci, nullptr, &fence);
    emit("vkgemv: setup done\n");

    struct SpecEntry { unsigned constantID; unsigned offset; unsigned long long size; };
    struct SpecInfo { unsigned count; const SpecEntry* entries; unsigned long long dataSize; const void* data; };
    struct Variant {
        const char* name;
        unsigned localX;   // lanes per row (localX * reads must equal vec4PerRow)
        unsigned localY;   // rows per workgroup (y-dim mapping mode)
        unsigned reads;    // consecutive uvec4 loads per lane per row
        unsigned rowLoop;  // 0: y-dim row mapping; >0: serial row loop per wg
        unsigned compute;  // 0: pure read; 1: +I2F nibble unpack+FMA; 2: +bit-trick unpack
        unsigned rows;     // total rows (workgroups when rowLoop==0)
        unsigned texel;    // 1: read via RGBA32_UINT texel buffer (texture path)
    };
    // Decomposition sweep at the REAL per-GEMV data sizes (FFN gate/up = 3.15MB):
    // oc1=6144wg x 512B, oc2=3072wg x 1KB, oc4=1536wg x 2KB, oc8=768wg x 4KB,
    // each SSBO vs texel-buffer, to find a >=20GB/s path toward OpenCL parity.
    // Large-tile sweep: prefill's GEMM reads 64KB weight tiles per workgroup
    // at ~200GB/s effective on the SAME driver, while every GEMV-shaped probe
    // (<=4KB/wg) caps at ~10GB/s. Test the untried "few wg x huge block" region
    // (8MB total each) to see if big tiles unlock DRAM-row/prefetch efficiency.
    // 8KB/wg sweet spot found (L5 1024wg x 8KB = 23.8GB/s). Now verify at the
    // REAL per-GEMV data sizes with oc16 tiles (16oc x K1024 x 0.5B = 8KB/wg):
    // FFN gate_up 3.15MB=384wg, qkv ~2MB=256wg, o ~1MB=128wg, down 1.5MB=192wg.
    // PURE-READ WGS sweep, configs identical to the OpenCL @ocgemv probe so the
    // two APIs can be compared with no compute confound. localX=WGS, reads=
    // vec4PerRow/WGS, global={WGS,rows}.
    const Variant variants[] = {
        {"oc1 6144wg x 512B W32 (3.15MB)", 32, 1, 1, 0, 0, 6144, 0},
        {"oc8 768wg x 4KB W32 (3.15MB)", 32, 1, 8, 0, 0, 768, 0},
        {"oc8 768wg x 4KB W128 (3.15MB)", 128, 1, 2, 0, 0, 768, 0},
        {"oc8 768wg x 4KB W256 (3.15MB)", 256, 1, 1, 0, 0, 768, 0},
        {"oc8 1536wg x 4KB W256 (6.3MB)", 256, 1, 1, 0, 0, 1536, 0},
        {"oc8 3072wg x 4KB W256 (12.6MB)", 256, 1, 1, 0, 0, 3072, 0},
    };
    for (const Variant& v : variants) {
        const unsigned nRows = v.rows;
        const unsigned vec4PerRow = v.localX * v.reads;
        const unsigned long long bytes = (unsigned long long)nRows * vec4PerRow * 16;
        if (bytes > kBufSize) { emit("vkgemv: variant too big, skip\n"); continue; }
        const unsigned rowsPerWG = v.rowLoop > 0 ? v.rowLoop : v.localY;
        unsigned specData[7] = {v.localX, v.localY, 1, v.reads, v.rowLoop, v.compute, v.texel};
        SpecEntry entries[7] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}, {5, 20, 4}, {6, 24, 4}};
        SpecInfo specInfo = {7, entries, 28, specData};
        StageCI stage = {18, nullptr, 0, 0x20, shaderModule, "main", &specInfo};
        ComputePipelineCI cpci = {29, nullptr, 0, stage, pipeLayout, nullptr, -1};
        void* pipeline = nullptr;
        if (vkCreateComputePipelines(device, nullptr, 1, &cpci, nullptr, &pipeline) != 0) {
            emit("vkgemv: pipeline failed\n");
            continue;
        }
        const unsigned gridX = (nRows + rowsPerWG - 1) / rowsPerWG;
        unsigned pcData[3] = {vec4PerRow, nRows, rowsPerWG};
        vkBeginCommandBuffer(cmd, &cbi);
        vkCmdBindPipeline(cmd, 1, pipeline);
        vkCmdBindDescriptorSets(cmd, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, pipeLayout, 0x20, 0, 12, pcData);
        vkCmdDispatch(cmd, gridX, 1, 1);
        vkEndCommandBuffer(cmd);
        SubmitInfo si = {4, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
        double best = 1e9;
        for (int rep = 0; rep < 10; ++rep) {
            vkResetFences(device, 1, &fence);
            vkQueueSubmit(queue, 1, &si, fence);
            auto t0 = std::chrono::steady_clock::now();
            vkWaitForFences(device, 1, &fence, 1, 0xFFFFFFFFFFFFFFFFull);
            auto t1 = std::chrono::steady_clock::now();
            if (rep >= 3) best = std::min(best, tms(t0, t1));
        }
        snprintf(line, sizeof(line), "vkgemv %s: %8.3f ms/dispatch, %6.2f MB => %6.1f GB/s (gridX=%u)\n", v.name, best,
                 bytes / 1e6, bytes / (best / 1000.0) / 1e9, gridX);
        emit(line);
    }
    emit("vkgemv: done\n");
}

// @ocgemv: OpenCL PURE-READ twin of @vkgemv. Same access pattern, same data
// sizes, same dispatch (32 lanes x N rows, readsPerLane uvec4 per lane), NO
// compute — isolates the driver/memory-path question from the FMA confound.
// Tests both CL_MEM_READ_WRITE (device) and CL_MEM_ALLOC_HOST_PTR (host-cached).
static void runOcGemvPattern(std::string& out) {
    auto emit = [&out](const std::string& s) { out += s; MNN_PRINT("%s", s.c_str()); };
    char line[512];
    typedef int (*GetPlatformIDsFn)(unsigned, void**, unsigned*);
    typedef int (*GetDeviceIDsFn)(void*, unsigned long long, unsigned, void**, unsigned*);
    typedef void* (*CreateContextFn)(const void*, unsigned, const void*, void*, void*, int*);
    typedef void* (*CreateQueueFn)(void*, void*, unsigned long long, int*);
    typedef void* (*CreateProgramFn)(void*, unsigned, const char**, const size_t*, int*);
    typedef int (*BuildProgramFn)(void*, unsigned, const void*, const char*, void*, void*);
    typedef void* (*CreateKernelFn)(void*, const char*, int*);
    typedef void* (*CreateBufferFn)(void*, unsigned long long, size_t, void*, int*);
    typedef int (*SetArgFn)(void*, unsigned, size_t, const void*);
    typedef int (*EnqueueFn)(void*, void*, unsigned, const size_t*, const size_t*, const size_t*, unsigned, const void*, void*);
    typedef int (*FinishFn)(void*);
    typedef int (*GetEventProfilingFn)(void*, unsigned, size_t, void*, size_t*);
    const char* libs[] = {"/vendor/lib64/chipsetsdk/libhvgr_v200.so", "/vendor/lib64/chipsetsdk/libGLES_mali.so",
                          "/system/lib64/libGLES_mali.so", "libGLES_mali.so", "libOpenCL.so"};
    void* lib = nullptr;
    for (const char* name : libs) { lib = dlopen(name, RTLD_NOW); if (lib) break; }
    if (!lib) { emit("ocgemv: no OpenCL lib\n"); return; }
    auto getPlatformIDs = (GetPlatformIDsFn)dlsym(lib, "clGetPlatformIDs");
    auto getDeviceIDs = (GetDeviceIDsFn)dlsym(lib, "clGetDeviceIDs");
    auto createContext = (CreateContextFn)dlsym(lib, "clCreateContext");
    auto createQueue = (CreateQueueFn)dlsym(lib, "clCreateCommandQueue");
    auto createProgram = (CreateProgramFn)dlsym(lib, "clCreateProgramWithSource");
    auto buildProgram = (BuildProgramFn)dlsym(lib, "clBuildProgram");
    auto createKernel = (CreateKernelFn)dlsym(lib, "clCreateKernel");
    auto createBuffer = (CreateBufferFn)dlsym(lib, "clCreateBuffer");
    auto setArg = (SetArgFn)dlsym(lib, "clSetKernelArg");
    auto enqueue = (EnqueueFn)dlsym(lib, "clEnqueueNDRangeKernel");
    auto finish = (FinishFn)dlsym(lib, "clFinish");
    auto getEventProfiling = (GetEventProfilingFn)dlsym(lib, "clGetEventProfilingInfo");
    void* platform = nullptr; unsigned count = 0; int err = 0;
    getPlatformIDs(1, &platform, &count);
    void* device = nullptr;
    if (getDeviceIDs(platform, 4, 1, &device, &count) != 0) { emit("ocgemv: no GPU\n"); return; }
    void* context = createContext(nullptr, 1, &device, nullptr, nullptr, &err);
    void* queue = createQueue(context, device, 2 /*PROFILING*/, &err);
    const char* src =
        "__kernel void gemvpat(__global const uint4* w, __global uint* o, uint vec4PerRow, uint nRows, uint readsPerLane) {\n"
        "    int lane = get_local_id(0);\n"
        "    int row = get_global_id(1);\n"
        "    if (row >= (int)nRows) return;\n"
        "    uint acc = 0u;\n"
        "    int base = row * vec4PerRow + lane * readsPerLane;\n"
        "    for (uint i = 0u; i < readsPerLane; ++i) { uint4 v = w[base + i]; acc += v.x + v.y + v.z + v.w; }\n"
        "    if (acc == 0xDEADBEEFu) { o[row & 63u] = acc; }\n"
        "}\n";
    size_t srcLen = strlen(src);
    void* program = createProgram(context, 1, &src, &srcLen, &err);
    if (buildProgram(program, 1, &device, nullptr, nullptr, nullptr) != 0) { emit("ocgemv: build failed\n"); return; }
    void* kernel = createKernel(program, "gemvpat", &err);
    const size_t kBufSize = 16ull * 1024 * 1024;
    void* outBuf = createBuffer(context, 1, 256, nullptr, &err);

    struct Variant { const char* name; unsigned rows; unsigned vec4PerRow; unsigned wgs; };
    // reads = vec4PerRow / wgs; global={wgs,rows}, local={wgs,1}. Sweep WGS to
    // match the real OpenCL gemv_conv_c8_buf (WGS=128/256), not just 32 lanes.
    const Variant variants[] = {
        {"oc1 6144wg x 512B W32 (3.15MB)", 6144, 32, 32},
        {"oc8 768wg x 4KB W32 (3.15MB)", 768, 256, 32},
        {"oc8 768wg x 4KB W128 (3.15MB)", 768, 256, 128},
        {"oc8 768wg x 4KB W256 (3.15MB)", 768, 256, 256},
        {"oc8 1536wg x 4KB W256 (6.3MB)", 1536, 256, 256},
        {"oc8 3072wg x 4KB W256 (12.6MB)", 3072, 256, 256},
    };
    struct MemType { const char* name; unsigned long long flags; };
    const MemType memtypes[] = {{"READ_WRITE(device)", 1}, {"ALLOC_HOST_PTR", 1 | 16}};
    for (const MemType& mt : memtypes) {
        void* wbuf = createBuffer(context, mt.flags, kBufSize, nullptr, &err);
        if (err != 0 || !wbuf) { snprintf(line, sizeof(line), "ocgemv %s: buffer failed %d\n", mt.name, err); emit(line); continue; }
        snprintf(line, sizeof(line), "ocgemv [%s]:\n", mt.name); emit(line);
        for (const Variant& v : variants) {
            const unsigned vec4PerRow = v.vec4PerRow;
            const unsigned reads = vec4PerRow / v.wgs;
            const unsigned long long bytes = (unsigned long long)v.rows * vec4PerRow * 16;
            if (bytes > kBufSize) continue;
            setArg(kernel, 0, sizeof(void*), &wbuf);
            setArg(kernel, 1, sizeof(void*), &outBuf);
            setArg(kernel, 2, sizeof(unsigned), &vec4PerRow);
            setArg(kernel, 3, sizeof(unsigned), &v.rows);
            setArg(kernel, 4, sizeof(unsigned), &reads);
            const size_t global[2] = {v.wgs, v.rows};
            const size_t local[2] = {v.wgs, 1};
            double best = 1e9; int lastEnq = 0, lastFin = 0;
            for (int rep = 0; rep < 10; ++rep) {
                void* event = nullptr;
                const auto t0 = std::chrono::steady_clock::now();
                lastEnq = enqueue(queue, kernel, 2, nullptr, global, local, 0, nullptr, &event);
                lastFin = finish(queue);
                const auto t1 = std::chrono::steady_clock::now();
                double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
                unsigned long long s = 0, e = 0; int ps = -1, pe = -1;
                if (event && getEventProfiling) {
                    ps = getEventProfiling(event, 0x1282, sizeof(s), &s, nullptr);
                    pe = getEventProfiling(event, 0x1283, sizeof(e), &e, nullptr);
                }
                double ms = (e > s) ? (e - s) / 1e6 : wall;
                if (rep >= 3 && ms > 0 && ms < best) best = ms;
                if (rep == 9) {
                    snprintf(line, sizeof(line), "    diag %s: event=%p ps=%d pe=%d s=%llu e=%llu wall=%.3f\n",
                             v.name, event, ps, pe, s, e, wall);
                    emit(line);
                }
            }
            snprintf(line, sizeof(line), "  %-30s %7.3f ms => %6.1f GB/s (enq=%d fin=%d)\n", v.name, best,
                     bytes / (best / 1000.0) / 1e9, lastEnq, lastFin);
            emit(line);
        }
    }
    emit("ocgemv: done\n");
}

// @ocapi: same isolation for OpenCL — empty kernel, measure enqueue CPU cost
// and per-kernel GPU front-end cost (1 vs 180 kernels + clFinish).
static void runOcApiBench(std::string& out) {
    typedef int (*GetPlatformIDsFn)(unsigned, void**, unsigned*);
    typedef int (*GetDeviceIDsFn)(void*, unsigned long long, unsigned, void**, unsigned*);
    typedef void* (*CreateContextFn)(const void*, unsigned, const void*, void*, void*, int*);
    typedef void* (*CreateQueueFn)(void*, void*, unsigned long long, int*);
    typedef void* (*CreateProgramFn)(void*, unsigned, const char**, const size_t*, int*);
    typedef int (*BuildProgramFn)(void*, unsigned, const void*, const char*, void*, void*);
    typedef void* (*CreateKernelFn)(void*, const char*, int*);
    typedef void* (*CreateBufferFn)(void*, unsigned long long, size_t, void*, int*);
    typedef int (*SetArgFn)(void*, unsigned, size_t, const void*);
    typedef int (*EnqueueFn)(void*, void*, unsigned, const size_t*, const size_t*, const size_t*, unsigned, const void*,
                             void*);
    typedef int (*FinishFn)(void*);
    auto emit = [&out](const std::string& s) {
        out += s;
        MNN_PRINT("%s", s.c_str());
    };
    char line[256];
    auto tms = [](const std::chrono::steady_clock::time_point& a, const std::chrono::steady_clock::time_point& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const char* libs[] = {"/vendor/lib64/chipsetsdk/libhvgr_v200.so",
                          "/vendor/lib64/chipsetsdk/libGLES_mali.so",
                          "/vendor/lib64/chipsetsdk/libEGL_impl.so",
                          "libOpenCL.so"};
    void* lib = nullptr;
    for (const char* name : libs) {
        lib = dlopen(name, RTLD_NOW);
        if (lib) {
            snprintf(line, sizeof(line), "ocapi: lib %s\n", name);
            emit(line);
            break;
        }
    }
    if (!lib) { emit("ocapi: no OpenCL library\n"); return; }
    auto getPlatformIDs = (GetPlatformIDsFn)dlsym(lib, "clGetPlatformIDs");
    auto getDeviceIDs = (GetDeviceIDsFn)dlsym(lib, "clGetDeviceIDs");
    auto createContext = (CreateContextFn)dlsym(lib, "clCreateContext");
    auto createQueue = (CreateQueueFn)dlsym(lib, "clCreateCommandQueue");
    auto createProgram = (CreateProgramFn)dlsym(lib, "clCreateProgramWithSource");
    auto buildProgram = (BuildProgramFn)dlsym(lib, "clBuildProgram");
    auto createKernel = (CreateKernelFn)dlsym(lib, "clCreateKernel");
    auto createBuffer = (CreateBufferFn)dlsym(lib, "clCreateBuffer");
    auto setArg = (SetArgFn)dlsym(lib, "clSetKernelArg");
    auto enqueue = (EnqueueFn)dlsym(lib, "clEnqueueNDRangeKernel");
    auto finish = (FinishFn)dlsym(lib, "clFinish");
    if (!getPlatformIDs || !getDeviceIDs || !createContext || !createQueue || !createProgram || !buildProgram ||
        !createKernel || !createBuffer || !setArg || !enqueue || !finish) {
        emit("ocapi: missing symbols\n");
        return;
    }
    void* platform = nullptr;
    unsigned count = 0;
    getPlatformIDs(1, &platform, &count);
    void* device = nullptr;
    if (getDeviceIDs(platform, 4, 1, &device, &count) != 0) { emit("ocapi: no gpu\n"); return; }
    int err = 0;
    void* context = createContext(nullptr, 1, &device, nullptr, nullptr, &err);
    void* queue = createQueue(context, device, 0, &err);
    const char* src = "__kernel void nop(__global float* o) { }";
    size_t srcLen = strlen(src);
    auto t0 = std::chrono::steady_clock::now();
    void* program = createProgram(context, 1, &src, &srcLen, &err);
    int buildErr = buildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    snprintf(line, sizeof(line), "ocapi buildProgram(empty): %d, %.1f ms\n", buildErr, tms(t0, t1));
    emit(line);
    if (buildErr != 0) { return; }
    void* kernel = createKernel(program, "nop", &err);
    void* buffer = createBuffer(context, 1, 256, nullptr, &err);
    setArg(kernel, 0, sizeof(void*), &buffer);
    const size_t global = 64, local = 64;
    finish(queue);
    emit("ocapi: setup done\n");

    // A. pure CPU enqueue cost (no finish): 180 enqueues
    for (int rep = 0; rep < 3; ++rep) {
        auto ta = std::chrono::steady_clock::now();
        for (int i = 0; i < 180; ++i) {
            enqueue(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        }
        auto tb = std::chrono::steady_clock::now();
        finish(queue);
        auto tc = std::chrono::steady_clock::now();
        snprintf(line, sizeof(line), "ocapi enqueue180: cpu %8.3f ms (%5.1f us/enqueue), finish %8.3f ms (%5.1f us/kernel gpu)\n",
                 tms(ta, tb), tms(ta, tb) * 1000 / 180, tms(tb, tc), tms(tb, tc) * 1000 / 180);
        emit(line);
    }
    // B. submit+finish for 1 vs 180 kernels, best-of-12
    for (int kn : {1, 180}) {
        double bestEnq = 1e9, bestFin = 1e9;
        for (int rep = 0; rep < 12; ++rep) {
            finish(queue);
            auto ta = std::chrono::steady_clock::now();
            for (int i = 0; i < kn; ++i) {
                enqueue(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            }
            auto tb = std::chrono::steady_clock::now();
            finish(queue);
            auto tc = std::chrono::steady_clock::now();
            bestEnq = std::min(bestEnq, tms(ta, tb));
            bestFin = std::min(bestFin, tms(tb, tc));
        }
        snprintf(line, sizeof(line), "ocapi %3d kernels: enqueue %7.3f ms (%5.1f us/cmd), clFinish %8.3f ms (%5.1f us/kernel)\n",
                 kn, bestEnq, bestEnq * 1000 / kn, bestFin, bestFin * 1000 / kn);
        emit(line);
    }
    // C. setKernelArg cost
    {
        auto ta = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i) {
            setArg(kernel, 0, sizeof(void*), &buffer);
        }
        auto tb = std::chrono::steady_clock::now();
        snprintf(line, sizeof(line), "ocapi setKernelArg: %.2f us/call\n", tms(ta, tb));
        emit(line);
    }
    emit("ocapi: done\n");
}

void* runNativeBenchThread(void* raw) {
    auto work = static_cast<NativeBenchWork*>(raw);
    if (work->args == "@vkinfo") {
        runVkInfo(work->output);
        return nullptr;
    }
    if (work->args == "@vkapi") {
        runVkApiBench(work->output);
        return nullptr;
    }
    if (work->args == "@ocapi") {
        runOcApiBench(work->output);
        return nullptr;
    }
    if (work->args == "@vkbw") {
        runVkBw(work->output);
        return nullptr;
    }
    if (work->args == "@vkgemv") {
        runVkGemvPattern(work->output);
        return nullptr;
    }
    if (work->args == "@vkgemvc") {
        runVkGemvPattern(work->output, true);
        return nullptr;
    }
    if (work->args == "@ocgemv") {
        runOcGemvPattern(work->output);
        return nullptr;
    }
    // @vksubgroup temporarily disabled: the hand-rolled device/pipeline probe
    // crashed the app on Maleoon; needs bisection before re-enabling.
    // if (work->args == "@vksubgroup") {
    //     runVkSubgroupProbe(work->output);
    //     return nullptr;
    // }
    if (work->args == "@microbench") {
#if defined(__aarch64__)
        runFmaMicrobench(work->output);
#else
        work->error = "microbench requires aarch64";
#endif
        return nullptr;
    }
    if (work->args == "@clinfo") {
        runClInfo(work->output);
        return nullptr;
    }
    if (work->args == "@gpubench") {
        runGpuBench(work->output);
        return nullptr;
    }
    if (applyPinPrefix(work->args) ) {
        work->output += "[pinned]\n";
    }
    // Watchdog must outlive llm_bench_main: all-process pinning stays active
    // for the entire run and is lifted when this scope exits.
    std::unique_ptr<PinWatchdog> pinAll = applyPinAllPrefix(work->args);
    if (pinAll) {
        work->output += "[pinall]\n";
    }
    auto tokens = tokenizeCommandLine(work->args);
    if (!tokens.empty() && tokens[0] == "@dump") {
        setenv("MNN_DUMP_OPS", "1", 1);
        tokens.erase(tokens.begin());
    }
    if (tokens.empty()) {
        work->error = "empty benchmark arguments";
        return nullptr;
    }
    std::vector<std::string> storage;
    storage.push_back("llm_bench");
    for (auto& token : tokens) {
        storage.push_back(std::move(token));
    }
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& item : storage) {
        argv.push_back(const_cast<char*>(item.c_str()));
    }

    // llm_bench resolves its tuning cache ("tmp/mnn_cachefile.bin") and relative
    // model paths against the CWD; the app process CWD (/) is not writable. Run
    // from the app cache dir so the OpenCL tuning cache persists across runs.
    static const char* benchCwd = "/data/storage/el2/base/haps/entry/cache";
    char prevCwd[512] = {0};
    const bool cwdOk = getcwd(prevCwd, sizeof(prevCwd)) != nullptr && chdir(benchCwd) == 0;

    const char* logPath = "/data/storage/el2/base/haps/entry/cache/native_bench.log";
    fflush(nullptr);
    const int savedOut = dup(STDOUT_FILENO);
    const int savedErr = dup(STDERR_FILENO);
    const int logFd = open(logPath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    const bool redirected = logFd >= 0 && savedOut >= 0 && savedErr >= 0;
    if (redirected) {
        dup2(logFd, STDOUT_FILENO);
        dup2(logFd, STDERR_FILENO);
    }
    if (logFd >= 0) {
        close(logFd);
    }

    if (argv.size() > 1 && argv[1] == std::string("@gemvbw")) {
        // GEMV bandwidth roofline: "@gemvbw M K threads precision" (argv[0] is "llm_bench").
        int M = argv.size() > 2 ? atoi(argv[2]) : 1024;
        int K = argv.size() > 3 ? atoi(argv[3]) : 1024;
        int threads = argv.size() > 4 ? atoi(argv[4]) : 4;
        int precision = argv.size() > 5 ? atoi(argv[5]) : 2;
        work->exitCode = runGemvBWRoofline(M, K, threads, precision);
    } else if (argv.size() > 1 && argv[1] == std::string("@gemm")) {
        // Warm compute GEMM roofline: "@gemm M oc ic threads precision nbit".
        int M = argv.size() > 2 ? atoi(argv[2]) : 512;
        int oc = argv.size() > 3 ? atoi(argv[3]) : 1024;
        int ic = argv.size() > 4 ? atoi(argv[4]) : 1024;
        int threads = argv.size() > 5 ? atoi(argv[5]) : 6;
        int precision = argv.size() > 6 ? atoi(argv[6]) : 2;
        int nbit = argv.size() > 7 ? atoi(argv[7]) : 4;
        int pinMode = argv.size() > 8 ? atoi(argv[8]) : 0;
        work->exitCode = runGemmRoofline(M, oc, ic, threads, precision, nbit, pinMode);
    } else if (argv.size() > 1 && argv[1] == std::string("@sdotphys")) {
        int t = argv.size() > 2 ? atoi(argv[2]) : 5;
        int it = argv.size() > 3 ? atoi(argv[3]) : 20000000;
        int n = argv.size() > 4 ? atoi(argv[4]) : 24;
        work->exitCode = runSdotPhys(t, it, n);
    } else if (argv.size() > 1 && argv[1] == std::string("@cpuset")) {
        work->exitCode = runCpuset();
    } else if (argv.size() > 1 && argv[1] == std::string("@topo")) {
        work->exitCode = runTopo();
    } else if (argv.size() > 1 && argv[1] == std::string("@sdotmem")) {
        int mode = argv.size() > 2 ? atoi(argv[2]) : 0;
        int iters = argv.size() > 3 ? atoi(argv[3]) : 2000000;
        int core = argv.size() > 4 ? atoi(argv[4]) : 6;
        int threads = argv.size() > 5 ? atoi(argv[5]) : 1;
        work->exitCode = runSdotMem(mode, iters, core, threads);
    } else if (argv.size() > 1 && argv[1] == std::string("@sdotn")) {
        int n = argv.size() > 2 ? atoi(argv[2]) : 24;
        int iters = argv.size() > 3 ? atoi(argv[3]) : 20000000;
        int core = argv.size() > 4 ? atoi(argv[4]) : 6;
        int threads = argv.size() > 5 ? atoi(argv[5]) : 1;
        work->exitCode = runSdotN(n, iters, core, threads);
    } else if (argv.size() > 1 && argv[1] == std::string("@sdot")) {
        // Pure int8 sdot compute roofline: "@sdot threads iters".
        int threads = argv.size() > 2 ? atoi(argv[2]) : 1;
        int iters = argv.size() > 3 ? atoi(argv[3]) : 20000000;
        int core = argv.size() > 4 ? atoi(argv[4]) : 12;
        work->exitCode = runSdotPeak(threads, iters, core);
    } else {
        work->exitCode = llm_bench_main(static_cast<int>(argv.size()), argv.data());
    }

    if (cwdOk) {
        chdir(prevCwd);
    }
    if (redirected) {
        fflush(nullptr);
        dup2(savedOut, STDOUT_FILENO);
        dup2(savedErr, STDERR_FILENO);
    }
    if (savedOut >= 0) {
        close(savedOut);
    }
    if (savedErr >= 0) {
        close(savedErr);
    }

    FILE* log = fopen(logPath, "r");
    if (log != nullptr) {
        char buffer[4096];
        size_t count = 0;
        while ((count = fread(buffer, 1, sizeof(buffer), log)) > 0) {
            work->output.append(buffer, count);
        }
        fclose(log);
    }
    return nullptr;
}

napi_value RunNativeBench(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    auto context = std::make_unique<NativeBenchWork>();
    if (argc != 1 || !readString(env, args[0], context->args)) {
        napi_throw_type_error(env, nullptr, "runNativeBench expects a command-line string");
        return nullptr;
    }
    bool expected = false;
    if (!gNativeBenchRunning.compare_exchange_strong(expected, true)) {
        napi_throw_error(env, nullptr, "a native benchmark is already running");
        return nullptr;
    }
    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName;
    napi_create_string_utf8(env, "MNN native bench", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_async_work(
        env, nullptr, resourceName,
        [](napi_env, void* data) {
            RunOnBigStack(data, runNativeBenchThread);
        },
        [](napi_env env, napi_status status, void* data) {
            std::unique_ptr<NativeBenchWork> work(static_cast<NativeBenchWork*>(data));
            gNativeBenchRunning.store(false);
            if (status == napi_ok && work->error.empty()) {
                napi_value result;
                napi_create_object(env, &result);
                napi_value exitCode;
                napi_create_int32(env, work->exitCode, &exitCode);
                napi_set_named_property(env, result, "exitCode", exitCode);
                setString(env, result, "output", work->output);
                napi_resolve_deferred(env, work->deferred, result);
            } else {
                napi_reject_deferred(env, work->deferred,
                                     createError(env, work->error.empty() ? "native benchmark failed" : work->error));
            }
            napi_delete_async_work(env, work->work);
        },
        context.get(), &context->work);
    napi_queue_async_work(env, context.release()->work);
    return promise;
}

napi_value GetMnnVersion(napi_env env, napi_callback_info) {
    napi_value result;
    napi_create_string_utf8(env, MNN_VERSION, NAPI_AUTO_LENGTH, &result);
    return result;
}

napi_value DumpLlmConfig(napi_env env, napi_callback_info) {
    const std::string config = LlmSession::instance().dumpConfig();
    napi_value result;
    napi_create_string_utf8(env, config.c_str(), config.size(), &result);
    return result;
}

} // namespace

napi_value InitLlmModule(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"loadLlm", nullptr, LoadLlm, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"generateLlm", nullptr, GenerateLlm, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runLlmBenchmark", nullptr, RunLlmBenchmark, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"cancelLlm", nullptr, CancelLlm, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetLlm", nullptr, ResetLlm, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"releaseLlm", nullptr, ReleaseLlm, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getMnnVersion", nullptr, GetMnnVersion, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"dumpLlmConfig", nullptr, DumpLlmConfig, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runDiffusion", nullptr, RunDiffusion, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runNativeBench", nullptr, RunNativeBench, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
