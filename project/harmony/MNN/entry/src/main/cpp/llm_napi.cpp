#include "llm_napi.h"

#include "diffusion_session.h"
#include "llm_session.h"

#include <MNN/MNNDefine.h>

#include <pthread.h>

#include <memory>
#include <string>
#include <utility>

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
            auto work = static_cast<LoadWork*>(data);
            LlmSession::instance().load(work->configPath, work->runtimeConfigJson, work->loadMs, work->reused,
                                        work->error);
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
    GenerateRequest request;
    GenerateResult result;
    std::string error;
};

napi_value GenerateLlm(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    auto context = std::make_unique<GenerateWork>();
    napi_valuetype callbackType = napi_undefined;
    if (argc != 2 || !readGenerateRequest(env, args[0], context->request) ||
        napi_typeof(env, args[1], &callbackType) != napi_ok || callbackType != napi_function) {
        napi_throw_type_error(env, nullptr, "generateLlm expects a request object and onChunk callback");
        return nullptr;
    }

    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName;
    napi_create_string_utf8(env, "MNN LLM generate", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_threadsafe_function(env, args[1], nullptr, resourceName, 0, 1, nullptr, nullptr, nullptr, callChunk,
                                    &context->callback);
    LlmSession::instance().registerRequest(context->request.requestId);
    napi_create_async_work(
        env, nullptr, resourceName,
        [](napi_env, void* data) {
            auto work = static_cast<GenerateWork*>(data);
            LlmSession::instance().generate(
                work->request,
                [work](const std::string& chunk) {
                    auto payload = new std::string(chunk);
                    if (napi_call_threadsafe_function(work->callback, payload, napi_tsfn_nonblocking) != napi_ok) {
                        delete payload;
                    }
                },
                work->result, work->error);
            napi_release_threadsafe_function(work->callback, napi_tsfn_release);
            work->callback = nullptr;
        },
        [](napi_env env, napi_status status, void* data) {
            std::unique_ptr<GenerateWork> work(static_cast<GenerateWork*>(data));
            if (work->callback) {
                napi_release_threadsafe_function(work->callback, napi_tsfn_abort);
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
        context->request.repeats > 20 || napi_typeof(env, args[1], &callbackType) != napi_ok ||
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
            pthread_attr_t attributes;
            pthread_attr_init(&attributes);
            pthread_attr_setstacksize(&attributes, 32 * 1024 * 1024);
            pthread_t thread;
            if (pthread_create(&thread, &attributes, runDiffusionThread, data) == 0) {
                pthread_join(thread, nullptr);
            } else {
                runDiffusionThread(data);
            }
            pthread_attr_destroy(&attributes);
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
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
