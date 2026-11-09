#include "llm_session.h"

#include <chrono>
#include <ostream>
#include <streambuf>
#include <utility>

using MNN::Transformer::Llm;
using MNN::Transformer::LlmContext;
using MNN::Transformer::LlmStatus;

namespace mnn_harmony {
namespace {

class BenchmarkStreamBuffer : public std::streambuf {
public:
    BenchmarkStreamBuffer(Llm* llm, const LlmSession::CancellationFlag& cancellation)
        : mLlm(llm), mCancellation(cancellation) {}

protected:
    std::streamsize xsputn(const char*, std::streamsize size) override {
        applyCancellation();
        return size;
    }

    int overflow(int value) override {
        applyCancellation();
        return value == traits_type::eof() ? traits_type::not_eof(value) : value;
    }

private:
    void applyCancellation() {
        if (!mCancellation->load()) {
            return;
        }
        auto context = const_cast<LlmContext*>(mLlm->getContext());
        if (context) {
            context->status = LlmStatus::USER_CANCEL;
        }
    }

    Llm* mLlm;
    LlmSession::CancellationFlag mCancellation;
};

class ChunkStreamBuffer : public std::streambuf {
public:
    ChunkStreamBuffer(const LlmSession::ChunkCallback& callback, Llm* llm,
                      const LlmSession::CancellationFlag& cancellation, std::chrono::steady_clock::time_point start,
                      double& firstTokenMs)
        : mCallback(callback), mLlm(llm), mCancellation(cancellation), mStart(start), mFirstTokenMs(firstTokenMs) {}

    std::string takeResult() { return std::move(mResult); }

    void finish() {
        // No more input: a held partial marker was ordinary text after all.
        if (!mHeldMarker.empty()) {
            mPending.insert(0, mHeldMarker);
            mHeldMarker.clear();
        }
        const size_t length = completeUtf8Prefix(mPending);
        if (length > 0) {
            const std::string chunk = mPending.substr(0, length);
            mResult.append(chunk);
            dispatch(chunk);
            mPending.clear();
        }
    }

protected:
    std::streamsize xsputn(const char* text, std::streamsize size) override {
        if (size <= 0) {
            return size;
        }
        if (mEnded) {
            // Text after the stop marker is not part of the answer.
            return size;
        }
        // Re-assemble with the marker prefix held back and any incomplete UTF-8 tail left over
        // from the previous write; both must be carried over or characters are dropped.
        std::string buffer;
        buffer.reserve(mHeldMarker.size() + mPending.size() + static_cast<size_t>(size));
        buffer.append(mHeldMarker);
        buffer.append(mPending);
        buffer.append(text, static_cast<size_t>(size));
        mHeldMarker.clear();
        mPending.clear();

        // Strip the "<eop>" stop marker (same end_with token as the Android client). Hold back a
        // trailing partial match so a marker split across writes never reaches the UI.
        static const char kMarker[] = "<eop>";
        constexpr size_t kMarkerLength = sizeof(kMarker) - 1;
        const size_t markerPos = buffer.find(kMarker);
        if (markerPos != std::string::npos) {
            buffer.erase(markerPos);
            mEnded = true;
        } else {
            for (size_t keep = kMarkerLength - 1; keep > 0; --keep) {
                if (buffer.size() >= keep && buffer.compare(buffer.size() - keep, keep, kMarker, keep) == 0) {
                    mHeldMarker = buffer.substr(buffer.size() - keep);
                    buffer.erase(buffer.size() - keep);
                    break;
                }
            }
        }
        mPending = std::move(buffer);

        const size_t length = completeUtf8Prefix(mPending);
        if (length > 0) {
            const std::string chunk = mPending.substr(0, length);
            mResult.append(chunk);
            dispatch(chunk);
            mPending.erase(0, length);
        }
        applyCancellation();
        return size;
    }

    int overflow(int value) override {
        if (value == traits_type::eof()) {
            return traits_type::not_eof(value);
        }
        const char byte = static_cast<char>(value);
        xsputn(&byte, 1);
        return value;
    }

private:
    static size_t completeUtf8Prefix(const std::string& text) {
        size_t index = 0;
        while (index < text.size()) {
            const unsigned char lead = static_cast<unsigned char>(text[index]);
            size_t width = 1;
            if ((lead & 0xE0) == 0xC0) {
                width = 2;
            } else if ((lead & 0xF0) == 0xE0) {
                width = 3;
            } else if ((lead & 0xF8) == 0xF0) {
                width = 4;
            }
            if (index + width > text.size()) {
                break;
            }
            index += width;
        }
        return index;
    }

    void dispatch(const std::string& chunk) {
        if (chunk.empty()) {
            return;
        }
        if (mFirstTokenMs == 0.0) {
            mFirstTokenMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mStart).count();
        }
        mCallback(chunk);
    }

    void applyCancellation() {
        if (!mCancellation->load()) {
            return;
        }
        auto context = const_cast<LlmContext*>(mLlm->getContext());
        if (context) {
            context->status = LlmStatus::USER_CANCEL;
        }
    }

    LlmSession::ChunkCallback mCallback;
    Llm* mLlm;
    LlmSession::CancellationFlag mCancellation;
    std::chrono::steady_clock::time_point mStart;
    double& mFirstTokenMs;
    std::string mResult;
    std::string mPending;
    std::string mHeldMarker;
    bool mEnded = false;
};

} // namespace

LlmSession& LlmSession::instance() {
    static LlmSession session;
    return session;
}

void LlmSession::load(const std::string& configPath, const std::string& runtimeConfigJson, double& loadMs, bool& reused,
                      std::string& error) {
    std::lock_guard<std::mutex> lock(mOperationMutex);
    const std::string loadIdentity = configPath + "\n" + runtimeConfigJson;
    if (mLlm && mLoadIdentity == loadIdentity) {
        loadMs = mLoadMs;
        reused = true;
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    if (!llm) {
        error = "Failed to create LLM";
        return;
    }
    if (!llm->set_config(runtimeConfigJson)) {
        error = "Invalid LLM runtime config";
        return;
    }
    if (!llm->load()) {
        error = "Failed to load LLM";
        return;
    }

    mLlm = std::move(llm);
    {
        std::lock_guard<std::mutex> configLock(mConfigMutex);
        mConfigDump = mLlm->dump_config();
    }
    mLoadIdentity = loadIdentity;
    mLoadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    loadMs = mLoadMs;
    reused = false;
}

void LlmSession::registerRequest(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(mCancellationMutex);
    mCancellations[requestId] = std::make_shared<std::atomic<bool>>(false);
}

void LlmSession::generate(const GenerateRequest& request, const ChunkCallback& onChunk, GenerateResult& result,
                          std::string& error) {
    std::lock_guard<std::mutex> lock(mOperationMutex);
    const auto cancellation = getCancellation(request.requestId);
    result.requestId = request.requestId;

    if (!mLlm) {
        error = "LLM is not loaded";
        completeRequest(request.requestId, cancellation);
        return;
    }
    if (cancellation->load()) {
        result.canceled = true;
        completeRequest(request.requestId, cancellation);
        return;
    }

    prepareForResponse();
    const auto start = std::chrono::steady_clock::now();
    ChunkStreamBuffer outputBuffer(onChunk, mLlm.get(), cancellation, start, result.firstTokenMs);
    std::ostream output(&outputBuffer);
    // Same end_with marker as the Android client; the stream buffer strips it before it reaches the UI.
    const int maxTokens = request.maxTokens <= 0 ? 2048 : (request.maxTokens > 8192 ? 8192 : request.maxTokens);
    mLlm->response(request.messages, &output, "<eop>", maxTokens);
    const auto end = std::chrono::steady_clock::now();
    outputBuffer.finish();
    result.text = outputBuffer.takeResult();
    result.totalMs = std::chrono::duration<double, std::milli>(end - start).count();

    const auto context = mLlm->getContext();
    if (context) {
        result.promptTokens = context->prompt_len;
        result.generatedTokens = context->gen_seq_len;
        result.prefillMs = context->prefill_us / 1000.0;
        result.decodeMs = context->decode_us / 1000.0;
        result.canceled = cancellation->load() || context->status == LlmStatus::USER_CANCEL;
        if (context->status == LlmStatus::INTERNAL_ERROR || context->status == LlmStatus::TIMEOUT) {
            error = context->status == LlmStatus::TIMEOUT ? "LLM generation timed out" : "LLM generation failed";
        }
    }
    // Parity with the Android client: a user cancel keeps the KV cache (the next request resends the
    // full history and only the delta is prefilled); only a broken session is reset.
    if (!error.empty()) {
        mLlm->reset();
        prepareForResponse();
    }
    completeRequest(request.requestId, cancellation);
}

void LlmSession::benchmark(const BenchmarkRequest& request, const BenchmarkCallback& onProgress,
                           BenchmarkResult& result, std::string& error) {
    std::lock_guard<std::mutex> lock(mOperationMutex);
    const auto cancellation = getCancellation(request.requestId);
    result.requestId = request.requestId;
    if (!mLlm) {
        error = "LLM is not loaded";
        completeRequest(request.requestId, cancellation);
        return;
    }

    const int totalIterations = request.repeats + 1;
    for (int iteration = 0; iteration < totalIterations; ++iteration) {
        if (cancellation->load()) {
            result.canceled = true;
            break;
        }
        mLlm->reset();
        prepareForResponse();
        const auto start = std::chrono::steady_clock::now();
        BenchmarkStreamBuffer outputBuffer(mLlm.get(), cancellation);
        std::ostream output(&outputBuffer);
        std::vector<int> promptTokens(request.promptTokens, 500);
        mLlm->response(promptTokens, &output, "", 1);
        const auto prefillContext = mLlm->getContext();
        if (!prefillContext) {
            error = "LLM benchmark prefill failed";
            break;
        }
        const double prefillMs = prefillContext->prefill_us / 1000.0;
        if (cancellation->load() || prefillContext->status == LlmStatus::USER_CANCEL) {
            result.canceled = true;
            break;
        }

        prepareForResponse();
        std::vector<int> decodeInput(1, 500);
        mLlm->response(decodeInput, &output, "", request.generateTokens);
        const auto decodeContext = mLlm->getContext();
        if (!decodeContext) {
            error = "LLM benchmark decode failed";
            break;
        }
        if (cancellation->load() || decodeContext->status == LlmStatus::USER_CANCEL) {
            result.canceled = true;
            break;
        }
        if (decodeContext->status == LlmStatus::INTERNAL_ERROR || decodeContext->status == LlmStatus::TIMEOUT) {
            error = decodeContext->status == LlmStatus::TIMEOUT ? "LLM benchmark timed out" : "LLM benchmark failed";
            break;
        }
        if (iteration > 0) {
            BenchmarkSample sample;
            sample.iteration = iteration;
            sample.promptTokens = request.promptTokens;
            sample.generatedTokens = decodeContext->gen_seq_len;
            sample.prefillMs = prefillMs;
            sample.decodeMs = decodeContext->decode_us / 1000.0;
            sample.totalMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            result.samples.push_back(sample);
            onProgress(sample);
        }
    }
    mLlm->reset();
    prepareForResponse();
    completeRequest(request.requestId, cancellation);
}

void LlmSession::cancel(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(mCancellationMutex);
    const auto found = mCancellations.find(requestId);
    if (found != mCancellations.end()) {
        found->second->store(true);
    }
}

void LlmSession::reset() {
    std::lock_guard<std::mutex> lock(mOperationMutex);
    if (mLlm) {
        mLlm->reset();
        prepareForResponse();
    }
}

std::string LlmSession::dumpConfig() {
    std::lock_guard<std::mutex> lock(mConfigMutex);
    return mConfigDump;
}

void LlmSession::release() {
    std::lock_guard<std::mutex> lock(mOperationMutex);
    mLlm.reset();
    {
        std::lock_guard<std::mutex> configLock(mConfigMutex);
        mConfigDump.clear();
    }
    mLoadIdentity.clear();
    mLoadMs = 0.0;
    std::lock_guard<std::mutex> cancellationLock(mCancellationMutex);
    mCancellations.clear();
}

LlmSession::CancellationFlag LlmSession::getCancellation(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(mCancellationMutex);
    const auto found = mCancellations.find(requestId);
    if (found != mCancellations.end()) {
        return found->second;
    }
    auto cancellation = std::make_shared<std::atomic<bool>>(false);
    mCancellations[requestId] = cancellation;
    return cancellation;
}

void LlmSession::completeRequest(const std::string& requestId, const CancellationFlag& cancellation) {
    std::lock_guard<std::mutex> lock(mCancellationMutex);
    const auto found = mCancellations.find(requestId);
    if (found != mCancellations.end() && found->second == cancellation) {
        mCancellations.erase(found);
    }
}

void LlmSession::prepareForResponse() {
    auto context = const_cast<LlmContext*>(mLlm->getContext());
    if (context && context->status != LlmStatus::NOT_LOADED) {
        context->status = LlmStatus::RUNNING;
    }
}

} // namespace mnn_harmony
