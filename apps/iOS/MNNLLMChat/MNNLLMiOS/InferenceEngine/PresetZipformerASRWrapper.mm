//
//  PresetZipformerASRWrapper.mm
//  MNNLLMiOS
//

#import "PresetZipformerASRWrapper.h"

#include "sherpa-mnn/c-api/c-api.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kZipformerSampleRate = 16000;
constexpr int32_t kZipformerFeatureDimension = 80;
// X-ASR Zipformer2 needs enough right-context frames to expose the final
// token. This padding is decoded immediately and does not sleep in real time.
constexpr double kTailPaddingSeconds = 1.6;
constexpr int32_t kRecognizerThreads = 2;

struct RecognitionJob {
    std::vector<float> samples;
    int32_t sampleRate = 0;
    uint64_t generation = 0;
    std::string text;
    std::string error;
    double inferenceSeconds = 0.0;
    bool done = false;
    std::mutex mutex;
    std::condition_variable condition;
};

class ZipformerWorker {
public:
    explicit ZipformerWorker(std::string modelDirectory) : mModelDirectory(std::move(modelDirectory)) {
        mThread = std::thread(&ZipformerWorker::threadMain, this);
        std::unique_lock<std::mutex> lock(mStateMutex);
        mStateCondition.wait(lock, [this] { return mInitialized; });
    }

    ~ZipformerWorker() {
        shutdown();
    }

    ZipformerWorker(const ZipformerWorker&) = delete;
    ZipformerWorker& operator=(const ZipformerWorker&) = delete;

    bool ready() const {
        std::lock_guard<std::mutex> lock(mStateMutex);
        return mReady;
    }

    double initializationSeconds() const {
        std::lock_guard<std::mutex> lock(mStateMutex);
        return mInitializationSeconds;
    }

    std::string initializationError() const {
        std::lock_guard<std::mutex> lock(mStateMutex);
        return mInitializationError;
    }

    bool recognize(const float* samples, size_t count, int32_t sampleRate, std::string& text,
                   double& inferenceSeconds, std::string& error) {
        if (samples == nullptr || count == 0 || sampleRate <= 0) {
            error = "Invalid PCM input";
            return false;
        }
        if (!ready()) {
            error = initializationError();
            if (error.empty()) {
                error = "X-ASR recognizer is not ready";
            }
            return false;
        }

        auto job = std::make_shared<RecognitionJob>();
        job->samples.assign(samples, samples + count);
        job->sampleRate = sampleRate;
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            if (mStopping) {
                error = "X-ASR worker is shutting down";
                return false;
            }
            job->generation = ++mGeneration;
            mJobs.emplace_back(job);
        }
        mQueueCondition.notify_one();

        std::unique_lock<std::mutex> jobLock(job->mutex);
        job->condition.wait(jobLock, [&job] { return job->done; });
        text = job->text;
        inferenceSeconds = job->inferenceSeconds;
        error = job->error;
        return error.empty();
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        ++mGeneration;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            if (mStopping) {
                return;
            }
            mStopping = true;
            ++mGeneration;
        }
        mQueueCondition.notify_one();
        if (mThread.joinable()) {
            mThread.join();
        }
    }

private:
    void threadMain() {
        initializeRecognizer();
        while (true) {
            std::shared_ptr<RecognitionJob> job;
            {
                std::unique_lock<std::mutex> lock(mQueueMutex);
                mQueueCondition.wait(lock, [this] { return mStopping || !mJobs.empty(); });
                if (mStopping && mJobs.empty()) {
                    break;
                }
                job = std::move(mJobs.front());
                mJobs.pop_front();
            }

            if (!isCurrent(job->generation)) {
                complete(job, "Recognition canceled");
                continue;
            }
            runRecognition(job);
            if (!isCurrent(job->generation)) {
                job->text.clear();
                job->error = "Recognition canceled";
            }
            complete(job, job->error);
        }

        cancelQueuedJobs();
        if (mRecognizer != nullptr) {
            SherpaMnnDestroyOnlineRecognizer(mRecognizer);
            mRecognizer = nullptr;
        }
    }

    void initializeRecognizer() {
        const auto start = std::chrono::steady_clock::now();
        const std::string encoder = mModelDirectory + "/encoder-160ms.mnn";
        const std::string decoder = mModelDirectory + "/decoder-160ms.mnn";
        const std::string joiner = mModelDirectory + "/joiner-160ms.mnn";
        const std::string tokens = mModelDirectory + "/tokens.txt";

        SherpaMnnOnlineRecognizerConfig config;
        std::memset(&config, 0, sizeof(config));
        config.feat_config.sample_rate = kZipformerSampleRate;
        config.feat_config.feature_dim = kZipformerFeatureDimension;
        config.model_config.transducer.encoder = encoder.c_str();
        config.model_config.transducer.decoder = decoder.c_str();
        config.model_config.transducer.joiner = joiner.c_str();
        config.model_config.tokens = tokens.c_str();
        config.model_config.num_threads = kRecognizerThreads;
        config.model_config.provider = "cpu";
        config.model_config.debug = 0;
        config.model_config.model_type = "zipformer2";
        config.model_config.modeling_unit = "";
        config.model_config.bpe_vocab = "";
        config.model_config.tokens_buf = nullptr;
        config.model_config.tokens_buf_size = 0;
        config.model_config.paraformer.encoder = "";
        config.model_config.paraformer.decoder = "";
        config.model_config.zipformer2_ctc.model = "";
        config.decoding_method = "greedy_search";
        config.max_active_paths = 4;
        config.enable_endpoint = 0;
        config.hotwords_file = "";
        config.hotwords_score = 1.5f;
        config.ctc_fst_decoder_config.graph = "";
        config.ctc_fst_decoder_config.max_active = 3000;
        config.rule_fsts = "";
        config.rule_fars = "";
        config.hotwords_buf = nullptr;
        config.hotwords_buf_size = 0;

        mRecognizer = SherpaMnnCreateOnlineRecognizer(&config);
        const auto end = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mStateMutex);
            mInitializationSeconds = std::chrono::duration<double>(end - start).count();
            mReady = mRecognizer != nullptr;
            if (!mReady) {
                mInitializationError = "Failed to create sherpa-mnn X-ASR Zipformer2 recognizer";
            }
            mInitialized = true;
        }
        mStateCondition.notify_all();
    }

    void runRecognition(const std::shared_ptr<RecognitionJob>& job) {
        const SherpaMnnOnlineStream* stream = SherpaMnnCreateOnlineStream(mRecognizer);
        if (stream == nullptr) {
            job->error = "Failed to create X-ASR stream";
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        SherpaMnnOnlineStreamAcceptWaveform(stream, job->sampleRate, job->samples.data(),
                                           static_cast<int32_t>(job->samples.size()));
        const int32_t tailPaddingSamples = static_cast<int32_t>(job->sampleRate * kTailPaddingSeconds);
        const std::vector<float> tailPadding(tailPaddingSamples, 0.0f);
        SherpaMnnOnlineStreamAcceptWaveform(stream, job->sampleRate, tailPadding.data(), tailPaddingSamples);
        SherpaMnnOnlineStreamInputFinished(stream);
        while (SherpaMnnIsOnlineStreamReady(mRecognizer, stream)) {
            SherpaMnnDecodeOnlineStream(mRecognizer, stream);
        }

        const SherpaMnnOnlineRecognizerResult* result = SherpaMnnGetOnlineStreamResult(mRecognizer, stream);
        const auto end = std::chrono::steady_clock::now();
        job->inferenceSeconds = std::chrono::duration<double>(end - start).count();
        if (result != nullptr) {
            if (result->text != nullptr) {
                job->text = result->text;
            }
            SherpaMnnDestroyOnlineRecognizerResult(result);
        } else {
            job->error = "X-ASR returned no result";
        }
        SherpaMnnDestroyOnlineStream(stream);

        if (job->error.empty() && job->text.empty()) {
            job->error = "X-ASR returned an empty transcript";
        }
    }

    bool isCurrent(uint64_t generation) const {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        return !mStopping && generation == mGeneration;
    }

    void complete(const std::shared_ptr<RecognitionJob>& job, const std::string& error) {
        {
            std::lock_guard<std::mutex> lock(job->mutex);
            job->error = error;
            job->done = true;
        }
        job->condition.notify_all();
    }

    void cancelQueuedJobs() {
        std::deque<std::shared_ptr<RecognitionJob>> jobs;
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            jobs.swap(mJobs);
        }
        for (const auto& job : jobs) {
            complete(job, "Recognition canceled");
        }
    }

private:
    std::string mModelDirectory;
    const SherpaMnnOnlineRecognizer* mRecognizer = nullptr;
    std::thread mThread;

    mutable std::mutex mStateMutex;
    std::condition_variable mStateCondition;
    bool mInitialized = false;
    bool mReady = false;
    double mInitializationSeconds = 0.0;
    std::string mInitializationError;

    mutable std::mutex mQueueMutex;
    std::condition_variable mQueueCondition;
    std::deque<std::shared_ptr<RecognitionJob>> mJobs;
    uint64_t mGeneration = 0;
    bool mStopping = false;
};

NSError* MakeASRError(NSString* description) {
    return [NSError errorWithDomain:@"com.alibaba.mnnchat.preset-asr"
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey : description ?: @"X-ASR failed"}];
}

}  // namespace

@implementation PresetZipformerASRWrapper {
    std::unique_ptr<ZipformerWorker> _worker;
}

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory {
    self = [super init];
    if (self) {
        _worker.reset(new ZipformerWorker(modelDirectory.UTF8String));
    }
    return self;
}

- (BOOL)isReady {
    return _worker != nullptr && _worker->ready();
}

- (NSTimeInterval)initializationSeconds {
    return _worker != nullptr ? _worker->initializationSeconds() : 0.0;
}

- (NSString *)initializationError {
    if (_worker == nullptr) {
        return @"X-ASR worker is unavailable";
    }
    const std::string error = _worker->initializationError();
    return error.empty() ? nil : [NSString stringWithUTF8String:error.c_str()];
}

- (nullable NSString *)recognizeFloatPCM:(NSData *)pcm
                              sampleRate:(int32_t)sampleRate
                        inferenceSeconds:(NSTimeInterval *)inferenceSeconds
                                   error:(NSError **)error {
    if (_worker == nullptr || !self.isReady) {
        if (error != nullptr) {
            *error = MakeASRError(self.initializationError ?: @"X-ASR recognizer is not ready");
        }
        return nil;
    }
    if (pcm.length == 0 || pcm.length % sizeof(float) != 0) {
        if (error != nullptr) {
            *error = MakeASRError(@"Invalid Float32 PCM buffer");
        }
        return nil;
    }

    std::string text;
    std::string nativeError;
    double nativeInferenceSeconds = 0.0;
    const bool success = _worker->recognize(static_cast<const float*>(pcm.bytes), pcm.length / sizeof(float),
                                            sampleRate, text, nativeInferenceSeconds, nativeError);
    if (inferenceSeconds != nullptr) {
        *inferenceSeconds = nativeInferenceSeconds;
    }
    if (!success) {
        if (error != nullptr) {
            NSString *description = [NSString stringWithUTF8String:nativeError.c_str()];
            *error = MakeASRError(description);
        }
        return nil;
    }
    return [NSString stringWithUTF8String:text.c_str()];
}

- (void)cancel {
    if (_worker != nullptr) {
        _worker->cancel();
    }
}

- (void)shutdown {
    if (_worker != nullptr) {
        _worker->shutdown();
        _worker.reset();
    }
}

- (void)dealloc {
    [self shutdown];
}

@end
