//
//  LLMInferenceEngineWrapper.m
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#import "LLMInferenceEngineWrapper.h"
#import <UIKit/UIApplication.h>
#include <MNN/llm/llm.hpp>
using namespace MNN::Transformer;

const char* GetMainBundleDirectory() {
    NSString *bundleDirectory = [[NSBundle mainBundle] bundlePath];
    return [bundleDirectory UTF8String];
}

// mmap cache dir must be wiped per load: stale caches from a previously bundled
// model survive app reinstall and would be silently reused, giving garbage output.
static std::string GetCleanTmpDirectory() {
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"mnn_mmap"];
    [[NSFileManager defaultManager] removeItemAtPath:dir error:nil];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
    return std::string([dir UTF8String]);
}

static void WaitForAppActive() {
    while (YES) {
        __block UIApplicationState state;
        if ([NSThread isMainThread]) {
            state = [UIApplication sharedApplication].applicationState;
        } else {
            dispatch_sync(dispatch_get_main_queue(), ^{
                state = [UIApplication sharedApplication].applicationState;
            });
        }
        if (state == UIApplicationStateActive) break;
        [NSThread sleepForTimeInterval:0.05];
    }
}

@implementation LLMInferenceEngineWrapper {
    std::shared_ptr<Llm> llm;
}

- (instancetype)initWithCompletionHandler:(ModelLoadingCompletionHandler)completionHandler {
    self = [super init];
    if (self) {
        // 在后台线程异步加载模型
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            BOOL success = [self loadModel]; // 假设loadModel方法加载模型并返回加载的成功或失败
            // 切回主线程回调
            dispatch_async(dispatch_get_main_queue(), ^{
                completionHandler(success);
            });
        });
    }
    return self;
}

- (BOOL)loadModel {
    if (!llm) {
        std::string model_dir = GetMainBundleDirectory();
        std::string config_path = model_dir + "/config.json";
        if (![[NSFileManager defaultManager] fileExistsAtPath:[NSString stringWithUTF8String:config_path.c_str()]]) {
            NSLog(@"[MNN_BENCH_ERROR] config.json not found in app bundle");
            return NO;
        }
        llm.reset(Llm::createLLM(config_path));
        if (!llm) {
            NSLog(@"[MNN_BENCH_ERROR] createLLM failed");
            return NO;
        }
        llm->set_config("{\"tmp_path\":\"" + GetCleanTmpDirectory() + "\", \"use_mmap\":true}");
        llm->load();
        NSLog(@"[MNN_BENCH] model_loaded");
    }
    return YES;
}
// Llm start
// llm stream buffer with callback
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    LlmStreamBuffer(CallBack callback) : callback_(callback) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }
private:
    CallBack callback_ = nullptr;
};
- (void)processInput:(NSString *)input withStreamHandler:(StreamOutputHandler)handler {
    LlmStreamBuffer::CallBack callback = [handler](const char* str, size_t len) {
        if (handler) {
            NSString *nsOutput = [NSString stringWithUTF8String:str];
            handler(nsOutput);
        }
    };
    LlmStreamBuffer streambuf(callback);
    std::ostream os(&streambuf);
    if ([input hasPrefix:@"ask "]) {
        [self runAsk:std::string([input UTF8String]) output:os];
    } else if ([input hasPrefix:@"benchfiles "]) {
        [self runFileBench:std::string([input UTF8String]) output:os];
    } else if ([input hasPrefix:@"bench "]) {
        [self runFixedBench:std::string([input UTF8String]) output:os];
    } else if (std::string([input UTF8String]) == "benchmark") {
        // do benchmark
        std::string model_dir = GetMainBundleDirectory();
        std::string prompt_file = model_dir + "/bench.txt";
        std::ifstream prompt_fs(prompt_file);
        std::vector<std::string> prompts;
        std::string prompt;
        while (std::getline(prompt_fs, prompt)) {
            // prompt start with '#' will be ignored
            if (prompt.substr(0, 1) == "#") {
                continue;
            }
            std::string::size_type pos = 0;
            while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
                prompt.replace(pos, 2, "\n");
                pos += 1;
            }
            prompts.push_back(prompt);
        }
        int prompt_len = 0;
        int decode_len = 0;
        int64_t prefill_time = 0;
        int64_t decode_time = 0;
        auto context = llm->getContext();
        for (int i = 0; i < prompts.size(); i++) {
            llm->response(prompts[i], &os, "\n");
            prompt_len += context->prompt_len;
            decode_len += context->gen_seq_len;
            prefill_time += context->prefill_us;
            decode_time += context->decode_us;
        }
        float prefill_s = prefill_time / 1e6;
        float decode_s = decode_time / 1e6;
        float prefill_speed = prefill_s > 0 ? prompt_len / prefill_s : 0.f;
        float decode_speed = decode_s > 0 ? decode_len / decode_s : 0.f;
        os << "\n#################################\n"
           << "prompt tokens num  = " << prompt_len << "\n"
           << "decode tokens num  = " << decode_len << "\n"
           << "prefill time = " << std::fixed << std::setprecision(2) << prefill_s << " s\n"
           << " decode time = " << std::fixed << std::setprecision(2) << decode_s << " s\n"
           << "prefill speed = " << std::fixed << std::setprecision(2) << prefill_speed << " tok/s\n"
           << " decode speed = " << std::fixed << std::setprecision(2) << decode_speed << " tok/s\n"
           << "##################################\n";
        os << "<eop>";
        NSLog(@"[MNN_BENCH] prompt_tokens=%d decode_tokens=%d prefill_s=%.2f decode_s=%.2f prefill_tok_s=%.2f decode_tok_s=%.2f",
              prompt_len, decode_len, prefill_s, decode_s, prefill_speed, decode_speed);
        NSLog(@"[MNN_BENCH_DONE]");
    } else {
        llm->response([input UTF8String], &os, "<eop>");
    }
}

- (void)dealloc {
    llm.reset();
}

// cmd: "ask <cpu|metal> <question>"
- (void)runAsk:(const std::string&)cmd output:(std::ostream&)os {
    std::istringstream iss(cmd);
    std::string tag, backend;
    iss >> tag >> backend;
    std::string question;
    std::getline(iss, question);
    while (!question.empty() && question.front() == ' ') question.erase(0, 1);
    if ((backend != "cpu" && backend != "metal") || question.empty()) {
        os << "usage: ask <cpu|metal> <question>\n<eop>";
        NSLog(@"[MNN_BENCH_ERROR] invalid ask command: %s", cmd.c_str());
        return;
    }
    WaitForAppActive();
    llm.reset();
    std::string model_dir = GetMainBundleDirectory();
    llm.reset(Llm::createLLM(model_dir + "/config.json"));
    if (!llm) {
        os << "createLLM failed\n<eop>";
        NSLog(@"[MNN_BENCH_ERROR] createLLM failed");
        return;
    }
    llm->set_config("{\"tmp_path\":\"" + GetCleanTmpDirectory() + "\", \"use_mmap\":true}");
    llm->set_config("{\"backend_type\":\"" + backend + "\"}");
    llm->set_config("{\"sampler_type\":\"greedy\"}");
    llm->load();
    std::ostringstream answer;
    llm->response(question, &answer, "");
    os << answer.str() << "\n<eop>";
    NSLog(@"[MNN_ASK_BEGIN] backend=%s question=%s", backend.c_str(), question.c_str());
    NSLog(@"[MNN_ASK_ANSWER] %s", answer.str().c_str());
    NSLog(@"[MNN_ASK_END]");
    NSLog(@"[MNN_BENCH_DONE]");
}

// Answers can exceed the os_log per-message limit; escape newlines and emit
// in UTF-8-safe chunks so the host script can reassemble the full text.
static void LogAnswerChunks(const std::string& answer) {
    std::string s;
    s.reserve(answer.size());
    for (char c : answer) {
        if (c == '\n') s += "\\n";
        else if (c != '\r') s += c;
    }
    const size_t kChunk = 700;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t len = std::min(kChunk, s.size() - pos);
        while (pos + len < s.size() && (s[pos + len] & 0xC0) == 0x80) len--;
        NSLog(@"[MNN_FILE_ANSWER] %s", s.substr(pos, len).c_str());
        pos += len;
    }
}

// cmd: "benchfiles <cpu|metal> [threads] [max_new_tokens] [file_index]"
// Runs every benchprompt_*.txt bundled with the app (or only the file_index-th
// when file_index >= 0), logging per-file perf and answer.
- (void)runFileBench:(const std::string&)cmd output:(std::ostream&)os {
    std::istringstream iss(cmd);
    std::string tag, backend;
    int threads = 4, max_new = 1024, file_index = -1;
    iss >> tag >> backend;
    int v;
    if (iss >> v) threads = v;
    if (iss >> v) max_new = v;
    if (iss >> v) file_index = v;
    if (backend != "cpu" && backend != "metal") {
        os << "usage: benchfiles <cpu|metal> [threads] [max_new_tokens]\n<eop>";
        NSLog(@"[MNN_BENCH_ERROR] invalid benchfiles command: %s", cmd.c_str());
        return;
    }
    std::string model_dir = GetMainBundleDirectory();
    NSString *bundleDir = [NSString stringWithUTF8String:model_dir.c_str()];
    NSArray *all = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:bundleDir error:nil]
                    sortedArrayUsingSelector:@selector(compare:)];
    std::vector<std::string> files;
    for (NSString *f in all) {
        if ([f hasPrefix:@"benchprompt_"] && [f hasSuffix:@".txt"]) {
            files.push_back(std::string([f UTF8String]));
        }
    }
    if (files.empty()) {
        os << "no benchprompt_*.txt in app bundle\n<eop>";
        NSLog(@"[MNN_BENCH_ERROR] no benchprompt_*.txt files bundled");
        return;
    }
    if (file_index >= 0) {
        if (file_index >= (int)files.size()) {
            os << "file_index out of range\n<eop>";
            NSLog(@"[MNN_BENCH_ERROR] file_index %d out of range (%zu files)", file_index, files.size());
            return;
        }
        files = {files[file_index]};
    }
    WaitForAppActive();
    llm.reset();
    llm.reset(Llm::createLLM(model_dir + "/config.json"));
    if (!llm) {
        os << "createLLM failed\n<eop>";
        NSLog(@"[MNN_BENCH_ERROR] createLLM failed");
        return;
    }
    llm->set_config("{\"tmp_path\":\"" + GetCleanTmpDirectory() + "\", \"use_mmap\":true}");
    llm->set_config("{\"backend_type\":\"" + backend + "\"}");
    llm->set_config("{\"thread_num\":" + std::to_string(threads) + "}");
    llm->set_config("{\"reuse_kv\":false}");
    llm->set_config("{\"async\":false}");
    llm->set_config("{\"sampler_type\":\"greedy\"}");
    llm->set_config("{\"max_new_tokens\":" + std::to_string(max_new) + "}");
    llm->load();
    auto context = llm->getContext();
    NSLog(@"[MNN_BENCH] filebench backend=%s threads=%d max_new=%d files=%zu",
          backend.c_str(), threads, max_new, files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        std::ifstream fs(model_dir + "/" + files[i]);
        std::string content((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
        while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
            content.pop_back();
        }
        if (content.empty()) continue;
        NSLog(@"[MNN_FILE_BEGIN] file=%s bytes=%zu", files[i].c_str(), content.size());
        os << "[" << (i + 1) << "/" << files.size() << "] " << files[i] << " ...\n";
        std::ostringstream answer;
        llm->response(content, &answer, "");
        double prefill_s = context->prefill_us / 1e6;
        double decode_s = context->decode_us / 1e6;
        double pf_speed = prefill_s > 0 ? context->prompt_len / prefill_s : 0;
        double dc_speed = decode_s > 0 ? context->gen_seq_len / decode_s : 0;
        LogAnswerChunks(answer.str());
        NSLog(@"[MNN_FILE_PERF] file=%s prompt_tokens=%d decode_tokens=%d prefill_s=%.2f decode_s=%.2f prefill_tok_s=%.2f decode_tok_s=%.2f",
              files[i].c_str(), context->prompt_len, context->gen_seq_len, prefill_s, decode_s, pf_speed, dc_speed);
        NSLog(@"[MNN_FILE_END] file=%s", files[i].c_str());
        os << "  prefill " << std::fixed << std::setprecision(2) << pf_speed
           << " tok/s, decode " << dc_speed << " tok/s\n";
    }
    os << "<eop>";
    NSLog(@"[MNN_BENCH_DONE]");
}

// cmd: "bench <cpu|metal> <prompt_len> <decode_len> [repeat] [threads] [attention_mode]"
- (void)runFixedBench:(const std::string&)cmd output:(std::ostream&)os {
    std::istringstream iss(cmd);
    std::string tag, backend;
    int prompt_len = 0, decode_len = 0, repeat = 3, threads = 4, attn_mode = -1;
    iss >> tag >> backend >> prompt_len >> decode_len;
    int v;
    if (iss >> v) repeat = v;
    if (iss >> v) threads = v;
    if (iss >> v) attn_mode = v;
    if ((backend != "cpu" && backend != "metal") || prompt_len <= 0 || decode_len < 0 || repeat <= 0) {
        os << "usage: bench <cpu|metal> <prompt_len> <decode_len> [repeat] [threads] [attention_mode]\n";
        os << "<eop>";
        NSLog(@"[MNN_BENCH_ERROR] invalid bench command: %s", cmd.c_str());
        return;
    }
    os << "bench backend=" << backend << " prompt=" << prompt_len << " decode=" << decode_len
       << " repeat=" << repeat << " threads=" << threads << " attn_mode=" << attn_mode << "\nreloading model...\n";
    WaitForAppActive();
    // reload with the requested backend so backend_type takes effect
    llm.reset();
    std::string model_dir = GetMainBundleDirectory();
    llm.reset(Llm::createLLM(model_dir + "/config.json"));
    if (!llm) {
        os << "createLLM failed\n";
        os << "<eop>";
        NSLog(@"[MNN_BENCH_ERROR] createLLM failed");
        return;
    }
    llm->set_config("{\"tmp_path\":\"" + GetCleanTmpDirectory() + "\", \"use_mmap\":true}");
    llm->set_config("{\"backend_type\":\"" + backend + "\"}");
    llm->set_config("{\"thread_num\":" + std::to_string(threads) + "}");
    llm->set_config("{\"reuse_kv\":false}");
    llm->set_config("{\"async\":false}");
    // greedy: deterministic benchmark; also avoids sampler sort crash on abnormal logits
    llm->set_config("{\"sampler_type\":\"greedy\"}");
    if (attn_mode >= 0) {
        llm->set_config("{\"attention_mode\":" + std::to_string(attn_mode) + "}");
    }
    llm->load();
    auto context = llm->getContext();
    std::vector<int> tokens(prompt_len, 16);
    double pf_sum = 0, dc_sum = 0;
    int done = 0;
    for (int i = 0; i < repeat + 1; ++i) {
        llm->response(tokens, nullptr, nullptr, decode_len);
        double prefill_s = context->prefill_us / 1e6;
        double decode_s = context->decode_us / 1e6;
        int gen_len = context->gen_seq_len;
        double pf_speed = prefill_s > 0 ? prompt_len / prefill_s : 0;
        double dc_speed = decode_s > 0 ? gen_len / decode_s : 0;
        if (i == 0) {
            os << "warmup: prefill " << std::fixed << std::setprecision(2) << pf_speed
               << " tok/s, decode " << dc_speed << " tok/s\n";
            continue;
        }
        pf_sum += pf_speed;
        dc_sum += dc_speed;
        done++;
        os << "run " << i << ": prefill " << std::fixed << std::setprecision(2) << pf_speed
           << " tok/s, decode " << dc_speed << " tok/s (gen " << gen_len << ")\n";
        NSLog(@"[MNN_BENCH] run=%d backend=%s prompt=%d decode=%d prefill_tok_s=%.2f decode_tok_s=%.2f",
              i, backend.c_str(), prompt_len, gen_len, pf_speed, dc_speed);
    }
    double pf_avg = done > 0 ? pf_sum / done : 0;
    double dc_avg = done > 0 ? dc_sum / done : 0;
    os << "\n#################################\n"
       << "backend = " << backend << ", threads = " << threads << "\n"
       << "prompt len = " << prompt_len << ", decode len = " << decode_len << ", repeat = " << done << "\n"
       << "avg prefill speed = " << std::fixed << std::setprecision(2) << pf_avg << " tok/s\n"
       << "avg  decode speed = " << std::fixed << std::setprecision(2) << dc_avg << " tok/s\n"
       << "##################################\n";
    os << "<eop>";
    NSLog(@"[MNN_BENCH] avg backend=%s threads=%d prompt=%d decode=%d prefill_tok_s=%.2f decode_tok_s=%.2f",
          backend.c_str(), threads, prompt_len, decode_len, pf_avg, dc_avg);
    NSLog(@"[MNN_BENCH_DONE]");
}
@end
