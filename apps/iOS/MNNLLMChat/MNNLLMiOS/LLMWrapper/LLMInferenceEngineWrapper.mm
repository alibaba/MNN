//
//  LLMInferenceEngineWrapper.m
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <functional>
#include <MNN/llm/llm.hpp>
#include <vector>
#include <utility>

#import <Foundation/Foundation.h>
#import "LLMInferenceEngineWrapper.h"

using namespace MNN::Transformer;

using ChatMessage = std::pair<std::string, std::string>;
static std::vector<ChatMessage> history{};

@implementation LLMInferenceEngineWrapper {
    std::shared_ptr<Llm> llm;
}

- (instancetype)initWithModelPath:(NSString *)modelPath completion:(CompletionHandler)completion {
    self = [super init];
    if (self) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            BOOL success = [self loadModelFromPath:modelPath];
            // MARK: Test Local Model
            // BOOL success = [self loadModel];

            dispatch_async(dispatch_get_main_queue(), ^{
                completion(success);
            });
        });
    }
    return self;
}


bool remove_directory(const std::string& path) {
    try {
        std::filesystem::remove_all(path); // 删除目录及其内容
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error removing directory: " << e.what() << std::endl;
        return false;
    }
}

- (BOOL)loadModel {
    if (!llm) {
        NSString *bundleDirectory = [[NSBundle mainBundle] bundlePath];
        std::string model_dir = [bundleDirectory UTF8String];
        std::string config_path = model_dir + "/config.json";
        llm.reset(Llm::createLLM(config_path));
        NSString *tempDirectory = NSTemporaryDirectory();
        llm->set_config("{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\", \"use_mmap\":true}");
        llm->load();
    }
    return YES;
}

- (BOOL)loadModelFromPath:(NSString *)modelPath {
    if (!llm) {
        std::string config_path = std::string([modelPath UTF8String]) + "/config.json";

        // Read the config file to get use_mmap value
        NSError *error = nil;
        NSData *configData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:config_path.c_str()]];
        NSDictionary *configDict = [NSJSONSerialization JSONObjectWithData:configData options:0 error:&error];
        // If use_mmap key doesn't exist, default to YES
        BOOL useMmap = configDict[@"use_mmap"] == nil ? YES : [configDict[@"use_mmap"] boolValue];

        llm.reset(Llm::createLLM(config_path));
        if (!llm) {
            return NO;
        }

        // Create temp directory inside the modelPath folder
        std::string model_path_str([modelPath UTF8String]);
        std::string temp_directory_path = model_path_str + "/temp";

        struct stat info;
        if (stat(temp_directory_path.c_str(), &info) == 0) {
            // Directory exists, so remove it
            if (!remove_directory(temp_directory_path)) {
                std::cerr << "Failed to remove existing temp directory: " << temp_directory_path << std::endl;
                return NO;
            }
            std::cerr << "Existing temp directory removed: " << temp_directory_path << std::endl;
        }

        // Now create the temp directory
        if (mkdir(temp_directory_path.c_str(), 0777) != 0) {
            std::cerr << "Failed to create temp directory: " << temp_directory_path << std::endl;
            return NO;
        }
        std::cerr << "Temp directory created: " << temp_directory_path << std::endl;

        // NSLog(@"useMmap value: %@", useMmap ? @"YES" : @"NO");

        // Explicitly convert BOOL to bool and ensure proper string conversion
        bool useMmapCpp = (useMmap == YES);
        std::string configStr = "{\"tmp_path\":\"" + temp_directory_path + "\", \"use_mmap\":" + (useMmapCpp ? "true" : "false") + "}";
        // Debug print to check the final config string
        // NSLog(@"Config string: %s", configStr.c_str());

        llm->set_config(configStr);

        llm->load();
    }
    else {
        std::cerr << "Warmming:: LLM have already been created!" << std::endl;
    }
    return YES;
}

// llm stream buffer with callback
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;
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

- (void)processInput:(NSString *)input withOutput:(OutputHandler)output {
    if (llm == nil) {
        output(@"Error: Model not loaded");
        return;
    }

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0), ^{

        LlmStreamBuffer::CallBack callback = [output](const char* str, size_t len) {
            if (output) {
                NSString *nsOutput = [[NSString alloc] initWithBytes:str
                                                                length:len
                                                              encoding:NSUTF8StringEncoding];
                if (nsOutput) {
                    output(nsOutput);
                }
            }
        };

        LlmStreamBuffer streambuf(callback);
        std::ostream os(&streambuf);

        history.emplace_back(ChatMessage("user", [input UTF8String]));

        if (std::string([input UTF8String]) == "benchmark") {
            [self performBenchmarkWithOutput:&os];
        } else {
            llm->response(history, &os, "<eop>", 999999);
        }

    });
}

// New method to handle benchmarking
- (void)performBenchmarkWithOutput:(std::ostream *)os {
    std::string model_dir = [[[NSBundle mainBundle] bundlePath] UTF8String];
    std::string prompt_file = model_dir + "/bench.txt";
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
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
    for (const auto& p : prompts) {
        llm->response(p, os, "\n");
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
    }

    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;

    *os << "\n#################################\n"
        << "prompt tokens num  = " << prompt_len << "\n"
        << "decode tokens num  = " << decode_len << "\n"
        << "prefill time = " << std::fixed << std::setprecision(2) << prefill_s << " s\n"
        << "decode time = " << std::fixed << std::setprecision(2) << decode_s << " s\n"
        << "prefill speed = " << std::fixed << std::setprecision(2) << (prefill_s > 0 ? prompt_len / prefill_s : 0) << " tok/s\n"
        << "decode speed = " << std::fixed << std::setprecision(2) << (decode_s > 0 ? decode_len / decode_s : 0) << " tok/s\n"
        << "##################################\n";
    *os << "<eop>";
}

- (void)dealloc {
    std::cerr << "llm dealloc reset" << std::endl;
    history.clear();
    llm.reset();
    llm = nil;
}

- (void)init:(const std::vector<std::string>&)chatHistory {
    history.clear();
    history.emplace_back("system", "You are a helpful assistant.");

    for (size_t i = 0; i < chatHistory.size(); ++i) {
        history.emplace_back(i % 2 == 0 ? "user" : "assistant", chatHistory[i]);
    }
}

- (void)addPromptsFromArray:(NSArray<NSDictionary *> *)array {

    history.clear();

    for (NSDictionary *dict in array) {
        [self addPromptsFromDictionary:dict];
    }
}

- (void)addPromptsFromDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        NSString *value = dictionary[key];

        std::string keyString = [key UTF8String];
        std::string valueString = [value UTF8String];

        history.emplace_back(ChatMessage(keyString, valueString));
    }
}

@end
