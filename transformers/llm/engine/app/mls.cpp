//
//  mls.cpp
//
//  Created by MNN on 2023/03/24.
//  Jinde.Song
//  LLM command line tool, based on llm_demo.cpp
//
#include "llm/llm.hpp"
#include "evaluation/dataset.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <initializer_list>
#include "downloader/remote_model_downloader.hpp"
#include "llm_benchmark.hpp"

using namespace MNN::Transformer;


static void trace_prepare(Llm* llm) {
    MNN_PRINT("Prepare for resize opt Begin\n");
    llm->trace(true);
    std::ostringstream cacheOs;
    llm->generate(std::initializer_list<int>{200, 200}, &cacheOs, "");
    MNN_PRINT("Prepare for resize opt End\n");
    llm->trace(false);
    llm->reset();
}

static void tuning_prepare(Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}


static std::unique_ptr<Llm> create_and_prepare_llm(const char* config_path) {
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        llm->load();
    }
    if (true) {
        AUTOTIME;
        trace_prepare(llm.get());
    }
    if (true) {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    return llm;
}

static int eval_prompts(Llm* llm, const std::vector<std::string>& prompts) {
    for (int i = 0; i < prompts.size(); i++) {
        const auto& prompt = prompts[i];
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        llm->response(prompt);
    }
    printf("\n#################################\n");
    printf("prompt tokens num  = %d\n", llm->getTotalPromptLen());
    printf("decode tokens num  = %d\n", llm->getTotalDecodeLen());
    printf("prefill time = %.2f s\n", llm->getTotalPrefillTime());
    printf(" decode time = %.2f s\n", llm->getTotalDecodeTime());
    printf("prefill speed = %.2f tok/s\n", llm->average_prefill_speed());
    printf(" decode speed = %.2f tok/s\n", llm->average_decode_speed());
    printf("##################################\n");
    return 0;
}

static int eval_csv(Llm* llm, const std::vector<std::string>& lines, std::string filename) {
    auto csv_data = parse_csv(lines);
    int right = 0, wrong = 0;
    std::vector<std::string> answers;
    for (int i = 1; i < csv_data.size(); i++) {
        const auto& elements = csv_data[i];
        std::string prompt = elements[1];
        prompt += "\n\nA. " + elements[2];
        prompt += "\nB. " + elements[3];
        prompt += "\nC. " + elements[4];
        prompt += "\nD. " + elements[5];
        prompt += "\n\n";
        printf("%s", prompt.c_str());
        printf("## 进度: %d / %lu\n", i, lines.size() - 1);
        auto res = llm->response(prompt.c_str());
        answers.push_back(res);
    }
    {
        auto position = filename.rfind("/");
        if (position != std::string::npos) {
            filename = filename.substr(position + 1, -1);
        }
        position = filename.find("_val");
        if (position != std::string::npos) {
            filename.replace(position, 4, "_res");
        }
        std::cout << "store to " << filename << std::endl;
    }
    std::ofstream ofp(filename);
    ofp << "id,answer" << std::endl;
    for (int i = 0; i < answers.size(); i++) {
        auto& answer = answers[i];
        ofp << i << ",\""<< answer << "\"" << std::endl;
    }
    ofp.close();
    return 0;
}

static int eval_file(Llm* llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    // eval_csv
    if (prompts[0] == "id,question,A,B,C,D,answer") {
        return eval_csv(llm, prompts, prompt_file);
    }
    return eval_prompts(llm, prompts);
}


static int print_usage() {
    std::cout << "Usage: " << std::endl;
    std::cout << "  ./mls list: list all supported models" << std::endl;
    return 0;
}

//list files int the directory of ~/.cache/modelscope/hub/MNN/Qwen-7B-Chat-MNN/
static int list_models() {
    std::cout << "List all supported models:" << std::endl;
    std::string cmd = "ls -l ~/.cache/modelscope/hub/MNN/";
    std::system(cmd.c_str());
    return 0;
}

static int serve(int argc, const char* argv[]) {
    std::cout << "Start serving..." << std::endl;
    return 0;
}

static int benchmark(int argc, const char* argv[]) {
    std::cout << "Start benchmark..." << std::endl;
    std::string arg{};
    std::string config_path{};
    bool invalid_param{false};
    for (int i = 2; i < argc; i++) {
        arg = argv[i];
        if (arg == "-c") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            config_path = argv[i];
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage();
        exit(1);
    }
    if (!config_path.empty()) {
        auto llm = create_and_prepare_llm(config_path.c_str());
        mls::LLMBenchmark benchmark;
        benchmark.Start(llm.get(), {});
    }
    return 0;
}

static int run(int argc, const char* argv[]) {
    std::cout << "Start run..." << std::endl;
    std::string arg{};
    std::string config_path{};
    std::string prompt;
    std::string prompt_file;
    bool invalid_param = false;
    for (int i = 2; i < argc; i++) {
        arg = argv[i];
        if (arg == "-c") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            config_path = argv[i];
        } else if (arg == "-p") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            prompt = argv[i];
        } else if (arg == "-pf") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            prompt_file = argv[i];
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage();
        exit(1);
    }
    if (config_path.empty()) {
        fprintf(stderr, "error: config path is empty\n");
        print_usage();
        exit(1);
    }

    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        llm->load();
    }
    if (true) {
        AUTOTIME;
        trace_prepare(llm.get());
    }
    if (true) {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    if (prompt.empty() && prompt_file.empty()) {
        llm->chat();
    } else if (!prompt.empty()) {
        eval_prompts(llm.get(), {prompt});
    } else {
        eval_file(llm.get(), prompt_file);
    }
    return 0;
}

int download(int argc, const char* argv[]) {
    if (argc < 3) {
        print_usage();
        return 1;
    }
    const std::string repo_name = argv[2];
    std::cout<<"download repo: "<<repo_name<<std::endl;
    mls::RemoteModelDownloader model_downloader{};
    std::string error_info;
    const auto repo_info = model_downloader.getRepoInfo(repo_name, "main", "", error_info);
    if (!error_info.empty()) {
        std::cout << "get repo info error: "<< error_info << std::endl;
        return 1;
    }
    std::cout << "repo_info sha " <<repo_info.sha << std::endl;
    // model_downloader.DownloadFromHF(repo_name, repo_info.sha, "embeddings_bf16.bin");
    // model_downloader.DownloadFromHF(repo_name, repo_info.sha, "config.json");
    for (auto & sub_file :  repo_info.siblings) {
        model_downloader.DownloadFromHF(repo_name, repo_info.sha, sub_file);
    }
    return 0;
}

int main(int argc, const char* argv[]) {
    std::cout<<"argc:"<<argc<<std::endl;
    if (argc < 2) {
        print_usage();
        return 1;
    }
    std::string cmd = argv[1];
    std::cout<<"cmd:"<<cmd<<std::endl;
    if (cmd == "list") {
        list_models();
    } else if (cmd == "serve") {
        serve(argc, argv);
    } else if (cmd == "run") {
        run(argc, argv);
    } else if (cmd == "benchmark") {
        benchmark(argc, argv);
    } else if (cmd == "download") {
        download(argc, argv);
    } else {
        print_usage();
    }
    return 0;
}
