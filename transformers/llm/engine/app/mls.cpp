//
//  mls.cpp
//
//  Created by MNN on 2023/03/24.
//  Jinde.Song
//  LLM command line tool, based on llm_demo.cpp
//
#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <fstream>
#include <cstdlib>
#include "file_utils.hpp"
#include "remote_model_downloader.hpp"
#include "llm_benchmark.hpp"
#include "mls_server.hpp"

using namespace MNN::Transformer;

static void tuning_prepare(Llm *llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

static std::unique_ptr<Llm> create_and_prepare_llm(const char *config_path) {
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        llm->load();
    }
    if (true)
    {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    return llm;
}

int list_local_models(const std::string &directory_path, std::vector<std::string> &model_names, bool sort = true) {
    std::error_code ec;
    if (!fs::exists(directory_path, ec)) {
        return 1;
    }
    if (!fs::is_directory(directory_path, ec)) {
        return 1;
    }
    for (const auto &entry : fs::directory_iterator(directory_path, ec)) {
        if (ec) {
            return 1;
        }
        if (fs::is_symlink(entry, ec)) {
            if (ec) {
                return 1;
            }
            std::string file_name = entry.path().filename().string();
            model_names.emplace_back(file_name);
        }
    }
    if (sort) {
        std::sort(model_names.begin(), model_names.end());
    }
    return 0;
}

static int eval_prompts(Llm *llm, const std::vector<std::string> &prompts) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    // llm->warmup();
    for (int i = 0; i < prompts.size(); i++)
    {
        const auto &prompt = prompts[i];
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#")
        {
            continue;
        }
        llm->response(prompt);
    }
    float vision_s = vision_time / 1e6;
    float audio_s = audio_time / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num = %d\n", prompt_len);
    printf("decode tokens num = %d\n", decode_len);
    printf(" vision time = %.2f s\n", vision_s);
    printf("  audio time = %.2f s\n", audio_s);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
    return 0;
}

static int eval_file(Llm *llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt))
    {
        if (prompt.back() == '\r')
        {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    if (prompts.empty())
    {
        return 1;
    }
    return eval_prompts(llm, prompts);
}

static int print_usage() {
    std::cout << "Available Commands:" << std::endl;
    std::cout << "  mls list: list all local model names" << std::endl;
    std::cout << "  mls search keyword: search all available remote models by key" << std::endl;
    std::cout << "  mls download model_name : download the model" << std::endl;
    std::cout << "  mls run  model_name : download the model" << std::endl;
    std::cout << "  mls benchmark:  model_name test benchmark of a model" << std::endl;
    std::cout << "  mls serve: serve with openai compatible api" << std::endl;
    std::cout << "  mls delete model_name: remove the download model" << std::endl;
    return 0;
}

// list files int the directory of ~/.cache/modelscope/hub/MNN/Qwen-7B-Chat-MNN/
static int list_models(int argc, const char *argv[]) {
    std::vector<std::string> model_names;
    list_local_models(mls::FileUtils::GetBaseCacheDir(), model_names);
    if (!model_names.empty()) {
        for (auto &name : model_names)
        {
            printf("%s\n", name.c_str());
        }
    } else {
        printf("no local models; use \'mls search\' to search remote models and download\n");
    }
    return 0;
}

static int serve(int argc, const char *argv[]) {
    bool invalid_param{false};
    std::string config_path{};
    std::string arg{};
    if (argc < 3) {
        print_usage();
        return 1;
    }
    arg = argv[2];
    if (arg.find('-') != 0) {
        config_path = (fs::path(mls::FileUtils::GetBaseCacheDir()) / arg / "config.json").string();
    }
    for (int i = 2; i < argc; i++) {
        arg = argv[i];
        if (arg == "-c") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            config_path = mls::FileUtils::ExpandTilde(argv[i]);
        }
    }
    mls::MlsServer server;
    auto llm = create_and_prepare_llm(config_path.c_str());
    server.Start(llm.get());
    return 0;
}

static int benchmark(int argc, const char *argv[]) {
    std::string arg{};
    bool invalid_param{false};
    std::string config_path{};
    if (argc < 3)
    {
        print_usage();
        return 1;
    }
    arg = argv[2];
    if (arg.find('-') != 0)
    {
        config_path = mls::FileUtils::GetConfigPath(arg);
    }
    for (int i = 2; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "-c")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            config_path = argv[i];
        }
    }
    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage();
        exit(1);
    }
    if (config_path.empty())
    {
        fprintf(stderr, "error: config path is empty\n");
        print_usage();
        exit(1);
    }
    auto llm = create_and_prepare_llm(config_path.c_str());
    mls::LLMBenchmark benchmark;
    benchmark.Start(llm.get(), {});
    return 0;
}

static int run(int argc, const char *argv[]) {
    std::cout << "Start run..." << std::endl;
    std::string arg{};
    std::string config_path{};
    std::string prompt;
    std::string prompt_file;
    bool invalid_param = false;
    if (argc < 3)
    {
        print_usage();
        return 1;
    }
    arg = argv[2];
    if (arg.find('-') != 0)
    {
        config_path = mls::FileUtils::GetConfigPath(arg);
    }
    for (int i = 2; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "-c")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            config_path = mls::FileUtils::ExpandTilde(argv[i]);
        }
        else if (arg == "-p")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            prompt = argv[i];
        }
        else if (arg == "-pf")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            prompt_file = argv[i];
        }
    }
    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage();
        exit(1);
    }
    if (config_path.empty())
    {
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
    if (true)
    {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    if (prompt.empty() && prompt_file.empty())
    {
        llm->chat();
    }
    else if (!prompt.empty())
    {
        eval_prompts(llm.get(), {prompt});
    }
    else
    {
        eval_file(llm.get(), prompt_file);
    }
    return 0;
}

int download(int argc, const char *argv[]) {
    if (argc < 3)
    {
        print_usage();
        return 1;
    }
    std::string repo_name = argv[2];
    std::cout << "download repo: " << repo_name << std::endl;
    mls::HfApiClient api_client;
    std::string error_info;
    if (repo_name.find("taobao-mnn/") != 0)
    {
        repo_name = "taobao-mnn/" + repo_name;
    }
    const auto repo_info = api_client.GetRepoInfo(repo_name, "main", error_info);
    if (!error_info.empty())
    {
        std::cout << "get repo info error: " << error_info << std::endl;
        return 1;
    }
    api_client.DownloadRepo(repo_info);
    return 0;
}

int search(int argc, const char *argv[]) {
    if (argc < 3)
    {
        print_usage();
        return 1;
    }
    const std::string key = argv[2];
    mls::HfApiClient client = mls::HfApiClient();
    auto repos = std::move(client.SearchRepos(key));
    for (auto &repo : repos)
    {
        auto pos = repo.model_id.rfind('/');
        if (pos != std::string::npos)
        {
            printf("%s\n", repo.model_id.substr(pos + 1).c_str());
        }
    }
    return 0;
}

int delete_model(int argc, const char *argv[]) {
    if (argc < 3)
    {
        print_usage();
        return 1;
    }
    std::string model_name = argv[2];
    std::string linker_path = mls::FileUtils::GetFolderLinkerPath(model_name);
    mls::FileUtils::RemoveFileIfExists(linker_path);
    if (model_name.find("taobao-mnn") != 0)
    {
        model_name = "taobao-mnn/" + model_name;
    }
    std::string storage_path = mls::FileUtils::GetStorageFolderPath(model_name);
    mls::FileUtils::RemoveFileIfExists(storage_path);
    return 0;
}

int main(int argc, const char *argv[]) {
    std::string cmd = argv[1];
    if (cmd == "list") {
        list_models(argc, argv);
    }
    else if (cmd == "serve") {
        serve(argc, argv);
    }
    else if (cmd == "run") {
        run(argc, argv);
    }
    else if (cmd == "benchmark") {
        benchmark(argc, argv);
    }
    else if (cmd == "download") {
        download(argc, argv);
    }
    else if (cmd == "search") {
        search(argc, argv);
    }
    else if (cmd == "delete") {
        delete_model(argc, argv);
    }
    else {
        print_usage();
    }
    return 0;
}
