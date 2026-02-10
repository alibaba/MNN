#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  benchmark_command_handler.cpp
//
//  Handler for 'benchmark' command
//

#include "handlers/benchmark_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "cli_config_manager.hpp"
#include "file_utils.hpp"
#include "llm_manager.hpp"
#include "llm_benchmark.hpp"
#include "log_utils.hpp"
#include "user_interface.hpp"
#include <filesystem>
#include <algorithm>

namespace mnncli {
using namespace mnn::downloader;

std::string BenchmarkCommandHandler::CommandName() const {
    return "benchmark";
}

const CommandSpec& BenchmarkCommandHandler::GetSpec() const {
    return mnncli::GetSpec("benchmark");
}

int BenchmarkCommandHandler::Handle(const ParsedCommand& cmd) {
    std::string model_name;
    std::string config_path;

    // Options
    model_name = CommandParser::GetOption(cmd, "model", "");
    config_path = CommandParser::GetOption(cmd, "config", "");
    auto repeat_str = CommandParser::GetOption(cmd, "repeat", "");
    auto warmup_str = CommandParser::GetOption(cmd, "warmup", ""); // currently unused

    // If positional provided, prefer it unless --model set
    if (model_name.empty() && !cmd.arguments.empty()) {
        model_name = cmd.arguments[0];
    }

    // Load application config
    auto& config_mgr = ConfigManager::GetInstance();
    auto app_config = config_mgr.LoadConfig();

    // Resolve config path similar to 'run'
    if (config_path.empty()) {
        if (model_name.empty()) {
            if (!app_config.default_model.empty()) {
                model_name = app_config.default_model;
                LOG_INFO("Using default model: " + model_name);
            } else {
                UserInterface::ShowError("Model name required and no default model set",
                                         "Set a default model with: mnncli config set default_model <model_name>");
                std::cout << MakeUsage(mnncli::GetSpec("benchmark")) << "\n";
                return 1;
            }
        }
        config_path = FileUtils::GetConfigPath(model_name);
    } else {
        // Expand path and best-effort derive model name
        config_path = FileUtils::ExpandTilde(config_path);
        std::string config_dir = std::filesystem::path(config_path).parent_path().string();
        std::string config_filename = std::filesystem::path(config_dir).filename().string();
        if (!config_filename.empty()) {
            model_name = config_filename;
        } else if (model_name.empty()) {
            model_name = "custom_model";
        }
    }

    if (config_path.empty()) {
        UserInterface::ShowError("Config path is empty", "Unable to determine config path for model: " + model_name);
        std::cout << MakeUsage(mnncli::GetSpec("benchmark")) << "\n";
        return 1;
    }

    LOG_INFO("Benchmarking model: " + model_name);
    LOG_INFO("Config path: " + config_path);

    // Create and load model
    auto llm = LLMManager::CreateLLM(config_path, true);

    // Prepare benchmark options
    LLMBenchMarkOptions options{ /*progress*/ true, /*reps*/ 1 };
    if (!repeat_str.empty()) {
        try { options.reps = std::stoi(repeat_str); } catch (...) { /* keep default */ }
    }
    // warmup_str parsed but not used by current LLMBenchmark interface
    (void)warmup_str;

    // Run benchmark
    LLMBenchmark bm;
    bm.Start(llm.get(), options);

    return 0;
}

} // namespace mnncli
