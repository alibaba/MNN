#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  run_command_handler.cpp
//
//  Handler for 'run' command
//

#include "handlers/run_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "cli_config_manager.hpp"
#include "file_utils.hpp"
#include "llm_manager.hpp"
#include "log_utils.hpp"
#include "model_runner.hpp"
#include "user_interface.hpp"
#include <llm/llm.hpp>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace mnncli {
using namespace mnn::downloader;
namespace {
bool TryParseThinkingOption(const std::string& value, bool& thinking_enabled) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower == "1" || lower == "true" || lower == "on" || lower == "yes" || lower == "enable" || lower == "enabled") {
        thinking_enabled = true;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "off" || lower == "no" || lower == "disable" || lower == "disabled") {
        thinking_enabled = false;
        return true;
    }
    return false;
}
} // namespace

std::string RunCommandHandler::CommandName() const {
    return "run";
}

const CommandSpec& RunCommandHandler::GetSpec() const {
    return mnncli::GetSpec("run");
}

int RunCommandHandler::Handle(const ParsedCommand& cmd) {
    std::string model_name;
    std::string config_path;
    std::string prompt;
    std::string prompt_file;
    std::string thinking_value;
    bool has_thinking_override = false;
    bool thinking_enabled = true;
    
    // Extract options from parsed command
    prompt = CommandParser::GetOption(cmd, "prompt", "");
    prompt_file = CommandParser::GetOption(cmd, "file", "");
    config_path = CommandParser::GetOption(cmd, "config", "");
    thinking_value = CommandParser::GetOption(cmd, "thinking", "");
    has_thinking_override = !thinking_value.empty();
    if (has_thinking_override && !TryParseThinkingOption(thinking_value, thinking_enabled)) {
        UserInterface::ShowError("Invalid thinking option value",
                                 "Use --thinking true|false (also supports 1|0, on|off, yes|no)");
        PrintRunUsage();
        return 1;
    }
    
    // Determine model name
    if (!cmd.arguments.empty()) {
        model_name = cmd.arguments[0];
    }
    
    LOG_INFO("Parsed arguments. model_name='" + model_name + "', config_path='" + config_path + "', prompt='" + prompt + "', prompt_file='" + prompt_file + "'");
    // If no config path specified, check if we have a model name or can use default model
    if (config_path.empty()) {
        if (model_name.empty()) {
            // No model name provided, try to use default model
            auto& config_mgr = ConfigManager::GetInstance();
            auto config = config_mgr.LoadConfig();
            if (!config.default_model.empty()) {
                model_name = config.default_model;
                LOG_INFO("Using default model: " + model_name);
            } else {
                UserInterface::ShowError("Model name required and no default model set",
                                        "Set a default model with: mnncli config set default_model <model_name>");
                PrintRunUsage();
                return 1;
            }
        }
        config_path = FileUtils::GetConfigPath(model_name);
    } else {
        // If config path is specified, extract model name from path or use a default
        std::string config_dir = fs::path(config_path).parent_path().string();
        std::string config_filename = fs::path(config_dir).filename().string();
        if (!config_filename.empty()) {
            model_name = config_filename;
        } else {
            model_name = "custom_model";
        }
        // Expand ~ in config path
        config_path = FileUtils::ExpandTilde(config_path);
    }
    
    if (config_path.empty()) {
        UserInterface::ShowError("Config path is empty", "Unable to determine config path for model: " + model_name);
        PrintRunUsage();
        return 1;
    }
    
    LOG_INFO("Starting model: " + model_name);
    LOG_INFO("Config path: " + config_path);
    
    auto llm = LLMManager::CreateLLM(config_path, true);
    if (has_thinking_override) {
        std::string thinking_cfg = std::string("{\"jinja\":{\"context\":{\"enable_thinking\":") +
                                   (thinking_enabled ? "true" : "false") + "}}}";
        llm->set_config(thinking_cfg);
        LOG_INFO("Applied thinking mode override: " + std::string(thinking_enabled ? "enabled" : "disabled"));
    }
    
    if (prompt.empty() && prompt_file.empty()) {
        ::ModelRunner runner(llm.get());
        runner.InteractiveChat();
    } else if (!prompt.empty()) {
        ::ModelRunner runner(llm.get());
        runner.EvalPrompts({prompt});
    } else {
        ::ModelRunner runner(llm.get());
        runner.EvalFile(prompt_file);
    }
    
    return 0;
}

void RunCommandHandler::PrintRunUsage() const {
    std::cout << MakeUsage(mnncli::GetSpec("run")) << "\n";
}

} // namespace mnncli
