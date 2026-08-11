#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  serve_command_handler.cpp
//
//  Handler for 'serve' command
//

#include "handlers/serve_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "cli_config_manager.hpp"
#include "file_utils.hpp"
#include "llm_manager.hpp"
#include "log_utils.hpp"
#include "mnncli_server.hpp"
#include "user_interface.hpp"
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

std::string ServeCommandHandler::CommandName() const {
    return "serve";
}

const CommandSpec& ServeCommandHandler::GetSpec() const {
    return mnncli::GetSpec("serve");
}

int ServeCommandHandler::Handle(const ParsedCommand& cmd) {
    std::string model_name;
    std::string config_path;
    std::string host;
    int port = -1;
    bool has_thinking_override = false;
    bool thinking_enabled = true;

    // Extract options
    host = CommandParser::GetOption(cmd, "host", "");
    auto port_str = CommandParser::GetOption(cmd, "port", "");
    config_path = CommandParser::GetOption(cmd, "config", "");
    auto thinking_value = CommandParser::GetOption(cmd, "thinking", "");
    has_thinking_override = !thinking_value.empty();
    if (has_thinking_override && !TryParseThinkingOption(thinking_value, thinking_enabled)) {
        UserInterface::ShowError("Invalid thinking option value",
                                 "Use --thinking true|false (also supports 1|0, on|off, yes|no)");
        std::cout << MakeUsage(mnncli::GetSpec("serve")) << "\n";
        return 1;
    }

    // Determine model name (if provided as positional)
    if (!cmd.arguments.empty()) {
        model_name = cmd.arguments[0];
    }

    // Load config for defaults
    auto& config_mgr = ConfigManager::GetInstance();
    auto app_config = config_mgr.LoadConfig();

    if (host.empty()) {
        host = app_config.api_host.empty() ? std::string("127.0.0.1") : app_config.api_host;
    }
    if (!port_str.empty()) {
        try { port = std::stoi(port_str); } catch (...) { port = -1; }
    }
    if (port <= 0) {
        port = app_config.api_port > 0 ? app_config.api_port : 8000;
    }

    // Resolve config path
    if (config_path.empty()) {
        if (model_name.empty()) {
            if (!app_config.default_model.empty()) {
                model_name = app_config.default_model;
                LOG_INFO("Using default model: " + model_name);
            } else {
                UserInterface::ShowError("Model name required and no default model set",
                                        "Set a default model with: mnncli config set default_model <model_name>");
                std::cout << MakeUsage(mnncli::GetSpec("serve")) << "\n";
                return 1;
            }
        }
        config_path = FileUtils::GetConfigPath(model_name);
    } else {
        // Expand ~ in config path and best-effort derive model name
        config_path = FileUtils::ExpandTilde(config_path);
        std::string config_dir = fs::path(config_path).parent_path().string();
        std::string config_filename = fs::path(config_dir).filename().string();
        if (!config_filename.empty()) {
            model_name = config_filename;
        } else if (model_name.empty()) {
            model_name = "custom_model";
        }
    }

    if (config_path.empty()) {
        UserInterface::ShowError("Config path is empty", "Unable to determine config path for model: " + model_name);
        std::cout << MakeUsage(mnncli::GetSpec("serve")) << "\n";
        return 1;
    }

    LOG_INFO("Starting server for model: " + model_name);
    LOG_INFO("Config path: " + config_path);
    LOG_INFO("Bind: http://" + host + ":" + std::to_string(port));

    // Create and load model
    auto llm = LLMManager::CreateLLM(config_path, true);
    if (has_thinking_override) {
        std::string thinking_cfg = std::string("{\"jinja\":{\"context\":{\"enable_thinking\":") +
                                   (thinking_enabled ? "true" : "false") + "}}}";
        llm->set_config(thinking_cfg);
        LOG_INFO("Applied thinking mode override: " + std::string(thinking_enabled ? "enabled" : "disabled"));
    }

    // Determine if this is an R1 model (affects prompt formatting)
    auto lower_path = config_path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
    bool is_r1 = (lower_path.find("deepseek-r1") != std::string::npos);

    // Start HTTP server (blocking call)
    MnncliServer server;
    server.Start(llm.get(), is_r1, host, port);

    return 0;
}

} // namespace mnncli
