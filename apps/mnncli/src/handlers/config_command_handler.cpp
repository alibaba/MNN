#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  config_command_handler.cpp
//
//  Handler for 'config' command
//

#include "handlers/config_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "cli_config_manager.hpp"
#include "user_interface.hpp"
#include <cctype>

namespace mnncli {
using namespace mnn::downloader;

std::string ConfigCommandHandler::CommandName() const {
    return "config";
}

const CommandSpec& ConfigCommandHandler::GetSpec() const {
    return mnncli::GetSpec("config");
}

int ConfigCommandHandler::Handle(const ParsedCommand& cmd) {
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    
    // If no arguments, show all config
    if (cmd.arguments.empty()) {
        auto cfg = config_mgr.LoadConfig();
        config_mgr.ShowConfig(cfg);
        return 0;
    }
    
    std::string subcommand = cmd.arguments[0];
    
    if (subcommand == "show") {
        auto cfg = config_mgr.LoadConfig();
        config_mgr.ShowConfig(cfg);
        return 0;
    } else if (subcommand == "get") {
        if (cmd.arguments.size() < 2) {
            mnncli::UserInterface::ShowError("Config key required", "Usage: mnncli config get <key>");
            return 1;
        }
        std::string key = cmd.arguments[1];
        auto cfg = config_mgr.LoadConfig();
        
        if (key == "default_model") {
            std::cout << "default_model = " << cfg.default_model << "\n";
        } else if (key == "download_provider") {
            std::cout << "download_provider = " << cfg.download_provider << "\n";
        } else if (key == "model_cache_dir") {
            std::cout << "model_cache_dir = " << cfg.cache_dir << "\n";
        } else {
            mnncli::UserInterface::ShowError("Unknown config key: " + key);
            return 1;
        }
        return 0;
    } else if (subcommand == "set") {
        if (cmd.arguments.size() < 3) {
            mnncli::UserInterface::ShowError("Usage: mnncli config set <key> <value>");
            return 1;
        }
        std::string key = cmd.arguments[1];
        std::string value = cmd.arguments[2];
        
        auto cfg = config_mgr.LoadConfig();
        if (key == "default_model") {
            cfg.default_model = value;
        } else if (key == "download_provider") {
            cfg.download_provider = value;
        } else if (key == "cache_dir") {
            cfg.cache_dir = value;
        } else {
            mnncli::UserInterface::ShowError("Unknown config key: " + key);
            return 1;
        }
        
        config_mgr.SaveConfig(cfg);
        std::cout << "Config updated: " << key << " = " << value << "\n";
        return 0;
    } else if (subcommand == "reset") {
        // Reset all configs to default without confirmation
        if (config_mgr.ResetConfig()) {
            std::cout << "Configuration reset to default settings.\n";
            return 0;
        } else {
            mnncli::UserInterface::ShowError("Failed to reset configuration");
            return 1;
        }
    } else {
        mnncli::UserInterface::ShowError("Unknown subcommand: " + subcommand);
        return 1;
    }
}

} // namespace mnncli
