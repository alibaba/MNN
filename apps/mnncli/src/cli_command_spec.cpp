//
//  cli_command_spec.cpp
//
//  Command specifications implementation
//

#include "cli_command_spec.hpp"
#include <sstream>
#include <algorithm>

namespace mnncli {

namespace {
    // Command specifications
    CommandSpec run_spec = {
        "run",
        {
            {"config", 'c', ArgKind::Value, false, "Specify custom config file path"},
            {"prompt", 'p', ArgKind::Value, false, "Provide prompt text directly"},
            {"file", 'f', ArgKind::Value, false, "Read prompts from file"}
        },
        0, 1,  // 0-1 positional args (model_name)
        "Run model inference"
    };

    CommandSpec delete_spec = {
        "delete",
        {},  // No options
        1, 1,  // Exactly 1 positional arg (model_name)
        "Delete a local model"
    };

    CommandSpec list_spec = {
        "list",
        {},  // No options
        0, 0,  // No positional args
        "List all downloaded local models"
    };

    CommandSpec search_spec = {
        "search",
        {},  // No options
        1, 1,  // Exactly 1 positional arg (keyword)
        "Search for models in remote repositories"
    };

    CommandSpec download_spec = {
        "download",
        {},  // No options
        1, 1,  // Exactly 1 positional arg (model_name)
        "Download a model from remote repository"
    };

    CommandSpec model_info_spec = {
        "model_info",
        {},  // No options
        1, 1,  // Exactly 1 positional arg (model_name)
        "Show detailed information about a model"
    };

    CommandSpec serve_spec = {
        "serve",
        {
            {"config", 'c', ArgKind::Value, false, "Specify custom config file path"},
            {"port",   'p', ArgKind::Value, false, "Port number (default: 8000)"},
            {"host",   'H', ArgKind::Value, false, "Host address (default: 127.0.0.1)"}
        },
        0, 1,  // 0-1 positional args (model_name)
        "Start a model serving server"
    };

    CommandSpec benchmark_spec = {
        "benchmark",
        {
            {"model",  'm', ArgKind::Value, false, "Model name or id (optional if positional provided)"},
            {"config", 'c', ArgKind::Value, false, "Specify custom config file path"},
            {"warmup", 'w', ArgKind::Value, false, "Warmup iterations (unused, default: 5)"},
            {"repeat", 'r', ArgKind::Value, false, "Repeat iterations (default: 1)"}
        },
        0, 1,  // 0-1 positional args (model_name)
        "Run performance benchmark on a model"
    };

    CommandSpec config_spec = {
        "config",
        {},  // No options
        0, 3,  // 0-3 positional args (subcommand [key [value]])
        "Manage configuration settings"
    };

    CommandSpec info_spec = {
        "info",
        {},  // No options
        0, 0,  // No positional args
        "Display MNN CLI system information"
    };
}

const CommandSpec& GetSpec(const std::string& command) {
    if (command == "run") return run_spec;
    if (command == "delete") return delete_spec;
    if (command == "list") return list_spec;
    if (command == "search") return search_spec;
    if (command == "download") return download_spec;
    if (command == "model_info") return model_info_spec;
    if (command == "serve") return serve_spec;
    if (command == "benchmark") return benchmark_spec;
    if (command == "config") return config_spec;
    if (command == "info") return info_spec;
    static CommandSpec empty_spec = {};
    return empty_spec;
}

std::string MakeUsage(const CommandSpec& spec) {
    std::ostringstream oss;
    oss << "Usage: mnncli " << spec.name;
    
    // Show positional arguments
    if (spec.max_positional > 0) {
        oss << " [";
        if (spec.max_positional == 1) {
            oss << "model_name";
        } else {
            oss << "model_name ...";
        }
        oss << "]";
    }
    
    // Show options
    if (!spec.options.empty()) {
        oss << " [options]";
    }
    
    oss << "\n\n";
    
    // Help text
    if (!spec.help.empty()) {
        oss << spec.help << "\n\n";
    }
    
    // Options
    if (!spec.options.empty()) {
        oss << "Options:\n";
        for (const auto& opt : spec.options) {
            oss << "  ";
            if (opt.short_name) {
                oss << "-" << opt.short_name << ", ";
            }
            oss << "--" << opt.long_name;
            
            if (opt.kind == ArgKind::Value) {
                oss << " <value>";
            }
            
            if (opt.required) {
                oss << " (required)";
            }
            
            oss << "  " << opt.help << "\n";
        }
    }
    
    return oss.str();
}

} // namespace mnncli
