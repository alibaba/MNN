//
//  cli_command_parser.cpp
//
//  Command-line argument parsing implementation
//

#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include <stdexcept>
#include <algorithm>

namespace mnncli {

CommandParser::CommandParser(int argc, const char* argv[]) 
    : argc_(argc), argv_(argv) {
}

void CommandParser::SetCommandSpec(const CommandSpec& spec) {
    spec_ = &spec;
}

bool CommandParser::IsShortOption(const std::string& arg) {
    return arg.size() >= 2 && arg[0] == '-' && arg[1] != '-';
}

bool CommandParser::IsLongOption(const std::string& arg) {
    return arg.size() >= 3 && arg[0] == '-' && arg[1] == '-';
}

void CommandParser::ParseGlobalOptions() {
    for (int i = 1; i < argc_; i++) {
        std::string arg = argv_[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose_ = true;
            break;
        }
    }
}

ParsedCommand CommandParser::Parse() {
    ParseGlobalOptions();
    
    ParsedCommand cmd;
    cmd.verbose = verbose_;
    
    if (argc_ < 2) {
        return cmd;
    }
    
    cmd.command = argv_[1];
    
    if (!spec_) {
        // No spec, just collect all args as positional
        for (int i = 2; i < argc_; i++) {
            std::string arg = argv_[i];
            if (!IsShortOption(arg) && !IsLongOption(arg)) {
                cmd.arguments.push_back(arg);
            }
        }
        return cmd;
    }
    
    ParseCommandOptions(cmd);
    
    if (spec_) {
        ValidateParsedCommand(cmd);
    }
    
    return cmd;
}

void CommandParser::ParseCommandOptions(ParsedCommand& cmd) {
    for (int i = 2; i < argc_; i++) {
        std::string arg = argv_[i];
        
        // Skip verbose flag (already processed)
        if (arg == "-v" || arg == "--verbose") {
            continue;
        }
        
        // Handle help flag
        if (arg == "-h" || arg == "--help") {
            throw std::runtime_error("help_requested");
        }
        
        // Handle short options
        if (IsShortOption(arg)) {
            char short_name = arg[1];
            auto it = std::find_if(spec_->options.begin(), spec_->options.end(),
                [short_name](const OptionSpec& opt) { return opt.short_name == short_name; });
            
            if (it != spec_->options.end()) {
                if (it->kind == ArgKind::Value) {
                    if (i + 1 >= argc_) {
                        throw std::runtime_error("Missing value for option: " + std::string(1, short_name));
                    }
                    cmd.options[it->long_name] = argv_[++i];
                } else {
                    cmd.options[it->long_name] = "true";
                }
            } else {
                // Unknown short option
                cmd.arguments.push_back(arg);
            }
        }
        // Handle long options
        else if (IsLongOption(arg)) {
            std::string long_name = arg.substr(2); // Remove "--"
            auto it = std::find_if(spec_->options.begin(), spec_->options.end(),
                [&long_name](const OptionSpec& opt) { return opt.long_name == long_name; });
            
            if (it != spec_->options.end()) {
                if (it->kind == ArgKind::Value) {
                    if (i + 1 >= argc_) {
                        throw std::runtime_error("Missing value for option: " + long_name);
                    }
                    cmd.options[it->long_name] = argv_[++i];
                } else {
                    cmd.options[it->long_name] = "true";
                }
            } else {
                // Unknown long option, treat as positional
                cmd.arguments.push_back(arg);
            }
        }
        // Handle positional arguments
        else {
            cmd.arguments.push_back(arg);
        }
    }
}

void CommandParser::ValidateParsedCommand(const ParsedCommand& cmd) {
    // Check required options
    for (const auto& opt_spec : spec_->options) {
        if (opt_spec.required) {
            if (cmd.options.find(opt_spec.long_name) == cmd.options.end()) {
                std::string option_str = "--" + opt_spec.long_name;
                if (opt_spec.short_name) {
                    option_str = std::string("-") + opt_spec.short_name + ", " + option_str;
                }
                throw std::runtime_error("Required option missing: " + option_str);
            }
        }
    }
    
    // Check positional argument count
    size_t arg_count = cmd.arguments.size();
    if (arg_count < spec_->min_positional || arg_count > spec_->max_positional) {
        std::string error = "Incorrect number of arguments. Expected ";
        if (spec_->min_positional == spec_->max_positional) {
            error += std::to_string(spec_->max_positional);
        } else {
            error += std::to_string(spec_->min_positional) + " to " + std::to_string(spec_->max_positional);
        }
        error += " positional arguments, got " + std::to_string(arg_count);
        throw std::runtime_error(error);
    }
}

bool CommandParser::HasOption(const ParsedCommand& cmd, const std::string& opt) {
    return cmd.options.find(opt) != cmd.options.end();
}

std::string CommandParser::GetOption(const ParsedCommand& cmd, const std::string& opt, const std::string& default_val) {
    auto it = cmd.options.find(opt);
    if (it != cmd.options.end()) {
        return it->second;
    }
    return default_val;
}

} // namespace mnncli
