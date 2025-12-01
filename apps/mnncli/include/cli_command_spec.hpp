//
//  cli_command_spec.hpp
//
//  Command specifications for declarative parsing
//

#pragma once

#include <string>
#include <vector>
#include <map>

namespace mnncli {

enum class ArgKind {
    Flag,       // Boolean flag (e.g., -v, --verbose)
    Value,      // Option with value (e.g., -c <path>, --config <path>)
    MultiValue  // Option with multiple values
};

struct OptionSpec {
    std::string long_name;
    char short_name;  // 0 if no short name
    ArgKind kind;
    bool required = false;
    std::string help;
};

struct CommandSpec {
    std::string name;
    std::vector<OptionSpec> options;
    size_t min_positional = 0;
    size_t max_positional = 0;
    std::string help;
};

// Command specs (defined in cli_command_spec.cpp)
const CommandSpec& GetSpec(const std::string& command);

// Generate usage text from spec
std::string MakeUsage(const CommandSpec& spec);

} // namespace mnncli
