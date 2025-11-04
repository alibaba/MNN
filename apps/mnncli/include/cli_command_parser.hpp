//
//  cli_command_parser.hpp
//
//  Command-line argument parsing and routing
//

#pragma once

#include <string>
#include <vector>
#include <map>

namespace mnncli {

struct ParsedCommand {
    std::string command;
    std::vector<std::string> arguments;
    std::map<std::string, std::string> options;
    bool verbose = false;
};

struct CommandSpec;

class CommandParser {
public:
    CommandParser(int argc, const char* argv[]);
    
    // Set command specification for parsing
    void SetCommandSpec(const CommandSpec& spec);
    
    // Parse command line arguments
    ParsedCommand Parse();
    
    // Utility functions
    static bool HasOption(const ParsedCommand& cmd, const std::string& opt);
    static std::string GetOption(const ParsedCommand& cmd, const std::string& opt, const std::string& default_val = "");
    
private:
    int argc_;
    const char** argv_;
    const CommandSpec* spec_ = nullptr;
    bool verbose_ = false;
    
    void ParseGlobalOptions();
    void ParseCommandOptions(ParsedCommand& cmd);
    void ValidateParsedCommand(const ParsedCommand& cmd);
    bool IsShortOption(const std::string& arg);
    bool IsLongOption(const std::string& arg);
};

} // namespace mnncli

