//
//  run_command_handler.hpp
//
//  Handler for 'run' command
//

#pragma once

#include "cli_command_handler.hpp"

namespace mnncli {

struct ParsedCommand;
struct CommandSpec;

class RunCommandHandler : public ICommandHandler {
public:
    std::string CommandName() const override;
    int Handle(const ParsedCommand& cmd) override;
    const CommandSpec& GetSpec() const override;
    
private:
    void PrintRunUsage() const;
};

} // namespace mnncli
