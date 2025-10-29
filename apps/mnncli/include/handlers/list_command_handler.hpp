//
//  list_command_handler.hpp
//
//  Handler for 'list' command
//

#pragma once

#include "cli_command_handler.hpp"

namespace mnncli {

struct ParsedCommand;
struct CommandSpec;

class ListCommandHandler : public ICommandHandler {
public:
    std::string CommandName() const override;
    int Handle(const ParsedCommand& cmd) override;
    const CommandSpec& GetSpec() const override;
};

} // namespace mnncli
