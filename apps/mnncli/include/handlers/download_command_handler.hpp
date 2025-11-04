//
//  download_command_handler.hpp
//
//  Handler for 'download' command
//

#pragma once

#include "cli_command_handler.hpp"

namespace mnncli {

struct ParsedCommand;
struct CommandSpec;

class DownloadCommandHandler : public ICommandHandler {
public:
    std::string CommandName() const override;
    int Handle(const ParsedCommand& cmd) override;
    const CommandSpec& GetSpec() const override;
};

} // namespace mnncli
