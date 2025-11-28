//
//  benchmark_command_handler.hpp
//
//  Handler for 'benchmark' command
//

#pragma once

#include "cli_command_handler.hpp"

namespace mnncli {

struct ParsedCommand;
struct CommandSpec;

class BenchmarkCommandHandler : public ICommandHandler {
public:
    std::string CommandName() const override;
    int Handle(const ParsedCommand& cmd) override;
    const CommandSpec& GetSpec() const override;
};

} // namespace mnncli
