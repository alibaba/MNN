//
//  cli_command_handler.hpp
//
//  Command execution handlers
//

#pragma once

#include <string>
#include <memory>
#include <map>

namespace mnncli {

struct ParsedCommand;
struct CommandSpec;

// Base interface for command handlers
class ICommandHandler {
public:
    virtual ~ICommandHandler() = default;
    virtual std::string CommandName() const = 0;
    virtual int Handle(const ParsedCommand& cmd) = 0;
    virtual const CommandSpec& GetSpec() const = 0;
};

// Dispatcher for command handlers
class CommandDispatcher {
public:
    void Register(std::unique_ptr<ICommandHandler> handler);
    int Dispatch(const ParsedCommand& cmd);
    
private:
    std::map<std::string, std::unique_ptr<ICommandHandler>> handlers_;
};

} // namespace mnncli

