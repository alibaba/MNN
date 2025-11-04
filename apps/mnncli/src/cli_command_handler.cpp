//
//  cli_command_handler.cpp
//
//  Command dispatcher implementation
//

#include "cli_command_handler.hpp"
#include "cli_command_parser.hpp"

namespace mnncli {

void CommandDispatcher::Register(std::unique_ptr<ICommandHandler> handler) {
    handlers_[handler->CommandName()] = std::move(handler);
}

int CommandDispatcher::Dispatch(const ParsedCommand& cmd) {
    auto it = handlers_.find(cmd.command);
    if (it == handlers_.end()) {
        return -2; // Not found
    }
    return it->second->Handle(cmd);
}

} // namespace mnncli
