#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  list_command_handler.cpp
//
//  Handler for 'list' command
//

#include "handlers/list_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "local_model_utils.hpp"

namespace mnncli {
using namespace mnn::downloader;

std::string ListCommandHandler::CommandName() const {
    return "list";
}

const CommandSpec& ListCommandHandler::GetSpec() const {
    return mnncli::GetSpec("list");
}

int ListCommandHandler::Handle(const ParsedCommand& cmd) {
    return LocalModelUtils::ListLocalModels();
}

} // namespace mnncli
