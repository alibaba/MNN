#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  search_command_handler.cpp
//
//  Handler for 'search' command
//

#include "handlers/search_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "model_manager.hpp"

namespace mnncli {
using namespace mnn::downloader;

std::string SearchCommandHandler::CommandName() const {
    return "search";
}

const CommandSpec& SearchCommandHandler::GetSpec() const {
    return mnncli::GetSpec("search");
}

int SearchCommandHandler::Handle(const ParsedCommand& cmd) {
    if (cmd.arguments.empty()) {
        return 1;
    }
    std::string keyword = cmd.arguments[0];
    return ModelManager::SearchRemoteModels(keyword, cmd.verbose);
}

} // namespace mnncli
