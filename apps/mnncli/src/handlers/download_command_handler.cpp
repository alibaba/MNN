#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  download_command_handler.cpp
//
//  Handler for 'download' command
//

#include "handlers/download_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "model_manager.hpp"

namespace mnncli {
using namespace mnn::downloader;

std::string DownloadCommandHandler::CommandName() const {
    return "download";
}

const CommandSpec& DownloadCommandHandler::GetSpec() const {
    return mnncli::GetSpec("download");
}

int DownloadCommandHandler::Handle(const ParsedCommand& cmd) {
    if (cmd.arguments.empty()) {
        return 1;
    }
    std::string model_name = cmd.arguments[0];
    return ModelManager::DownloadModel(model_name, cmd.verbose);
}

} // namespace mnncli
