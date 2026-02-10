//
//  cli_download_listener.cpp
//
//  CLI-specific download listener for user interface feedback
//

#include "cli_download_listener.hpp"
#include "log_utils.hpp"
#include "user_interface.hpp"
#include "model_repo_downloader.hpp"
#include <iostream>

using namespace mnn::downloader; // Use library namespace

namespace mnncli {

void CLIDownloadListener::OnDownloadStart(const std::string& model_id) {
    if (!model_id.empty() && model_id.find('/') != std::string::npos) {
        UserInterface::ShowInfo("Starting download: " + model_id);
    }
}

void CLIDownloadListener::OnDownloadProgress(const std::string& model_id, const mnn::downloader::DownloadProgress& progress) {
    if (model_id.empty() || progress.progress < 0) {
        return;
    }
    
    if (progress.total_size == 0) {
        UserInterface::ShowProgress("Downloading " + model_id, progress.progress);
        return;
    }
    
    std::string file_name = progress.current_file.empty() ? "file" : progress.current_file;
    file_name = mnn::downloader::ModelRepoDownloader::ExtractFileName(file_name);
    std::string size_info = mnn::downloader::ModelRepoDownloader::FormatFileSizeInfo(
        progress.saved_size, progress.total_size);
        
    std::string message = file_name + size_info;
    UserInterface::ShowProgress(message, progress.progress);
}

void CLIDownloadListener::OnDownloadFinished(const std::string& model_id, const std::string& path) {
    UserInterface::EndProgress();
    UserInterface::ShowInfo("Download finished: " + path);
}

void CLIDownloadListener::OnDownloadFailed(const std::string& model_id, const std::string& error) {
    UserInterface::EndProgress();
    UserInterface::ShowError("Download failed: " + error);
}

void CLIDownloadListener::OnDownloadTaskAdded() {
    UserInterface::ShowInfo("Download task added");
}

void CLIDownloadListener::OnDownloadTaskRemoved() {
    UserInterface::ShowInfo("Download task removed");
}

void CLIDownloadListener::OnRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) {
    if (repo_size > 0) {
        UserInterface::ShowInfo("Repository info for " + model_id + ": " + LogUtils::FormatFileSize(repo_size));
    }
}

std::string CLIDownloadListener::GetClassTypeName() const {
    return "CLIDownloadListener";
}

void CLIDownloadListener::OnDownloadPaused(const std::string& message) {
    // Not used in CLI
}

} // namespace mnncli