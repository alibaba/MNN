//
//  cli_download_listener.hpp
//
//  CLI-specific download listener for user interface feedback
//

#pragma once

#include <string>
#include "model_file_downloader.hpp"
#include "model_repo_downloader.hpp"
#include "user_interface.hpp"

namespace mnncli {

class CLIDownloadListener : public mnn::downloader::DownloadListener {
public:
    CLIDownloadListener() = default;
    ~CLIDownloadListener() override = default;
    
    void OnDownloadStart(const std::string& model_id) override;
    void OnDownloadProgress(const std::string& model_id, const mnn::downloader::DownloadProgress& progress) override;
    void OnDownloadFinished(const std::string& model_id, const std::string& path) override;
    void OnDownloadFailed(const std::string& model_id, const std::string& error) override;
    void OnDownloadPaused(const std::string& model_id) override;
    void OnDownloadTaskAdded() override;
    void OnDownloadTaskRemoved() override;
    void OnRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) override;
    std::string GetClassTypeName() const override;
};

} // namespace mnncli

