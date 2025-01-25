//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "remote_model_downloader.hpp"
#include "../include/httplib.h"
#include <cstdio>
#include <fstream>
#include <functional>
#include "file_utils.hpp"

static const std::string HUGGINGFACE_HEADER_X_REPO_COMMIT = "x-repo-commit";
static const std::string HUGGINGFACE_HEADER_X_LINKED_ETAG = "x-linked-etag";
static const std::string HUGGINGFACE_HEADER_X_LINKED_SIZE = "x-linked-size";

namespace mls {

    size_t ParseContentLength(const std::string& content_length) {
        if (!content_length.empty()) {
            return std::stoul(content_length);
        }
        return 0;
    }

    std::string NormalizeETag(const std::string& etag) {
        if (!etag.empty() && etag[0] == '"' && etag[etag.size() - 1] == '"') {
            return etag.substr(1, etag.size() - 2);
        }
        return etag;
    }

    //from get_hf_file_metadata
    HfFileMetadata RemoteModelDownloader::GetFileMetadata(const std::string& url, std::string& error_info) {
        httplib::SSLClient cli(this->host_, 443);
        httplib::Headers headers;
        HfFileMetadata metadata{};
        headers.emplace("Accept-Encoding", "identity");
        auto res = cli.Head(url, headers);
        if (!res || (res->status != 200 && res->status != 302)) {
            error_info = "GetFileMetadata Failed to fetch metadata status " + std::to_string(res ? res->status : -1);
            return metadata;
        }
        metadata.location = url;
        if (res->status == 302) {
            metadata.location = res->get_header_value("Location");
        }
        std::string linked_etag = res->get_header_value(HUGGINGFACE_HEADER_X_LINKED_ETAG);
        std::string etag = res->get_header_value("ETag");
        metadata.etag = NormalizeETag(!linked_etag.empty() ? linked_etag : etag);
        std::string linked_size = res->get_header_value(HUGGINGFACE_HEADER_X_LINKED_SIZE);
        std::string content_length = res->get_header_value("Content-Length");
        metadata.size = ParseContentLength(!linked_size.empty() ? linked_size : content_length);
        metadata.commit_hash = res->get_header_value(HUGGINGFACE_HEADER_X_REPO_COMMIT);
        return metadata;
    }

    RemoteModelDownloader::RemoteModelDownloader(std::string host, int max_attempts, int retry_delay_seconds)
        : max_attempts_(max_attempts),
          retry_delay_seconds_(retry_delay_seconds),
        host_(std::move(host)){
    }

    std::string RemoteModelDownloader::DownloadWithRetries(
                      const fs::path& storage_folder,
                      const std::string& repo,
                      const std::string& revision,
                      const std::string& relative_path,
                      std::string& error_info,
                      int max_retries) {
        int attempt = 0;
        bool success = false;
        while (attempt < max_retries) {
            // Reset error message for each attempt
            error_info.clear();
            auto result = this->DownloadFile(storage_folder, repo, revision, relative_path, error_info);
            if (error_info.empty()) {
                success = true;
                return result;
            } else {
                attempt++;
                fprintf(stderr, "DownloadFile error at file: %s error message: %s, attempt: %d\n", 
                        relative_path.c_str(), error_info.c_str(), attempt);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        return "";
    }

    std::string RemoteModelDownloader::DownloadFile(
                                    const fs::path& storage_folder,
                                    const std::string& repo,
                                    const std::string& revision,
                                    const std::string& relative_path,
                                    std::string& error_info) {
        std::string url = "https://" + this->host_ + "/";
        url += repo;
        url += "/resolve/" + revision + "/" + relative_path;
        auto metadata = GetFileMetadata(url, error_info);
        if (!error_info.empty()) {
            printf("DownloadFile GetFileMetadata faield\n");
            return "";
        }
        auto repo_folder_name = mls::FileUtils::RepoFolderName(repo, "model");
        fs::path blob_path = storage_folder / "blobs" / metadata.etag;
        std::filesystem::path blob_path_incomplete = storage_folder / "blobs" / (metadata.etag + ".incomplete");
        fs::path pointer_path = mls::FileUtils::GetPointerPath(
            storage_folder, metadata.commit_hash, relative_path);
        fs::create_directories(blob_path.parent_path());
        fs::create_directories(pointer_path.parent_path());

        if (fs::exists(pointer_path)) {
            return pointer_path.string();
        } else if (fs::exists(blob_path)) {
            std::error_code ec;
            mls::FileUtils::CreateSymlink(blob_path, pointer_path, ec);
            if (ec) {
                fprintf(stderr, "DownloadFile create symlink error for pointer_path: %s", pointer_path.string().c_str());
                error_info = ec.message();
                return "";
            }
            printf("DownloadFile  %s already exists just create symlink\n", relative_path.c_str());
            return pointer_path.string();
        }

        std::mutex lock;
        {
            std::lock_guard guard(lock);
            httplib::Headers headers;
            DownloadToTmpAndMove(blob_path_incomplete, blob_path, metadata.location, headers, metadata.size,
                                 relative_path, false, error_info);
            if (error_info.empty()) {
                std::error_code ec;
                FileUtils::CreateSymlink(blob_path, pointer_path, ec);
                if (ec) {
                    error_info = "create link error" + ec.message();
                }
            }
        }
        return pointer_path.string();
    }

    void RemoteModelDownloader::DownloadToTmpAndMove(
        const fs::path& incomplete_path,
        const fs::path& destination_path,
        const std::string& url_to_download,
        httplib::Headers& headers,
        size_t expected_size,
        const std::string& file_name,
        bool force_download,
        std::string& error_info) {
        if (fs::exists(destination_path) && !force_download) {
            return;
        }
        if (std::filesystem::exists(incomplete_path) && force_download) {
            std::filesystem::remove(incomplete_path);
        }
        size_t resume_size = std::filesystem::exists(incomplete_path) ? std::filesystem::file_size(incomplete_path) : 0;
        DownloadFileInner(url_to_download, incomplete_path, {}, resume_size, headers, expected_size, file_name, error_info);
        if (error_info.empty()) {
            printf("DownloadFile  %s success\n", file_name.c_str());
            MoveWithPermissions(incomplete_path, destination_path, error_info);
        } else {
            printf("DownloadFile  %s failed\n", file_name.c_str());
        }
    }

    void RemoteModelDownloader::DownloadFileInner(
        const std::string& url,
        const std::filesystem::path& temp_file,
        const std::unordered_map<std::string, std::string>& proxies,
        size_t resume_size,
        const httplib::Headers& headers,
        const size_t expected_size,
        const std::string& displayed_filename,
        std::string& error_info
    ) {
        auto [host, path] = HfApiClient::ParseUrl(url);
        httplib::SSLClient client(host, 443);
        httplib::Headers request_headers(headers.begin(), headers.end());
        if (resume_size > 0) {
            printf("DownloadFile %s resume size %zu", displayed_filename.c_str(), resume_size);
            request_headers.emplace("Range", "bytes=" + std::to_string(resume_size) + "-");
        }
        std::ofstream output(temp_file, std::ios::binary | std::ios::app);
        DownloadProgress progress;
        progress.downloaded = resume_size;
        auto res = client.Get(path, request_headers,
              [&](const httplib::Response& response) {
                  auto content_length_str = response.get_header_value("Content-Length");
                  if (!content_length_str.empty()) {
                      progress.content_length = std::stoull(content_length_str) + resume_size;
                  }
                  return true;
              },
              [&](const char* data, size_t data_length) {
                  output.write(data, data_length);
                  progress.downloaded += data_length;
                  if (expected_size > 0) {
                      double percentage = (static_cast<double>(progress.downloaded) / progress.content_length) * 100.0;
                      printf("\rDownloadFile %s progress: %.2f%%", displayed_filename.c_str(), percentage);
                      fflush(stdout);
                  }
                  return true;
              }
        );
        output.flush();
        output.close();
        if (res) {
            if (res->status >= 200 && res->status < 300 || res->status == 416) {
                progress.success = true;
                printf("\n");
            } else {
                error_info = "HTTP error: " + std::to_string(res->status);
            }
        } else {
            error_info  = "Connection error: " + std::string(httplib::to_string(res.error()));
        }
        if (!error_info.empty()) {
            printf("HTTP Get Error: %s\n", error_info.c_str());
        }
    }

    bool RemoteModelDownloader::CheckDiskSpace(size_t required_size, const std::filesystem::path& path) {
        auto space = std::filesystem::space(path);
        if (space.available < required_size) {
            return false;
        }
        return true;
    }

    void RemoteModelDownloader::MoveWithPermissions(const std::filesystem::path& src,
                                                    const std::filesystem::path& dest,
                                                    std::string& error_info) {
        if (FileUtils::Move(src, dest, error_info)) {
            std::filesystem::permissions(dest, std::filesystem::perms::owner_all);
        }
    }

}
