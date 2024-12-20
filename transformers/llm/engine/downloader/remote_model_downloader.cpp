//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "remote_model_downloader.hpp"
#include "httplib.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <regex>
#include <sys/stat.h>
#include <thread>

#include "downloader_file_utils.hpp"

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
            return etag.substr(1, etag.size() - 2); // 去掉引号
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
        // 文件大小解析
        std::string linked_size = res->get_header_value(HUGGINGFACE_HEADER_X_LINKED_SIZE);
        std::string content_length = res->get_header_value("Content-Length");
        metadata.size = ParseContentLength(!linked_size.empty() ? linked_size : content_length);
        metadata.commit_hash = res->get_header_value(HUGGINGFACE_HEADER_X_REPO_COMMIT);
        return metadata;
    }


    RemoteModelDownloader::RemoteModelDownloader(int max_attempts, int retry_delay_seconds)
        : max_attempts_(max_attempts),
          retry_delay_seconds_(retry_delay_seconds) {
        cache_path_ = mls::DownloaderFileUtils::ExpandTilde("~/.mnnmodels");
    }

    std::string RemoteModelDownloader::DownloadFromHF(const std::string& repo,
                                                      const std::string& revision,
                                                      const std::string& file_on_repo,
                                                      const std::string& local_path,
                                                      const std::string& hf_token) {
        std::string url = "https://huggingface.co/";
        url += repo;
        url += "/resolve/" + revision + "/" + file_on_repo;
        std::string local_path_real = local_path;
        if (local_path_real.empty()) {
            local_path_real = "~/.mnnmodels/" + repo + "/" + file_on_repo;
            local_path_real = mls::DownloaderFileUtils::ExpandTilde(local_path_real);
        }
        auto result = DownloadFile(url, local_path_real, hf_token, repo, file_on_repo);
        return result;
    }

    mls::RepoInfo RemoteModelDownloader::getRepoInfo(
        const std::string& repo_name,
        const std::string& revision,
        const std::string& hf_token,
        std::string& error_info) {
        // Construct the API URL
        const std::string url = "https://huggingface.co/api/models/" + repo_name + "/revision/" + revision;
        // Parsed repository info
        RepoInfo repo_info;

        // Perform the API request
        auto request_func = [&]() -> bool {
            // Parse host and path from the URL
            static const std::regex url_regex(R"(https://([^/]+)(/.+))");
            std::smatch match;
            if (!std::regex_match(url, match, url_regex)) {
                error_info = "Invalid URL format";
                return {};
            }
            std::string host = match[1];
            std::string path = match[2];

            // Make the HTTPS request
            httplib::SSLClient cli(host, 443);
            httplib::Headers headers;
            if (!hf_token.empty()) {
                headers.emplace("Authorization", "Bearer " + hf_token);
            }

            auto res = cli.Get(path, headers);
            if (!res || res->status != 200) {
                return false;
            }

            // Parse the JSON response
            rapidjson::Document doc;
            if (doc.Parse(res->body.c_str()).HasParseError()) {
                error_info = "Failed to parse JSON response";
                return {};
            }

            // Extract fields
            if (doc.HasMember("modelId") && doc["modelId"].IsString()) {
                repo_info.model_id = doc["modelId"].GetString();
            }
            if (doc.HasMember("sha") && doc["sha"].IsString()) {
                repo_info.sha = doc["sha"].GetString();
            }
            if (doc.HasMember("revision") && doc["revision"].IsString()) {
                repo_info.revision = doc["revision"].GetString();
            }
            if (doc.HasMember("siblings") && doc["siblings"].IsArray()) {
                const rapidjson::Value& siblings = doc["siblings"];
                for (rapidjson::Value::ConstValueIterator it = siblings.Begin(); it != siblings.End(); ++it) {
                    if (it->IsObject() && it->HasMember("rfilename") && (*it)["rfilename"].IsString()) {
                        repo_info.siblings.emplace_back((*it)["rfilename"].GetString());
                    }
                }
            }

            return true;
        };

        if (!PerformRequestWithRetry(request_func)) {
            error_info = "Failed to fetch repository info after retries";
            return {};
        }

        return repo_info;
    }

    std::string RemoteModelDownloader::DownloadFile(const std::string& url,
                                                    const std::string& local_path,
                                                    const std::string& hf_token,
                                                    const std::string& repo,
                                                    const std::string& relative_path) {
        if (!EnsureDirectoryExists(local_path)) {
            return "";
        }
        // Step 1: 获取文件元信息
        std::string error_info;
        auto metadata = GetFileMetadata(url, error_info);
        if (!error_info.empty()) {
            printf("DownloadFile GetFileMetadata faield\n");
            return "";
        }
        printf("DownloadFile GetFileMetadata file_size :%ld, real_url: %s \n", metadata.size,
               metadata.location.c_str());

        auto repo_folder_name = mls::DownloaderFileUtils::RepoFolderName(repo, "model");
        fs::path storage_folder = fs::path(this->cache_path_) / repo_folder_name;
        fs::path blob_path = storage_folder / "blobs" / metadata.etag;
        fs::path blob_path_incomplete = storage_folder / "blobs" / (metadata.etag + ".incomplete");
        fs::path pointer_path = mls::DownloaderFileUtils::GetPointerPath(
            storage_folder, metadata.commit_hash, relative_path);

        printf("DownloadFile GetFileMetadata storage_folder :%s blob_path: %s pointer_path:%s \n",
               storage_folder.c_str(), blob_path.c_str(), pointer_path.c_str());

        fs::create_directories(blob_path.parent_path());
        fs::create_directories(pointer_path.parent_path());

        if (fs::exists(pointer_path)) {
            return pointer_path.string();
        }

        if (fs::exists(blob_path)) {
            mls::DownloaderFileUtils::CreateSymlink(blob_path, pointer_path);
            return pointer_path.string();
        }

        std::mutex lock;
        {
            std::lock_guard<std::mutex> guard(lock);
            httplib::Headers headers;
            DownloadToTmpAndMove(blob_path_incomplete, blob_path, metadata.location, headers, metadata.size,
                                 relative_path, false);
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
        bool force_download
    ) {
        std::cout << "Downloading " << url_to_download << " to " << destination_path << std::endl;

        if (fs::exists(destination_path) && !force_download) {
            return;
        }

        if (std::filesystem::exists(incomplete_path) && force_download) {
            std::filesystem::remove(incomplete_path);
        }
        std::ofstream temp_file(incomplete_path, std::ios::binary | std::ios::app);
        size_t resume_size = std::filesystem::exists(incomplete_path) ? std::filesystem::file_size(incomplete_path) : 0;
        // CheckDiskSpace(*expected_size, incomplete_path.parent_path());
        // CheckDiskSpace(*expected_size, destination_path.parent_path());
        std::string error_info;
        HttpGet(url_to_download, incomplete_path, {}, resume_size, headers, expected_size, file_name, error_info);
        if (error_info.empty()) {
            MoveWithPermissions(incomplete_path, destination_path);
        }
    }

    void RemoteModelDownloader::HttpGet(
        const std::string& url,
        const std::filesystem::path& temp_file,
        const std::unordered_map<std::string, std::string>& proxies,
        size_t resume_size,
        const httplib::Headers& headers,
        const std::optional<size_t>& expected_size,
        const std::string& displayed_filename,
        std::string& error_info
    ) {
        httplib::SSLClient client(this->host_, 443);
        httplib::Headers request_headers(headers.begin(), headers.end());
        if (resume_size > 0) {
            request_headers.emplace("Range", "bytes=" + std::to_string(resume_size) + "-");
        }
        std::ofstream output(temp_file, std::ios::binary | std::ios::app);
        Progress progress;
        auto res = client.Get(url, request_headers,
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

                  if (expected_size.has_value()) {
                      double percentage = (static_cast<double>(progress.downloaded) / progress.content_length) * 100.0;
                      printf("\r%s download Progress: %.2f%%", displayed_filename.c_str(), percentage);
                      fflush(stdout);
                  }
                  return true;
              }
        );
        if (res) {
            if (res->status >= 200 && res->status < 300 || res->status == 416) {
                printf("\r%s download success\n", displayed_filename.c_str());
                progress.success = true;
            } else {
                error_info = "HTTP error: " + std::to_string(res->status);
            }
        } else {
            error_info  = "Connection error: " + std::string(httplib::to_string(res.error()));
        }
        if (!error_info.empty()) {
            printf("HTTP Get Error: %s\n", progress.error_message.c_str());
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
                                                    const std::filesystem::path& dest) {
        std::filesystem::rename(src, dest);
        std::filesystem::permissions(dest, std::filesystem::perms::owner_all);
    }


    bool RemoteModelDownloader::ShouldDownload(const std::string& local_path,
                                               const std::string& url,
                                               const std::string& new_etag,
                                               const std::string& new_last_modified) {
        struct stat st;
        bool file_exists = (stat(local_path.c_str(), &st) == 0);

        // 如果本地没有文件，显然要下载
        if (!file_exists) {
            return true;
        }

        // 读取 metadata 文件
        std::string metadata_path = local_path + ".json";
        std::string old_etag, old_last_modified, old_url;
        LoadMetadata(metadata_path, old_etag, old_last_modified, old_url);

        // 如果 URL 和当前不一致，触发强制重新下载
        if (!old_url.empty() && old_url != url) {
            return true;
        }

        // 如果 ETag 不同，或者 Last-Modified 不同，就要下载
        if (!new_etag.empty() && !old_etag.empty() && new_etag != old_etag) {
            return true;
        }
        if (!new_last_modified.empty() && !old_last_modified.empty() && new_last_modified != old_last_modified) {
            return true;
        }

        // 否则不需要下载
        return false;
    }


    bool RemoteModelDownloader::EnsureDirectoryExists(const std::string& path) {
        std::string expanded_path = mls::DownloaderFileUtils::ExpandTilde(path);
        // Expand ~ to home directory if present

        // Start building the path
        size_t pos = 0;
        std::string current_path;

        // Handle absolute paths or tilde
        if (expanded_path[0] == '/') {
            current_path = "/";
            pos = 1; // Skip the leading '/'
        }
        else if (expanded_path[0] == '~') {
            // This should already be handled by ExpandTilde, but double-check
            std::cerr << "Error: Path contains unexpanded ~: " << expanded_path << std::endl;
            return false;
        }

        // Iterate through components
        while (pos < expanded_path.size()) {
            size_t next_pos = expanded_path.find('/', pos);
            std::string segment = expanded_path.substr(pos, next_pos - pos);

            if (!segment.empty()) {
                current_path += segment + "/";
                if (current_path.size() > path.size()) {
                    break;
                }
                struct stat info;
                if (stat(current_path.c_str(), &info) != 0) {
                    // Directory does not exist; create it
                    if (mkdir(current_path.c_str(), 0755) != 0) {
                        std::cerr << "Failed to create directory: " << current_path << std::endl;
                        return false;
                    }
                }
                else if (!(info.st_mode & S_IFDIR)) {
                    std::cerr << "Path exists but is not a directory: " << current_path << std::endl;
                    return false;
                }
            }

            if (next_pos == std::string::npos) {
                break;
            }
            pos = next_pos + 1; // Move to the next segment
        }

        return true;
    }

    void RemoteModelDownloader::LoadMetadata(const std::string& metadata_path,
                                             std::string& old_etag,
                                             std::string& old_last_modified,
                                             std::string& old_url) {
        struct stat st;
        if (stat(metadata_path.c_str(), &st) != 0) {
            // 没有 metadata 文件
            return;
        }

        std::ifstream ifs(metadata_path);
        if (!ifs.is_open()) {
            return;
        }

        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        ifs.close();

        rapidjson::Document doc;
        if (doc.Parse(content.c_str()).HasParseError()) {
            // 解析错误
            return;
        }

        if (doc.HasMember("url") && doc["url"].IsString()) {
            old_url = doc["url"].GetString();
        }
        if (doc.HasMember("etag") && doc["etag"].IsString()) {
            old_etag = doc["etag"].GetString();
        }
        if (doc.HasMember("lastModified") && doc["lastModified"].IsString()) {
            old_last_modified = doc["lastModified"].GetString();
        }
    }

    void RemoteModelDownloader::WriteMetadata(const std::string& metadata_path,
                                              const std::string& url,
                                              const std::string& etag,
                                              const std::string& last_modified) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

        doc.AddMember("url", rapidjson::Value(url.c_str(), allocator), allocator);
        doc.AddMember("etag", rapidjson::Value(etag.c_str(), allocator), allocator);
        doc.AddMember("lastModified", rapidjson::Value(last_modified.c_str(), allocator), allocator);

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);

        std::ofstream ofs(metadata_path);
        ofs << buffer.GetString();
        ofs.close();
    }

    bool RemoteModelDownloader::PerformRequestWithRetry(std::function<bool()> request_func) {
        int attempts_left = max_attempts_;
        int attempt_count = 0;

        while (attempts_left > 0) {
            attempt_count++;
            if (request_func()) {
                return true;
            }
            attempts_left--;
            // 指数退避 delay = (retry_delay_seconds ^ (attempt_count-1)) * 1000 ms
            int backoff_ms = static_cast<int>(std::pow(retry_delay_seconds_, attempt_count - 1) * 1000);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        }
        return false;
    }
}
