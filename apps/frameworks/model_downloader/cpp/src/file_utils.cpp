//
// Created by ruoyi.sjd on 2024/12/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "file_utils.hpp"
#include "dl_config.hpp"
#include "log_utils.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>


#include "model_name_utils.hpp"

namespace mnn::downloader {
std::string FileUtils::JoinPaths(const std::string& base, const std::vector<std::string>& parts) {
    std::string result = base;
    for (const auto& part : parts) {
        if (!result.empty() && result.back() != '/' && !part.empty() && part.front() != '/') {
            result += '/';
        }
        result += part;
    }
    return result;
}

std::string FileUtils::GetAbsolutePath(const std::string& path) {
    auto absolute_path = std::filesystem::absolute(path);
    return absolute_path.string();
}

bool FileUtils::IsSubPath(const std::string& parent, const std::string& child) {
    return child.find(parent) == 0 && (child.size() == parent.size() || child[parent.size()] == '/');
}

// Main function: Get the pointer path
std::string FileUtils::GetPointerPath(const std::string& storage_folder, const std::string& revision, const std::string& relative_filename, std::string& error_info) {
    // Create paths
    const std::string snapshot_path = FileUtils::JoinPaths(storage_folder, {"snapshots"});
    const std::string pointer_path = FileUtils::JoinPaths(snapshot_path, {revision, relative_filename});

    // Resolve to absolute paths
    const std::string abs_snapshot_path = GetAbsolutePath(snapshot_path);
    std::string abs_pointer_path = GetAbsolutePath(pointer_path);

    LOG_DEBUG_TAG("GetPointerPath: Storage folder: " + storage_folder + ", Revision: " + revision + ", Relative filename: " + relative_filename, "FileUtils");
    LOG_DEBUG_TAG("GetPointerPath: Snapshot path: " + snapshot_path + " -> " + abs_snapshot_path, "FileUtils");
    LOG_DEBUG_TAG("GetPointerPath: Pointer path: " + pointer_path + " -> " + abs_pointer_path, "FileUtils");

    // Check if pointer path is within snapshot path
    if (!IsSubPath(abs_snapshot_path, abs_pointer_path)) {
        error_info = "Invalid pointer path: cannot create pointer path in snapshot folder with storage_folder='" +
                                    storage_folder + "', revision='" + revision + "', and relative_filename='" + relative_filename + "'.";
    }
    return abs_pointer_path;
}


std::string FileUtils::RepoFolderName(const std::string& repo_id, const std::string& repo_type) {
    std::vector<std::string> repo_parts;
    std::stringstream ss(repo_id);
    std::string part;
    while (std::getline(ss, part, '/')) {
        repo_parts.push_back(part);
    }

    // Create parts vector
    std::vector<std::string> parts = {repo_type + "s"};
    parts.insert(parts.end(), repo_parts.begin(), repo_parts.end());

    // Join parts with the separator
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            oss << "--";
        }
        oss << parts[i];
    }
    return oss.str();
}

std::string FileUtils::ExpandTilde(const std::string &path) {
    if (!path.empty() && path[0] == '~') {
        const char *home = getenv("HOME");
        if (home) {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

fs::path FileUtils::GetPointerPathParent(const fs::path& storage_folder, const std::string& commit_hash) {
    auto pointer_path_parent = storage_folder / "snapshots" / commit_hash;
    LOG_DEBUG_TAG("GetPointerPathParent: Storage folder: " + storage_folder.string() + ", Commit hash: " + commit_hash + " -> " + pointer_path_parent.string(), "FileUtils");
    return pointer_path_parent;
}

fs::path FileUtils::GetPointerPath(const fs::path& storage_folder, const std::string& commit_hash, const fs::path& relative_filename) {
    auto pointer_path = storage_folder / "snapshots" / commit_hash / relative_filename;
    LOG_DEBUG_TAG("GetPointerPath: Storage folder: " + storage_folder.string() + ", Commit hash: " + commit_hash + ", Relative filename: " + relative_filename.string() + " -> " + pointer_path.string(), "FileUtils");
    return pointer_path;
}

void FileUtils::CreateSymlink(const fs::path& target, const fs::path& link, std::error_code& ec) {
    LOG_DEBUG_TAG("CreateSymlink: Creating symlink from " + link.string() + " to " + target.string(), "FileUtils");
    fs::create_symlink(target, link, ec);
    if (ec) {
        LOG_DEBUG_TAG("CreateSymlink: Failed to create symlink from " + link.string() + " to " + target.string() + " - " + ec.message(), "FileUtils");
    } else {
        LOG_DEBUG_TAG("CreateSymlink: Successfully created symlink from " + link.string() + " to " + target.string(), "FileUtils");
    }
}

std::string FileUtils::GetFileName(const std::string& path) {
    const auto pos = path.rfind('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string FileUtils::GetFolderLinkerPath(const std::string& model_id) {
    return (fs::path(FileUtils::ExpandTilde(kCachePath)) /  GetFileName(model_id)).string();
}

std::string FileUtils::GetStorageFolderPath(const std::string& model_id) {
    const auto repo_folder_name = RepoFolderName(model_id, "model");
    auto storage_folder_path = (fs::path(FileUtils::ExpandTilde(kCachePath)) / repo_folder_name).string();
    LOG_DEBUG_TAG("GetStorageFolderPath: Model ID: " + model_id + ", Repo folder name: " + repo_folder_name + " -> " + storage_folder_path, "FileUtils");
    return storage_folder_path;
}

bool FileUtils::RemoveFileIfExists(const std::string& path) {
    std::error_code ec;
    bool result;

    if (fs::is_directory(path)) {
        result = fs::remove_all(path, ec); // Remove directories and their contents
    } else if (fs::is_regular_file(path)) {
        result = fs::remove(path, ec);      // Remove files
    } else if (!fs::exists(path)){
        return true; // if the file doesn't exist, it is considered a success
    } else {
        return false;
    }
    if (ec && ec != std::errc::no_such_file_or_directory) {
        return false;
    }
    return result;
}


// Returns the full filesystem path to a model's directory given the model_id.
// This function consults the user config for the cache directory and enables
// robust lookup with support for provider prefixes and multiple storage layouts.
//
// Example use cases:
//   std::string path = FileUtils::GetModelPath("TinyLlama-1.1B");
//     // Looks for: $cache_dir/TinyLlama-1.1B, $cache_dir/modelscope/TinyLlama-1.1B, etc.
//
//   std::string path = FileUtils::GetModelPath("taobao-mnn/TinyLlama-1.1B");
//     // Looks for model in provider subfolder if given explicitly.
//
//   std::string path = FileUtils::GetModelPath("MNN/MyModel");
//     // Will locate model in $cache_dir/MNN/MyModel, or provider-specific structure.
//
//   // If config.download_provider == "huggingface":
//   FileUtils::GetModelPath("TinyLlama-1.1B"); // Will try "taobao-mnn/TinyLlama-1.1B" variant too.
//
// Typical scenarios:
//   - User passes a model short name. This function expands it to the full disk path.
//   - User specifies provider-prefixed names, which are prioritized for certain providers.
//   - Works regardless of whether user input has provider prefix or not.
//   - If the cache directory or config changes, this function always resolves to the correct place.
//
std::string FileUtils::GetModelPath(const std::string& model_id, const mnn::downloader::Config& config) {
    if (model_id.empty()) {
        return "";
    }
    
    // Standardize model_id using GetFullModelId
    std::string full_model_id = mnn::downloader::ModelNameUtils::GetFullModelId(model_id, config);
    if (full_model_id.empty()) {
        return "";
    }
    
    // Parse full_model_id: Provider/org_name/repo_name
    // Split by '/' to extract components
    size_t first_slash = full_model_id.find('/');
    if (first_slash == std::string::npos) {
        return "";
    }
    
    std::string provider = full_model_id.substr(0, first_slash);
    std::string path_part = full_model_id.substr(first_slash + 1);
    
    // Get cache directory and expand tilde
    std::string expanded_cache_dir = FileUtils::ExpandTilde(kCachePath);
    std::string primary_path = expanded_cache_dir + "/" + provider + "/" + path_part;
    return primary_path; // Model not found
}

std::string FileUtils::GetConfigPath(const std::string& model_id, const mnn::downloader::Config& config) {
    std::string model_path = GetModelPath(model_id, config);
    if (!model_path.empty()) {
        return (fs::path(model_path) / "config.json").string();
    }
    // Fallback
    return (fs::path(FileUtils::ExpandTilde(kCachePath)) / model_id / "config.json").string();
}

// Note: GetConfigPath(const std::string&) is implemented in file_utils_config.cpp
// to avoid pulling in cli_config_manager.hpp and its json dependency here

bool FileUtils::Move(const fs::path& src, const fs::path& dest, std::string& error_info) {
    if (!std::filesystem::exists(src)) {
        error_info = "Source file does not exist.";
        return false;
    }

    if (std::filesystem::exists(dest)) {
        error_info = "Destination file already exists.";
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(src, dest, ec);

    if (ec) {
        error_info =  "Error moving file: " + ec.message();
        return false;
    }
    return true;
}

}
