//
// Created by ruoyi.sjd on 2024/12/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "file_utils.hpp"
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#include "mls_config.hpp"

namespace mls {
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
    return storage_folder / "snapshots" / commit_hash;
}

fs::path FileUtils::GetPointerPath(const fs::path& storage_folder, const std::string& commit_hash, const fs::path& relative_filename) {
    return storage_folder / "snapshots" / commit_hash / relative_filename;
}

void FileUtils::CreateSymlink(const fs::path& target, const fs::path& link, std::error_code& ec) {
    fs::create_symlink(target, link, ec);
}

std::string FileUtils::GetFileName(const std::string& path) {
    const auto pos = path.rfind('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string FileUtils::GetFolderLinkerPath(const std::string& model_id) {
    return (fs::path(GetBaseCacheDir()) /  GetFileName(model_id)).string();
}

std::string FileUtils::GetStorageFolderPath(const std::string& model_id) {
    const auto repo_folder_name = RepoFolderName(model_id, "model");
    return (fs::path(GetBaseCacheDir()) / repo_folder_name).string();
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

std::string FileUtils::GetBaseCacheDir() {
    std::string cache_dir;

#ifdef _WIN32
    const char* home_dir = std::getenv("USERPROFILE");
    if (home_dir) {
        cache_dir = std::string(home_dir) + "\\" + kCachePath;
    }
#else
    const char* home_dir = std::getenv("HOME");
    if (home_dir) {
        cache_dir = std::string(home_dir) + "/" + kCachePath;
    }
#endif

    if (cache_dir.empty()) {
        fprintf(stderr, "Unable to get home directory.");
        return ""; // Handle error appropriately in your application
    }

    std::filesystem::path cache_path(cache_dir);

    if (!exists(cache_path)) {
        create_directory(cache_path);
    }

    return cache_path.string();
}

std::string FileUtils::GetConfigPath(const std::string& model_id) {
    return (fs::path(GetBaseCacheDir())/model_id/"config.json").string();
}

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
