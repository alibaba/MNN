//
// Created by ruoyi.sjd on 2024/12/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "file_utils.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace mls {
// Helper function: Combine paths
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

// Helper function: Get the absolute path
std::string FileUtils::GetAbsolutePath(const std::string& path) {
    char absolute_path[PATH_MAX];
    if (realpath(path.c_str(), absolute_path)) {
        return {absolute_path};
    } else {
        return "";
    }
}

// Helper function: Check if `child` is a sub-path of `parent`
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

void FileUtils::CreateSymlink(const fs::path& target, const fs::path& link) {
    if (!fs::exists(link)) {
        fs::create_symlink(target, link);
    }
}

std::string FileUtils::GetFileName(const std::string& path) {
    const auto pos = path.rfind('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

}