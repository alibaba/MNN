//
// Created by ruoyi.sjd on 2024/12/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once
#include <string>
#include <stdexcept>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

namespace mls {

class FileUtils {
public:

    static std::string GetPointerPath(const std::string& storage_folder, const std::string& revision, const std::string& relative_filename, std::string& error_info);

    static std::string ExpandTilde(const std::string &path);

    static bool IsSubPath(const std::string& parent, const std::string& child);

    static std::string JoinPaths(const std::string& base, const std::vector<std::string>& parts);

    static std::string RepoFolderName(const std::string& repo_id, const std::string& repo_type);

    static fs::path GetPointerPath(const fs::path& storage_folder, const std::string& commit_hash, const fs::path& relative_filename);

    static fs::path GetPointerPathParent(const fs::path& storage_folder, const std::string& commit_hash);

    static void CreateSymlink(const fs::path& target, const fs::path& link, std::error_code& ec);

    static std::string GetFileName(const std::string& path);

    static std::string GetFolderLinkerPath(const std::string& model_id);

    static std::string GetStorageFolderPath(const std::string& model_id);

    static std::string GetBaseCacheDir();

    static bool RemoveFileIfExists(const std::string& path);
    
    static std::string GetConfigPath(const std::string& model_id);

    static bool Move(const fs::path& source, const fs::path& dest, std::string& error_info);

private:

    static std::string GetAbsolutePath(const std::string& path);

};

}