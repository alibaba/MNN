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
    /**
     * @brief Get the pointer path inside the storage folder
     *
     * @param storage_folder The base storage folder
     * @param revision The revision identifier
     * @param relative_filename The relative filename to resolve
     * @param error_info error_info
     * @return Resolved absolute pointer path
     */
    static std::string GetPointerPath(const std::string& storage_folder, const std::string& revision, const std::string& relative_filename, std::string& error_info);

    /**
     * 
     * @param path
     * @return expanded path
     */
    static std::string ExpandTilde(const std::string &path);

    /**
     * @brief Check if the child path is a sub-path of the parent path
     *
     * @param parent The parent path
     * @param child The child path
     * @return true if child is a sub-path of parent, false otherwise
     */
    static bool IsSubPath(const std::string& parent, const std::string& child);

    /**
     * @brief Join multiple path segments into one path
     *
     * @param base The base path
     * @param parts The additional path segments
     * @return Joined path
     */
    static std::string JoinPaths(const std::string& base, const std::vector<std::string>& parts);


    /**
     * @brief RepoFolderName
     * @param repo_id
     * @param repo_type
     * @return repo folder name
     */
    static std::string RepoFolderName(const std::string& repo_id, const std::string& repo_type);

    static fs::path GetPointerPath(const fs::path& storage_folder, const std::string& commit_hash, const fs::path& relative_filename);

    static fs::path GetPointerPathParent(const fs::path& storage_folder, const std::string& commit_hash);

    static void CreateSymlink(const fs::path& target, const fs::path& link);

    static std::string GetFileName(const std::string& path);

private:

    /**
     * @brief Get the absolute path of a given path
     *
     * @param path The input path
     * @return Resolved absolute path
     */
    static std::string GetAbsolutePath(const std::string& path);

};

}