//
//  model_manager.hpp
//
//  Model lifecycle management: download, search, list, delete
//

#pragma once

#include <string>
#include <vector>
#include "model_download_manager.hpp"
#include "cli_config_manager.hpp"
#include "cli_download_listener.hpp"
#include "model_repository.hpp"
#include "file_utils.hpp"
#include "log_utils.hpp"
#include "user_interface.hpp"
#include "model_name_utils.hpp"

namespace mnncli {

class ModelManager {
public:
    // Model search operations
    static int SearchRemoteModels(
        const std::string& keyword, 
        bool verbose = false, 
        const std::string& cache_dir_override = ""
    );
    
    // Model download operations
    static int DownloadModel(
        const std::string& model_name, 
        bool verbose = false, 
        const std::string& cache_dir_override = ""
    );
    
    // Model deletion operations
    static int DeleteModel(const std::string& model_name);
    
    // Model information operations
    static int ShowModelInfo(const std::string& model_name, bool verbose = false);

private:
    // Validation utilities
    static bool IsValidModelName(const std::string& model_name);
};

} // namespace mnncli

