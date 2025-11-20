//
//  local_model_utils.hpp
//
//  Utility functions for local model operations (listing, scanning)
//

#pragma once

#include <string>
#include <vector>

namespace mnncli {

class LocalModelUtils {
public:
    // Get local model names from a specific provider's cache directory
    // Only returns models with .mnncli/.complete marker (fully downloaded)
    // Args:
    //   provider: Provider name (e.g., "HuggingFace", "ModelScope", "Modelers")
    //   cache_dir: Optional cache directory (for testing). If empty, uses ConfigManager
    // Returns:
    //   Vector of model identifiers in "provider/owner/model_name" format
    static std::vector<std::string> ListLocalModelsInner(const std::string& provider, 
                                                          const std::string& cache_dir = "");
    
    // List all local models (with formatted output)
    static int ListLocalModels();

    // Check if a model directory is fully downloaded (has .mnncli/.complete marker)
    // Args:
    //   model_entry_path: Absolute path to the model directory
    // Returns:
    //   true if the directory represents a fully downloaded model, false otherwise
    static bool CheckIsDownloadedModel(const std::string& model_entry_path);
};

} // namespace mnncli

