#pragma once

#include <string>
#include <utility>

namespace mnn::downloader {

// Model source types
enum class ModelSource {
    HUGGING_FACE,
    MODEL_SCOPE,
    MODELERS,
    UNKNOWN
};

// Model source utilities
class ModelSources {
public:
    static constexpr const char* SOURCE_HUGGING_FACE = "HuggingFace";
    static constexpr const char* SOURCE_MODEL_SCOPE = "ModelScope";
    static constexpr const char* SOURCE_MODELERS = "Modelers";
    
    // Get default org_name for a provider (lowercase)
    static std::string GetDefaultOrgName(const std::string& provider);
    
    // Convert string to ModelSource
    static ModelSource FromString(const std::string& source_str);
    
    // Convert ModelSource to string
    static std::string ToString(ModelSource source);
    
    // Extract source from model ID
    static ModelSource GetSource(const std::string& model_id);
    
    // Split model ID into source and path
    static std::pair<std::string, std::string> SplitSource(const std::string& model_id);
    
    // Get model name from model ID
    static std::string GetModelName(const std::string& model_id);
};

} // namespace mnn::downloader
