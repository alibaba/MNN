#pragma once

#include <string>
#include "MNN/expr/Expr.hpp"
#include "llm/prompt.hpp"
#include "video/video_processor.h"

namespace mls {

struct MultimodalProcessingResult {
    bool hasMultimodal{false};
    MNN::Transformer::MultimodalPrompt multimodalPrompt;
    std::string errorMessage;
};

struct MultimodalProcessorConfig {
    int max_debug_images{999999};
    bool save_first_image{true};
    VideoProcessorConfig video_processor_config;
};

class MultimodalProcessor {
public:
    explicit MultimodalProcessor(MultimodalProcessorConfig config = {});

    MultimodalProcessingResult process(const std::string& prompt_text) const;

private:
    struct ProcessorState {
        std::string final_prompt;
        int image_index{0};
        int successful_loads{0};
        int failed_loads{0};
        bool first_image_saved{false};
    };

    static MNN::Express::VARP loadImageFromPath(const std::string& image_path);
    static std::string escapeForRegex(const std::string& text);
    bool handleImageTags(const std::string& prompt_text,
                         MultimodalProcessingResult& result,
                         ProcessorState& state) const;
    bool handleVideoTags(const std::string& prompt_text,
                         MultimodalProcessingResult& result,
                         ProcessorState& state) const;

    MultimodalProcessorConfig config_;
};

} // namespace mls
