#pragma once

#include <string>
#include "MNN/expr/Expr.hpp"
#include "llm/llm.hpp"
#include "video/video_processor.h"

namespace mls {

struct PromptProcessingResult {
    bool has_multimodal{false};
    MNN::Transformer::MultimodalPrompt multimodal_prompt;
    std::string error_message;
};

struct PromptProcessorConfig {
    int max_debug_images{64};
    bool save_first_image{true};
    VideoProcessorConfig video_processor_config;
};

class PromptProcessor {
public:
    explicit PromptProcessor(PromptProcessorConfig config = {});

    PromptProcessingResult Process(const std::string& prompt_text) const;

private:
    struct ProcessorState {
        std::string final_prompt;
        int image_index{0};
        int successful_loads{0};
        int failed_loads{0};
        bool first_image_saved{false};
    };

    static MNN::Express::VARP LoadImageFromPath(const std::string& image_path);
    static std::string EscapeForRegex(const std::string& text);
    bool HandleImageTags(const std::string& prompt_text,
                         PromptProcessingResult& result,
                         ProcessorState& state) const;
    bool HandleVideoTags(const std::string& prompt_text,
                         PromptProcessingResult& result,
                         ProcessorState& state) const;

    PromptProcessorConfig config_;
};

} // namespace mls
