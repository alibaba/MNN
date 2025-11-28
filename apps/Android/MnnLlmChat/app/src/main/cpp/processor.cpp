#include "processor.h"

#include <algorithm>
#include <regex>
#include <utility>

#include "mls_log.h"
#include "video/video_processor.h"

namespace mls {
PromptProcessor::PromptProcessor(PromptProcessorConfig config)
    : config_(std::move(config)) {
    config_.video_processor_config.max_debug_images = config_.max_debug_images;
    config_.video_processor_config.save_first_image = config_.save_first_image;
    config_.video_processor_config.max_frames = std::max(config_.video_processor_config.max_frames,
                                                         config_.max_debug_images);
}

MNN::Express::VARP PromptProcessor::LoadImageFromPath(const std::string& image_path) {
    MNN_ERROR("Image loading not yet implemented for Android native path: %s", image_path.c_str());
    return nullptr;
}

std::string PromptProcessor::EscapeForRegex(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (char c : text) {
        switch (c) {
            case '.':
            case '^':
            case '$':
            case '|':
            case '(': 
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '*':
            case '+':
            case '?':
            case '\\':
                escaped.push_back('\\');
                break;
            default:
                break;
        }
        escaped.push_back(c);
    }
    return escaped;
}

PromptProcessingResult PromptProcessor::Process(const std::string& prompt_text) const {
    PromptProcessingResult result;
    result.multimodal_prompt.prompt_template = prompt_text;

    ProcessorState state;
    state.final_prompt = prompt_text;

    if (prompt_text.find("<img>") == std::string::npos && prompt_text.find("<video>") == std::string::npos) {
        return result;
    }

    bool has_images = false;
    bool has_videos = false;

    has_images = HandleImageTags(prompt_text, result, state);
    has_videos = HandleVideoTags(prompt_text, result, state);

    result.has_multimodal = (has_images || has_videos) && (state.successful_loads > 0);
    result.multimodal_prompt.prompt_template = state.final_prompt;

    if (result.has_multimodal) {
        MNN_DEBUG("Processed multimodal prompt with %zu images (%d successful, %d failed)",
                  result.multimodal_prompt.images.size(), state.successful_loads, state.failed_loads);
    } else if (state.failed_loads > 0) {
        MNN_DEBUG("All multimodal content failed to load, falling back to text-only mode");
    }

    return result;
}

bool PromptProcessor::HandleImageTags(const std::string& prompt_text,
                                          PromptProcessingResult& result,
                                          ProcessorState& state) const {
    bool has_images = false;

    std::regex img_regex("<img>([^<]*)</img>");
    std::smatch match;
    auto search_start = prompt_text.cbegin();

    while (std::regex_search(search_start, prompt_text.cend(), match, img_regex)) {
        if (state.image_index >= config_.max_debug_images) {
            MNN_DEBUG("Reached MAX_DEBUG_IMAGES limit (%d), skipping remaining images", config_.max_debug_images);
            break;
        }

        std::string image_path = match[1].str();
        if (!image_path.empty()) {
            MNN_DEBUG("Found image tag with path: %s", image_path.c_str());

            auto image_var = LoadImageFromPath(image_path);
            if (image_var.get() != nullptr) {
                std::string image_key = "image_" + std::to_string(state.image_index);
                MNN::Transformer::PromptImagePart image_part;
                image_part.image_data = image_var;
                image_part.width = 0;
                image_part.height = 0;
                result.multimodal_prompt.images[image_key] = image_part;
                has_images = true;
                state.image_index++;
                state.successful_loads++;
                MNN_DEBUG("Successfully loaded image: %s as %s", image_path.c_str(), image_key.c_str());
            } else {
                state.failed_loads++;
                MNN_ERROR("Failed to load image from path: %s", image_path.c_str());
                result.error_message += "Failed to load image: " + image_path + "; ";
                const std::string escaped_path = EscapeForRegex(image_path);
                state.final_prompt = std::regex_replace(
                    state.final_prompt,
                    std::regex("<img>" + escaped_path + "</img>"),
                    ""
                );
            }
        }
        search_start = match.suffix().first;
    }

    return has_images;
}

bool PromptProcessor::HandleVideoTags(const std::string& prompt_text,
                                          PromptProcessingResult& result,
                                          ProcessorState& state) const {
    bool has_videos = false;

    std::regex video_regex("<video>([^<]*)</video>");
    std::smatch match;
    auto search_start = prompt_text.cbegin();

    while (std::regex_search(search_start, prompt_text.cend(), match, video_regex)) {
        if (state.image_index >= config_.max_debug_images) {
            MNN_DEBUG("Reached MAX_DEBUG_IMAGES limit (%d), skipping video processing", config_.max_debug_images);
            break;
        }

        std::string video_path = match[1].str();
        if (!video_path.empty()) {
            MNN_DEBUG("Found video tag with path: %s", video_path.c_str());

            int remaining_slots = config_.max_debug_images - state.image_index;
            if (remaining_slots <= 0) {
                break;
            }

            VideoProcessorConfig video_config = config_.video_processor_config;
            video_config.max_frames = std::min(video_config.max_frames, remaining_slots);
            video_config.max_debug_images = std::min(video_config.max_debug_images, remaining_slots);

            auto video_result = VideoProcessor::ProcessVideoFrames(video_path, video_config);
            if (video_result.success && !video_result.images.empty()) {
                // Use the processed prompt template directly
                const std::string escaped_path = EscapeForRegex(video_path);
                state.final_prompt = std::regex_replace(
                    state.final_prompt,
                    std::regex("<video>" + escaped_path + "</video>"),
                    video_result.prompt_template
                );
                
                // Add images to multimodal_prompt
                for (const auto& pair : video_result.images) {
                    result.multimodal_prompt.images[pair.first] = pair.second;
                    state.image_index++;
                }
                
                has_videos = true;
                state.successful_loads++;
                MNN_DEBUG("Successfully processed video: %s into %zu frames with SmolVLM format",
                          video_path.c_str(), video_result.images.size());
            } else {
                state.failed_loads++;
                MNN_ERROR("Failed to process video from path: %s", video_path.c_str());
                result.error_message += "Failed to process video: " + video_path + "; ";
                const std::string escaped_path = EscapeForRegex(video_path);
                state.final_prompt = std::regex_replace(
                    state.final_prompt,
                    std::regex("<video>" + escaped_path + "</video>"),
                    ""
                );
            }
        }
        search_start = match.suffix().first;
    }

    return has_videos;
}

} // namespace mls
