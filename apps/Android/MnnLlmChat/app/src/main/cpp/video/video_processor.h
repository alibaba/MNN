//
// Created by ruoyi.sjd on 2025/4/18.
//

#ifndef VIDEO_PROCESSOR_H_
#define VIDEO_PROCESSOR_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>

#ifdef __ANDROID__
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#endif
#include "llm/llm.hpp"
#include "MNN/expr/Expr.hpp"
#include "image_utils.hpp"
#include "video_decoder.hpp"

namespace mls {

// Video Processor Configuration (similar to Hugging Face Transformers)
// we use the smolvlm config for default.
struct VideoProcessorConfig {
  // Frame extraction settings
  int max_frames = 64;  // Maximum number of frames to extract
  float fps = 1.0f;     // Target FPS for frame extraction
  float skip_secs = 1.0f;  // Number of seconds to skip from start and end
  // Preprocessing settings
  int target_width = 512;   // Target width for resizing
  int target_height = 512;  // Target height for resizing
  
  // Debug settings
  int max_debug_images = 9999;  // Limit number of images for debugging
  bool save_first_image = false;  // Save first image to local path for debugging
  std::string debug_output_path = 
      "/data/data/com.alibaba.mnnllm.android/files/";  // Debug output directory
};

// Video Frame Structure
struct VideoFrame {
  MNN::Express::VARP pixel_values;  // Raw pixel values as tensor
  int64_t timestamp_us;             // Timestamp in microseconds
  int frame_index;                  // Frame index in sequence
  int width, height;                // Frame dimensions
};

// Video Processing Result Structure
struct VideoProcessingResult {
  std::string prompt_template;              // Processed prompt template with SmolVLM format
  std::map<std::string, MNN::Transformer::PromptImagePart> images;  // Image data for multimodal_prompt
  VideoMetadata metadata;                   // Video metadata
  bool success;                            // Processing success flag
};

// Video Processor Class (similar to Hugging Face VideoProcessor)
class VideoProcessor {
 public:
  using DebugFrameCallback = std::function<void(
      MNN::Express::VARP tensor,
      int width,
      int height,
      int frame_index,
      int64_t pts)>;

  explicit VideoProcessor(
      const VideoProcessorConfig& config = VideoProcessorConfig{});

  // Main processing pipeline
  VideoProcessingResult ProcessVideo(const std::string& video_path);

  // Optional debug callback for accessing raw RGB frames
  void SetDebugCallback(DebugFrameCallback callback);

  // Extract frames from video file
  std::vector<VideoFrame> ExtractFrames(const std::string& video_path);
  
  // Preprocess frames (resize, etc.)
  std::vector<VideoFrame> PreprocessFrames(
      const std::vector<VideoFrame>& frames);
  
  // Resize frame tensor
  MNN::Express::VARP ResizeFrame(MNN::Express::VARP input,
                                 int src_width,
                                 int src_height,
                                 int dst_width,
                                 int dst_height);
  
  // Get configuration
  const VideoProcessorConfig& GetConfig() const;
  
  // Update configuration
  void UpdateConfig(const VideoProcessorConfig& config);

  // Convenience helper: run full pipeline and return processed prompt with images
  static VideoProcessingResult ProcessVideoFrames(
      const std::string& video_path,
      const VideoProcessorConfig& config);

  // Generate SmolVLM format video description
  static std::string GenerateSmolVLMVideoDescription(
      const std::vector<VideoFrame>& video_frames,
      const VideoMetadata& metadata,
      int start_image_index);

 private:
  VideoProcessorConfig config_;
  std::unique_ptr<VideoDecoder> decoder_;
  DebugFrameCallback debug_callback_;
};

// Convert raw RGB data to MNN::Express::VARP
MNN::Express::VARP CreateTensorFromRgb(const uint8_t* rgb_data,
                                      int width,
                                      int height);

} // namespace mls

#endif  // VIDEO_PROCESSOR_H_
