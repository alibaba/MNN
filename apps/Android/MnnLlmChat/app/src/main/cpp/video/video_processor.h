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

#ifdef __ANDROID__
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#endif

#include "MNN/expr/Expr.hpp"
#include "image_utils.hpp"
#include "video_decoder.hpp"

namespace mls {

// Video Processor Configuration (similar to Hugging Face Transformers)
struct VideoProcessorConfig {
  // Frame extraction settings
  int max_frames = 50;  // Maximum number of frames to extract
  float fps = 5.0f;     // Target FPS for frame extraction
  // Preprocessing settings
  int target_width = 224;   // Target width for resizing
  int target_height = 224;  // Target height for resizing
  
  // Debug settings
  int max_debug_images = 9999;  // Limit number of images for debugging
  bool save_first_image = true;  // Save first image to local path for debugging
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

// Video Processor Class (similar to Hugging Face VideoProcessor)
class VideoProcessor {
 public:
  using DebugFrameCallback = std::function<void(
      const std::vector<uint8_t>& rgb,
      int width,
      int height,
      int frame_index,
      int64_t pts)>;

  explicit VideoProcessor(
      const VideoProcessorConfig& config = VideoProcessorConfig{});

  // Main processing pipeline
  std::vector<VideoFrame> ProcessVideo(const std::string& video_path);

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

  // Convenience helper: run full pipeline and return tensors
  static std::vector<MNN::Express::VARP> ProcessVideoFrames(
      const std::string& video_path,
      const VideoProcessorConfig& config);

 private:
  VideoProcessorConfig config_;
  std::unique_ptr<VideoDecoder> decoder_;
  DebugFrameCallback debug_callback_;
};

// Convert raw RGB data to MNN::Express::VARP
MNN::Express::VARP RawRgbToVar(const uint8_t* rgb_data,
                               int width,
                               int height);

} // namespace mls

#endif  // VIDEO_PROCESSOR_H_
