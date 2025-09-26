//
// Created by ruoyi.sjd on 2025/4/18.
//

#include "video_processor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>
#include <inttypes.h>
#include "llm/llm.hpp"
#include "../mls_log.h"
#include "MNN/MNNDefine.h"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/MathOp.hpp"

namespace {

bool IsValidVideoFile(const char* path) {
  if (!path || std::strlen(path) == 0) {
    return false;
  }

  if (access(path, R_OK) != 0) {
    return false;
  }

  struct stat st {};
  if (stat(path, &st) != 0) {
    return false;
  }

  return st.st_size > 1024;
}

}  // namespace

namespace mls {

VideoProcessor::VideoProcessor(const VideoProcessorConfig& config)
    : config_(config) {}

void VideoProcessor::SetDebugCallback(DebugFrameCallback callback) {
  debug_callback_ = std::move(callback);
}

// Main processing pipeline
VideoProcessingResult VideoProcessor::ProcessVideo(
    const std::string& video_path) {
  MNN_DEBUG("VideoProcessor: Starting video processing pipeline for: %s",
            video_path.c_str());
  
  VideoProcessingResult result;
  result.success = false;
  
  // Step 1: Extract raw frames
  auto raw_frames = ExtractFrames(video_path);
  if (raw_frames.empty()) {
    MNN_ERROR("VideoProcessor: No frames extracted from: %s",
               video_path.c_str());
    return result;
  }
  MNN_DEBUG("VideoProcessor: Extracted %zu raw frames", raw_frames.size());
  // Step 2: Preprocess frames
  auto preprocessed_frames = PreprocessFrames(raw_frames);
  MNN_DEBUG("VideoProcessor: Preprocessed %zu frames",
            preprocessed_frames.size());
  
  if (preprocessed_frames.empty()) {
    MNN_ERROR("VideoProcessor: No frames after preprocessing");
    return result;
  }
  
  for (size_t i = 0; i < preprocessed_frames.size(); ++i) {
    std::string frame_key = "video_frame_" + std::to_string(i);
    MNN::Transformer::PromptImagePart image_part;
    image_part.image_data = preprocessed_frames[i].pixel_values;
    image_part.width = preprocessed_frames[i].width;
    image_part.height = preprocessed_frames[i].height;
    result.images[frame_key] = image_part;
  }
  
  result.prompt_template = GenerateSmolVLMVideoDescription(
    preprocessed_frames, result.metadata, 0);
  
  result.success = true;
  MNN_DEBUG("VideoProcessor: Completed processing, returning %zu frames",
    preprocessed_frames.size());
  return result;
}

// Extract frames from video file
std::vector<VideoFrame> VideoProcessor::ExtractFrames(
    const std::string& video_path) {
  std::vector<VideoFrame> frames;
  
  // Validate video file
  if (!IsValidVideoFile(video_path.c_str())) {
    MNN_ERROR("VideoProcessor: Invalid video file: %s", video_path.c_str());
    return frames;
  }
  
  // Open video file
  int fd = open(video_path.c_str(), O_RDONLY);
  if (fd < 0) {
    MNN_ERROR("VideoProcessor: Failed to open video file: %s",
               video_path.c_str());
    return frames;
  }
  
  // Get file size
  struct stat st;
  if (fstat(fd, &st) != 0) {
    MNN_ERROR("VideoProcessor: Failed to get file stats: %s",
               video_path.c_str());
    close(fd);
    return frames;
  }
  
  // Create decoder
  decoder_.reset(VideoDecoder::CreateByteBufferDecoder(fd, 0, st.st_size));
  close(fd);
  
  if (!decoder_) {
    MNN_ERROR("VideoProcessor: Failed to create video decoder");
    return frames;
  }
  
  // Select video track
  if (!decoder_->SelectVideoTrack()) {
    MNN_ERROR("VideoProcessor: No video track found");
    return frames;
  }
  
  // Configure decoder
  if (!decoder_->Configure()) {
    MNN_ERROR("VideoProcessor: Failed to configure decoder");
    return frames;
  }
  
  MNN_DEBUG("VideoProcessor: Decoder configured, dimensions: %dx%d",
             decoder_->width(), decoder_->height());
  
  // Use VideoDecoder's DecodeWithFps method - now returns tensors directly
  std::vector<MNN::Express::VARP> tensors;
  std::vector<int64_t> timestamps;
  
  // Create debug callback wrapper
  VideoDecoder::FrameDebugCallback decoder_callback = nullptr;
  if (debug_callback_) {
    decoder_callback = [this](MNN::Express::VARP tensor,
                             int64_t pts, long native_ms, int64_t target_us,
                             int width, int height) {
      debug_callback_(tensor, width, height, 0, pts);  // frame_index not available here
    };
  }
  
  int frames_decoded = decoder_->DecodeWithConfig(config_,
                                                  &tensors, &timestamps,
                                                  decoder_callback);
  
  if (frames_decoded <= 0) {
    MNN_ERROR("VideoProcessor: Failed to decode frames");
    return frames;
  }
  
  MNN_DEBUG("VideoProcessor: Decoded %d frames, creating VideoFrames",
            frames_decoded);
  
  // Create VideoFrame objects directly from tensors
  for (int i = 0; i < frames_decoded; ++i) {
    VideoFrame frame;
    frame.pixel_values = tensors[i];
    frame.timestamp_us = timestamps[i];
    frame.frame_index = i;
    frame.width = decoder_->width();
    frame.height = decoder_->height();
    frames.push_back(frame);

    MNN_DEBUG("VideoProcessor: Created VideoFrame %d, pts=%" PRId64 ", size=%dx%d",
              frame.frame_index, frame.timestamp_us, frame.width, frame.height);
  }

  MNN_DEBUG("VideoProcessor: Successfully extracted %zu frames from video",
            frames.size());
  return frames;
}

// Preprocess frames (resize, etc.)
std::vector<VideoFrame> VideoProcessor::PreprocessFrames(
    const std::vector<VideoFrame>& frames) {
  std::vector<VideoFrame> processed_frames;
  
  for (const auto& frame : frames) {
    VideoFrame processed_frame = frame;
    
    // Resize frame if needed
    if (frame.width != config_.target_width ||
        frame.height != config_.target_height) {
      processed_frame.pixel_values = ResizeFrame(
          frame.pixel_values,
          frame.width,
          frame.height,
          config_.target_width,
          config_.target_height);
      processed_frame.width = config_.target_width;
      processed_frame.height = config_.target_height;
    }
    processed_frames.push_back(processed_frame);
  }
  
  return processed_frames;
}


// Resize frame tensor
MNN::Express::VARP VideoProcessor::ResizeFrame(
    MNN::Express::VARP input,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height) {
  // Simple nearest neighbor resize (placeholder)
  // In a real implementation, this would use proper interpolation
  MNN_DEBUG("VideoProcessor: Resizing frame from %dx%d to %dx%d",
             src_width, src_height, dst_width, dst_height);
  return input;
}



// Get configuration
const VideoProcessorConfig& VideoProcessor::GetConfig() const {
  return config_;
}

// Update configuration
void VideoProcessor::UpdateConfig(const VideoProcessorConfig& config) {
  config_ = config;
}

// Convert raw RGB data to MNN::Express::VARP
MNN::Express::VARP CreateTensorFromRgb(const uint8_t* rgb_data,
                                       int width,
                                       int height) {
  if (!rgb_data || width <= 0 || height <= 0) {
    MNN_ERROR("Invalid RGB data or dimensions: %dx%d", width, height);
    return nullptr;
  }
  // Create tensor with HWC layout: {height, width, channels}
  auto var = MNN::Express::_Input({height, width, 3}, MNN::Express::NHWC,
                                  halide_type_of<uint8_t>());
  auto ptr = var->writeMap<uint8_t>();
  memcpy(ptr, rgb_data, width * height * 3);
  MNN_DEBUG("CreateTensorFromRgb: tensor created successfully, size=%dx%dx3",
            height, width);
  return var;
}

// Convenience static helper
VideoProcessingResult VideoProcessor::ProcessVideoFrames(
    const std::string& video_path,
    const VideoProcessorConfig& config) {
  MNN_DEBUG("Using VideoProcessor (Hugging Face style) for: %s",
            video_path.c_str());
  
  // Create VideoProcessor with provided configuration
  VideoProcessor processor(config);
  
  // Process video through the complete pipeline
  return processor.ProcessVideo(video_path);
}

// Generate SmolVLM format video description
std::string VideoProcessor::GenerateSmolVLMVideoDescription(
    const std::vector<VideoFrame>& video_frames,
    const VideoMetadata& metadata,
    int start_image_index) {
  
  std::string description;
  
  // Calculate video duration in seconds
  int64_t duration_seconds = metadata.duration_us / 1000000;
  int hours = duration_seconds / 3600;
  int minutes = (duration_seconds % 3600) / 60;
  int seconds = duration_seconds % 60;
  
  // Generate title in SmolVLM format
  char duration_str[32];
  snprintf(duration_str, sizeof(duration_str), "%d:%02d:%02d", hours, minutes, seconds);
  
  description += "You are provided the following series of " + 
                 std::to_string(video_frames.size()) + 
                 " frames from a " + duration_str + " [H:MM:SS] video.\n\n";
  
  // Generate frame descriptions using MNN compatible <img> format
  for (size_t i = 0; i < video_frames.size(); ++i) {
    // Use actual timestamp from the frame instead of estimating
    int64_t timestamp_us = video_frames[i].timestamp_us;
    int frame_seconds = timestamp_us / 1000000;
    int frame_minutes = frame_seconds / 60;
    int frame_secs = frame_seconds % 60;
    
    char timestamp_str[16];
    snprintf(timestamp_str, sizeof(timestamp_str), "%02d:%02d", frame_minutes, frame_secs);
    
    // Debug logging to verify timestamp calculation
    MNN_DEBUG("GenerateSmolVLMVideoDescription: Frame %zu - timestamp_us=%" PRId64 ", formatted=%s", 
              i, timestamp_us, timestamp_str);
    
    // Generate frame description (MNN compatible format)
    std::string frame_key = "video_frame_" + std::to_string(start_image_index + i);
    description += "Frame from " + std::string(timestamp_str) + ":";
    description += "<img>" + frame_key + "</img>\n";
  }
  
  MNN_DEBUG("Generated SmolVLM video description with %zu frames", video_frames.size());
  return description;
}

} // namespace mls

