//
// Created by ruoyi.sjd on 2025/4/18.
//

#include "video_processor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

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
std::vector<VideoFrame> VideoProcessor::ProcessVideo(
    const std::string& video_path) {
  MNN_DEBUG("VideoProcessor: Starting video processing pipeline for: %s",
            video_path.c_str());
  
  std::vector<VideoFrame> processed_frames;
  
  // Step 1: Extract raw frames
  auto raw_frames = ExtractFrames(video_path);
  if (raw_frames.empty()) {
    MNN_ERROR("VideoProcessor: No frames extracted from: %s",
               video_path.c_str());
    return processed_frames;
  }
  
  MNN_DEBUG("VideoProcessor: Extracted %zu raw frames", raw_frames.size());
  
  // Step 2: Preprocess frames
  auto preprocessed_frames = PreprocessFrames(raw_frames);
  MNN_DEBUG("VideoProcessor: Preprocessed %zu frames",
            preprocessed_frames.size());
  
  processed_frames = std::move(preprocessed_frames);
  
  MNN_DEBUG("VideoProcessor: Completed processing, returning %zu frames",
            processed_frames.size());
  return processed_frames;
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
                             const char* strategy, int width, int height) {
      debug_callback_(tensor, width, height, 0, pts);  // frame_index not available here
    };
  }
  
  int frames_decoded = decoder_->DecodeWithFps(config_.max_frames, config_.fps,
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

    MNN_DEBUG("VideoProcessor: Created VideoFrame %d, pts=%ld, size=%dx%d",
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
  
  MNN_DEBUG("CreateTensorFromRgb: creating tensor for %dx%d RGB data", width, height);
  
  // Log first few RGB values for debugging
  if (width * height >= 4) {
    MNN_DEBUG("CreateTensorFromRgb: input RGB values: R=%d,G=%d,B=%d, "
              "R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, R=%d,G=%d,B=%d",
              rgb_data[0], rgb_data[1], rgb_data[2], rgb_data[3],
              rgb_data[4], rgb_data[5], rgb_data[6], rgb_data[7],
              rgb_data[8], rgb_data[9], rgb_data[10], rgb_data[11]);
  }
  
  // Create tensor with HWC layout: {height, width, channels}
  auto var = MNN::Express::_Input({height, width, 3}, MNN::Express::NHWC,
                                  halide_type_of<uint8_t>());
  auto ptr = var->writeMap<uint8_t>();
  
  // Direct memcpy should work for continuous RGB data
  memcpy(ptr, rgb_data, width * height * 3);
  
  MNN_DEBUG("CreateTensorFromRgb: tensor created successfully, size=%dx%dx3",
            height, width);
  
  // Verify the tensor data by reading back a few values
  auto verify_ptr = var->readMap<uint8_t>();
  if (verify_ptr && width * height >= 4) {
    MNN_DEBUG("CreateTensorFromRgb: tensor verification - first RGB values: "
              "R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, "
              "R=%d,G=%d,B=%d",
              verify_ptr[0], verify_ptr[1], verify_ptr[2], verify_ptr[3],
              verify_ptr[4], verify_ptr[5], verify_ptr[6], verify_ptr[7],
              verify_ptr[8], verify_ptr[9], verify_ptr[10], verify_ptr[11]);
  }
  
  return var;
}

// Convenience static helper
std::vector<MNN::Express::VARP> VideoProcessor::ProcessVideoFrames(
    const std::string& video_path,
    const VideoProcessorConfig& config) {
  std::vector<MNN::Express::VARP> images;
  VideoProcessorConfig processor_config = config;
  
  MNN_DEBUG("Using VideoProcessor (Hugging Face style) for: %s",
            video_path.c_str());
  
  // Create VideoProcessor with provided configuration
  VideoProcessor processor(processor_config);
  
  // Process video through the complete pipeline
  auto processed_frames = processor.ProcessVideo(video_path);
  
  if (!processed_frames.empty()) {
    MNN_DEBUG("VideoProcessor: Successfully processed %zu frames",
              processed_frames.size());
    size_t max_frames_to_return = std::min(
        processed_frames.size(),
        static_cast<size_t>(processor_config.max_debug_images));
    
    for (size_t i = 0; i < max_frames_to_return; ++i) {
      if (processed_frames[i].pixel_values.get() != nullptr) {
        images.push_back(processed_frames[i].pixel_values);
      }
    }
    
    MNN_DEBUG("VideoProcessor: Converted %zu frames to VARP format "
              "(limited to %zu for debugging)",
              processed_frames.size(), max_frames_to_return);
  } else {
    MNN_ERROR("VideoProcessor: No frames processed from: %s",
               video_path.c_str());
  }
  
  MNN_DEBUG("Total frames processed: %zu", images.size());
  return images;
}

} // namespace mls
