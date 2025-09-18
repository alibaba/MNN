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
  
  int frames_extracted = 0;
  int64_t next_target_us = 0;
  bool saw_eos = false;
  const int64_t step_us = static_cast<int64_t>(1000000.0f / config_.fps);

  MNN_DEBUG("VideoProcessor: Starting frame extraction with decoder instance");

  int consecutive_failures = 0;
  while (frames_extracted < config_.max_frames && !saw_eos) {
    std::vector<uint8_t> rgb_data;
    int64_t pts_us = 0;
    if (!decoder_->DecodeFrame(next_target_us, &rgb_data, &pts_us, nullptr,
                               &saw_eos)) {
      if (saw_eos) {
        break;
      }
      if (++consecutive_failures > 50) {
        MNN_ERROR("VideoProcessor: Too many consecutive decode failures "
                  "at frame %d", frames_extracted);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    consecutive_failures = 0;
    if (rgb_data.empty()) {
      next_target_us += step_us;
      continue;
    }

    if (debug_callback_) {
      debug_callback_(rgb_data, decoder_->width(), decoder_->height(),
                      frames_extracted, pts_us);
    }

    auto pixel_values = RawRgbToVar(rgb_data.data(), decoder_->width(),
                                    decoder_->height());
    if (pixel_values.get() == nullptr) {
      MNN_ERROR("VideoProcessor: Failed to create MNN tensor from RGB data");
      next_target_us += step_us;
      continue;
    }

    VideoFrame frame;
    frame.pixel_values = pixel_values;
    frame.timestamp_us = pts_us;
    frame.frame_index = frames_extracted;
    frame.width = decoder_->width();
    frame.height = decoder_->height();
    frames.push_back(frame);

    MNN_DEBUG("VideoProcessor: Extracted frame %d, pts=%ld, size=%dx%d",
              frame.frame_index, pts_us, frame.width, frame.height);

    ++frames_extracted;
    next_target_us += step_us;
  }

  MNN_DEBUG("VideoProcessor: Extracted %d frames from video",
            frames_extracted);
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
MNN::Express::VARP RawRgbToVar(const uint8_t* rgb_data,
                               int width,
                               int height) {
  if (!rgb_data || width <= 0 || height <= 0) {
    MNN_ERROR("Invalid RGB data or dimensions: %dx%d", width, height);
    return nullptr;
  }
  
  MNN_DEBUG("RawRgbToVar: creating tensor for %dx%d RGB data", width, height);
  
  // Log first few RGB values for debugging
  if (width * height >= 4) {
    MNN_DEBUG("RawRgbToVar: input RGB values: R=%d,G=%d,B=%d, "
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
  
  MNN_DEBUG("RawRgbToVar: tensor created successfully, size=%dx%dx3",
            height, width);
  
  // Verify the tensor data by reading back a few values
  auto verify_ptr = var->readMap<uint8_t>();
  if (verify_ptr && width * height >= 4) {
    MNN_DEBUG("RawRgbToVar: tensor verification - first RGB values: "
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
    
    // Convert VideoFrames to MNN::Express::VARP for compatibility
    // Apply debug limit
    size_t max_frames_to_return = std::min(
        processed_frames.size(),
        static_cast<size_t>(processor_config.max_debug_images));
    
    for (size_t i = 0; i < max_frames_to_return; ++i) {
      if (processed_frames[i].pixel_values.get() != nullptr) {
        images.push_back(processed_frames[i].pixel_values);
        
        // Save first frame for debugging if enabled
        if (processor_config.save_first_image && i == 0) {
          // Try multiple possible paths for saving the debug image
          std::vector<std::string> debug_paths = {
              processor_config.debug_output_path +
                  "debug_first_video_frame.jpg",
              "/data/data/com.alibaba.mnnllm.android/files/"
                  "debug_first_video_frame.jpg",
              "/sdcard/Android/data/com.alibaba.mnnllm.android/files/"
                  "debug_first_video_frame.jpg",
              "/tmp/debug_first_video_frame.jpg"
          };
          
          bool saved = false;
          for (const auto& debug_path : debug_paths) {
            MNN_DEBUG("Attempting to save first video frame to: %s",
                      debug_path.c_str());
            
            // Check if we can write to this path
            FILE* test_file = fopen(debug_path.c_str(), "w");
            if (test_file) {
              fclose(test_file);
              // Remove the test file
              remove(debug_path.c_str());
              
              // Debug: Log tensor information before saving
              auto tensor = processed_frames[i].pixel_values;
              if (tensor.get()) {
                auto info = tensor->getInfo();
                if (info) {
                  auto dims = info->dim;
                  MNN_DEBUG("Debug: Tensor dimensions: [%s]",
                            [&dims]() {
                              std::string dim_str;
                              for (size_t j = 0; j < dims.size(); j++) {
                                if (j > 0) dim_str += ", ";
                                dim_str += std::to_string(dims[j]);
                              }
                              return dim_str;
                            }().c_str());
                  
                  // Try to save with detailed debugging
                  if (ImageUtils::SaveTensorAsJPG(tensor,
                                                  debug_path.c_str())) {
                    MNN_DEBUG("First video frame saved for debugging: %s",
                              debug_path.c_str());
                    saved = true;
                    break;
                  } else {
                    MNN_ERROR("Failed to save first video frame to: %s",
                               debug_path.c_str());
                    
                    // Try alternative approach: save raw RGB data
                    std::vector<uint8_t> rgb_data;
                    int height = dims[0], width = dims[1],
                        channels = dims[2];
                    MNN_DEBUG("Debug: Trying to convert tensor to RGB: "
                              "%dx%dx%d", height, width, channels);
                    
                    if (ImageUtils::TensorToRgb(tensor, height, width,
                                                channels, rgb_data)) {
                      MNN_DEBUG("Debug: Successfully converted tensor to RGB, "
                                "size: %zu", rgb_data.size());
                                            
                      // Save raw RGB data as BMP for debugging
                      std::string bmp_path = debug_path + ".bmp";
                      if (ImageUtils::SaveAsBMP(rgb_data.data(), width,
                                                height, bmp_path.c_str())) {
                        MNN_DEBUG("Debug: Saved raw RGB data as BMP: %s",
                                  bmp_path.c_str());
                      } else {
                        MNN_ERROR("Debug: Failed to save raw RGB data as BMP");
                      }
                    } else {
                      MNN_ERROR("Debug: Failed to convert tensor to RGB");
                    }
                  }
                } else {
                  MNN_ERROR("Debug: Failed to get tensor info");
                }
              } else {
                MNN_ERROR("Debug: Tensor is null");
              }
            } else {
              MNN_DEBUG("Cannot write to path: %s (permission denied or "
                        "path doesn't exist)", debug_path.c_str());
            }
          }
          
          if (!saved) {
            MNN_ERROR("Failed to save first video frame to any available path");
            
            // Try to save with a simpler approach - convert tensor to RGB
            // and save directly
            auto tensor = processed_frames[i].pixel_values;
            if (tensor.get()) {
              auto info = tensor->getInfo();
              if (info) {
                auto dims = info->dim;
                int height = dims[0], width = dims[1], channels = dims[2];
                
                std::vector<uint8_t> rgb_data;
                if (ImageUtils::TensorToRgb(tensor, height, width,
                                              channels, rgb_data)) {
                  // Try to save to a simple path
                  std::string simple_path = "/tmp/first_frame.jpg";
                  if (ImageUtils::SaveAsJPEG(rgb_data.data(), width, height,
                                             simple_path.c_str())) {
                    MNN_DEBUG("First video frame saved with simple approach: %s",
                              simple_path.c_str());
                  } else {
                    MNN_ERROR("Even simple save approach failed");
                  }
                }
              }
            }
          }
        }
        
        // NEW: Save original YUV data directly as RGB for comparison
        if (processor_config.save_first_image && i == 0) {
          MNN_DEBUG("Attempting to save original YUV data directly as RGB "
                    "for comparison");
          
          // Get the original YUV data from the first frame
          // We need to access the original YUV data that was used to create
          // this tensor. This is a bit tricky since we don't store the
          // original YUV data. Let's try to reconstruct it or find another way.
          
          // For now, let's try to save the tensor data directly as RGB
          auto tensor = processed_frames[i].pixel_values;
          if (tensor.get()) {
            auto info = tensor->getInfo();
            if (info) {
              auto dims = info->dim;
              int height = dims[0], width = dims[1], channels = dims[2];
              
              // Convert tensor to RGB data
              std::vector<uint8_t> rgb_data;
              if (ImageUtils::TensorToRgb(tensor, height, width, channels,
                                          rgb_data)) {
                MNN_DEBUG("Direct YUV->RGB conversion successful, size: %zu",
                          rgb_data.size());
                
                // Try multiple paths for the direct RGB save
                std::vector<std::string> direct_paths = {
                    processor_config.debug_output_path +
                        "direct_yuv_to_rgb.jpg",
                    "/data/data/com.alibaba.mnnllm.android/files/"
                        "direct_yuv_to_rgb.jpg",
                    "/sdcard/Android/data/com.alibaba.mnnllm.android/files/"
                        "direct_yuv_to_rgb.jpg",
                    "/tmp/direct_yuv_to_rgb.jpg"
                };
                
                bool direct_saved = false;
                for (const auto& direct_path : direct_paths) {
                  MNN_DEBUG("Attempting to save direct YUV->RGB to: %s",
                            direct_path.c_str());
                  
                  // Check if we can write to this path
                  FILE* test_file = fopen(direct_path.c_str(), "w");
                  if (test_file) {
                    fclose(test_file);
                    remove(direct_path.c_str());
                    
                    if (ImageUtils::SaveAsJPEG(rgb_data.data(), width, height,
                                               direct_path.c_str())) {
                      MNN_DEBUG("Direct YUV->RGB saved successfully: %s",
                                direct_path.c_str());
                      direct_saved = true;
                      break;
                    } else {
                      MNN_ERROR("Failed to save direct YUV->RGB to: %s",
                                 direct_path.c_str());
                    }
                  } else {
                    MNN_DEBUG("Cannot write to direct path: %s",
                               direct_path.c_str());
                  }
                }
                
                if (!direct_saved) {
                  MNN_ERROR("Failed to save direct YUV->RGB to any path");
                }
              } else {
                MNN_ERROR("Failed to convert tensor to RGB for direct save");
              }
            }
          }
        }
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
