#ifndef VIDEO_DECODER_HPP_
#define VIDEO_DECODER_HPP_

#include <unistd.h>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

#include "image_utils.hpp"

// Enum to specify data source for GetVideoMetadata
enum class MetadataSource {
  MEDIA_EXTRACTOR,  // Get metadata from media extractor (track format)
  MEDIA_CODEC       // Get metadata from media codec (output format)
};

// Video metadata structure for improved sampling
struct VideoMetadata {
  int64_t total_frames = 0;      // Total number of frames
  float native_fps = -1.0f;      // Original frame rate
  int64_t duration_us = 0;       // Video duration in microseconds
  int64_t frame_interval_us = 0; // Frame interval in microseconds
  int64_t start_time_us = 0;     // Video start time offset in microseconds
  
  // Video dimensions and format information
  int width = 0;                 // Video width
  int height = 0;                // Video height
  std::string mime_type;         // Video MIME type
  
  // Output format information (from AMediaCodec_getOutputFormat)
  int32_t color_format = 0;      // Output color format
  int32_t stride = 0;            // Output stride
  int32_t slice_height = 0;      // Output slice height
  int32_t crop_left = 0;         // Crop rectangle left
  int32_t crop_top = 0;          // Crop rectangle top
  int32_t crop_right = 0;        // Crop rectangle right
  int32_t crop_bottom = 0;       // Crop rectangle bottom
  
  // Output dimensions (after cropping)
  int output_width = 0;          // Final output width
  int output_height = 0;         // Final output height
  
  // YUV format information
  ImageUtils::YUVFormatInfo yuv_format_info; // YUV format details
  
  // Status flags
  bool format_info_ready = false; // Whether output format info is available
  bool metadata_complete = false; // Whether all metadata is available
};

namespace MNN {
namespace Express {
class VARP;
} // namespace Express
} // namespace MNN

namespace mls {
struct VideoProcessorConfig;
} // namespace mls

class VideoDecoder {
 public:
  // Debug callback receives MNN tensor ready for inspection/saving.
  using FrameDebugCallback = std::function<void(
      MNN::Express::VARP tensor,
      int64_t pts,
      long native_ms,
      int64_t target_us,
      int width,
      int height)>;

  VideoDecoder();
  virtual ~VideoDecoder();

  static VideoDecoder* CreateFromFd(int fd, off64_t offset, off64_t length);
  static VideoDecoder* CreateByteBufferDecoder(int fd, off64_t offset, off64_t length);

  bool OpenFromFd(int fd, off64_t offset, off64_t length);
  bool SelectVideoTrack();
  virtual bool Configure() = 0;

  // Decode with VideoProcessorConfig
  int DecodeWithConfig(const mls::VideoProcessorConfig& config,
                      std::vector<MNN::Express::VARP>* out_tensors,
                      std::vector<int64_t>* out_timestamps,
                      FrameDebugCallback callback = nullptr);

  bool GetVideoMetadata(VideoMetadata* metadata);

  bool CalculateRealFps(VideoMetadata* metadata, int sample_frames = 10);

  virtual float GetDetectedFps() const { return -1.0f; }

  // Helper function to determine if we should capture the current frame based on target_fps
  bool ShouldCaptureFrame(int64_t current_pts_us, int64_t last_captured_pts_us, float target_fps);

  // Helper function to determine if we should capture the current frame based on frame index (Video-style)
  bool ShouldCaptureFrameByIndex(int64_t current_frame_index, const std::vector<int64_t>& sample_indices);

  // Calculate sample indices using Video-style sampling (similar to video_sample_indices_fn)
  std::vector<int64_t> CalculateVideoSampleIndices(const VideoMetadata& metadata, 
                                                     int max_frames, 
                                                     float target_fps,
                                                     float skip_secs = 0.0f);

  // Get next available frame from codec (simplified interface)
  // Pure virtual - must be implemented by subclasses
  virtual bool GetNextFrame(std::vector<uint8_t>* out_yuv,
                            ImageUtils::YUVFormatInfo* format_info,
                            int64_t* out_pts_us,
                            long* native_ms,
                            bool* out_eos) = 0;

  // Convert YUV data to RGB (for ByteBufferDecoder)
  // Default implementation does nothing for compatibility
  virtual bool ConvertYuvToRgb(const std::vector<uint8_t>& yuv_data,
                               const ImageUtils::YUVFormatInfo& format_info,
                               std::vector<uint8_t>* out_rgb) {
    // Default implementation: assume input is already RGB
    if (out_rgb) {
      *out_rgb = yuv_data;
    }
    return true;
  }

  int width() const { return video_width_; }
  int height() const { return video_height_; }
  const std::string& mime_type() const { return mime_type_; }

 protected:
  void Teardown();
  void LogCodecName(const char* prefix);
  bool StepFeedInput(int mode, int64_t target_timestamp_us, bool* saw_input_eos);
  
  // Helper function to extract output format information from codec
  bool ExtractOutputFormatInfo(VideoMetadata* metadata);

  AMediaExtractor* media_extractor_ = nullptr;
  AMediaCodec* media_codec_ = nullptr;
  
  // Unified video metadata - replaces individual member variables
  VideoMetadata video_metadata_;
  
  // Legacy members for backward compatibility (will be removed)
  int video_width_ = 0;
  int video_height_ = 0;
  std::string mime_type_;
  float native_fps_ = -1.0f;
  int64_t frame_interval_us_ = -1;
};

#endif  // VIDEO_DECODER_HPP_
