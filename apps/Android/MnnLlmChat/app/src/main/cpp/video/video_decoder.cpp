#include "video_decoder.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <inttypes.h>
#include <string>
#include <vector>

#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"

#include "../mls_log.h"
#include "byte_buffer_decoder.hpp"
#include "video_utils.hpp"

// Forward declaration for tensor creation
namespace mls {
MNN::Express::VARP CreateTensorFromRgb(const uint8_t* rgb_data, int width, int height);
}

#define TAG "VideoDecoder"

VideoDecoder::VideoDecoder() = default;

VideoDecoder::~VideoDecoder() { Teardown(); }

VideoDecoder* VideoDecoder::CreateFromFd(int fd, off64_t offset, off64_t length) {
  VIDEO_LOGV(TAG,"CreateFromFd: fd=%d, offset=%lld, length=%lld", fd,
       (long long)offset, (long long)length);
  VideoDecoder* decoder = new ByteBufferDecoder();
  if (!decoder->OpenFromFd(fd, offset, length)) {
    VIDEO_LOGE(TAG,"CreateFromFd: OpenFromFd failed");
    delete decoder;
    return nullptr;
  }
  VIDEO_LOGV(TAG,"CreateFromFd: success");
  return decoder;
}

VideoDecoder* VideoDecoder::CreateByteBufferDecoder(int fd, off64_t offset, off64_t length) {
  VIDEO_LOGV(TAG,"CreateByteBufferDecoder: fd=%d, offset=%lld, length=%lld", fd,
       (long long)offset, (long long)length);
  VideoDecoder* decoder = new ByteBufferDecoder();
  if (!decoder->OpenFromFd(fd, offset, length)) {
    VIDEO_LOGE(TAG,"CreateByteBufferDecoder: OpenFromFd failed");
    delete decoder;
    return nullptr;
  }
  VIDEO_LOGV(TAG,"CreateByteBufferDecoder: success");
  return decoder;
}


bool VideoDecoder::OpenFromFd(int fd, off64_t offset, off64_t length) {
  Teardown();

  media_extractor_ = AMediaExtractor_new();
  if (!media_extractor_) {
    VIDEO_LOGE(TAG,"OpenFromFd: AMediaExtractor_new() failed");
    return false;
  }

  media_status_t result = AMediaExtractor_setDataSourceFd(media_extractor_,
                                                          fd, offset, length);
  if (result != AMEDIA_OK) {
    VIDEO_LOGE(TAG,"OpenFromFd: AMediaExtractor_setDataSourceFd failed with status %d",
         result);
    AMediaExtractor_delete(media_extractor_);
    media_extractor_ = nullptr;
    return false;
  }

  return true;
}

bool VideoDecoder::SelectVideoTrack() {
  if (!media_extractor_) {
    return false;
  }

  int tracks = AMediaExtractor_getTrackCount(media_extractor_);
  for (int i = 0; i < tracks; ++i) {
    AMediaFormat* fmt = AMediaExtractor_getTrackFormat(media_extractor_, i);
    const char* mime = nullptr;
    bool ok = AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime) &&
              (mime && strncmp(mime, "video/", 6) == 0);

    if (ok) {
      AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, &video_width_);
      AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, &video_height_);
      mime_type_ = mime ? mime : "";
      
      // Update video metadata with basic track information
      video_metadata_.width = video_width_;
      video_metadata_.height = video_height_;
      video_metadata_.mime_type = mime_type_;
      
      AMediaExtractor_selectTrack(media_extractor_, i);
      AMediaFormat_delete(fmt);
      return video_width_ > 0 && video_height_ > 0;
    }
    AMediaFormat_delete(fmt);
  }
  return false;
}

void VideoDecoder::LogCodecName(const char* prefix) {
  using GetNameFunc = media_status_t (*)(AMediaCodec*, char**);
  static GetNameFunc get_name_func = nullptr;
  static bool func_checked = false;

  if (!func_checked) {
    void* handle = dlopen("libmediandk.so", RTLD_NOW);
    if (handle) {
      get_name_func = reinterpret_cast<GetNameFunc>(
          dlsym(handle, "AMediaCodec_getName"));
      dlclose(handle);
    }
    func_checked = true;
  }

  if (get_name_func && media_codec_) {
    char* cname = nullptr;
    media_status_t result = get_name_func(media_codec_, &cname);
    if (result == AMEDIA_OK && cname) {
      VIDEO_LOGV(TAG,"%s using decoder %s for %s", prefix, cname, mime_type_.c_str());
      free(cname);
      return;
    }
  }
  VIDEO_LOGV(TAG,"%s using decoder for %s", prefix, mime_type_.c_str());
}

bool VideoDecoder::StepFeedInput(int mode, int64_t target_timestamp_us, bool* saw_input_eos) {
  if (!media_codec_ || !media_extractor_) {
    return false;
  }

  if (!*saw_input_eos && mode == 1) {
    int64_t st = AMediaExtractor_getSampleTime(media_extractor_);
    while (st >= 0 && st + 100000 < target_timestamp_us) {
      AMediaExtractor_advance(media_extractor_);
      st = AMediaExtractor_getSampleTime(media_extractor_);
    }
  }

  if (*saw_input_eos) {
    return true;
  }

  ssize_t in_idx = AMediaCodec_dequeueInputBuffer(media_codec_, 10000);
  if (in_idx >= 0) {
    size_t in_size = 0;
    uint8_t* in = AMediaCodec_getInputBuffer(media_codec_, in_idx, &in_size);
    ssize_t ss = AMediaExtractor_readSampleData(media_extractor_, in, in_size);
    int64_t pts = AMediaExtractor_getSampleTime(media_extractor_);

    if (ss < 0) {
      ss = 0;
      *saw_input_eos = true;
    }

    AMediaCodec_queueInputBuffer(media_codec_, in_idx, 0, ss, pts,
                                 *saw_input_eos ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
    if (!*saw_input_eos) {
      AMediaExtractor_advance(media_extractor_);
    }
  }
  return true;
}

void VideoDecoder::Teardown() {
  if (media_codec_) {
    AMediaCodec_stop(media_codec_);
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
  }
  if (media_extractor_) {
    AMediaExtractor_delete(media_extractor_);
    media_extractor_ = nullptr;
  }
  
  // Reset legacy members
  video_width_ = video_height_ = 0;
  mime_type_.clear();
  native_fps_ = -1.0f;
  frame_interval_us_ = -1;
  
  // Reset video metadata
  video_metadata_ = VideoMetadata{};
}

int VideoDecoder::DecodeWithFps(int max_frames, float target_fps, 
                               std::vector<MNN::Express::VARP>* out_tensors,
                               std::vector<int64_t>* out_timestamps,
                               FrameDebugCallback callback) {
  return DecodeWithFps(max_frames, target_fps, out_tensors, out_timestamps, 1.0f, callback);
}

bool VideoDecoder::GetVideoMetadata(VideoMetadata* metadata) {
  if (!metadata || !media_extractor_) {
    VIDEO_LOGE(TAG, "GetVideoMetadata: invalid parameters or extractor not initialized");
    return false;
  }
  int tracks = AMediaExtractor_getTrackCount(media_extractor_);
  for (int i = 0; i < tracks; ++i) {
    AMediaFormat* fmt = AMediaExtractor_getTrackFormat(media_extractor_, i);
    const char* mime = nullptr;
    if (AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime) &&
        mime && strncmp(mime, "video/", 6) == 0) {
      AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, &metadata->width);
      AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, &metadata->height);
      metadata->mime_type = mime ? mime : "";
      int64_t duration_us = 0;
      if (AMediaFormat_getInt64(fmt, AMEDIAFORMAT_KEY_DURATION, &duration_us) && duration_us > 0) {
        metadata->duration_us = duration_us;
      }
      VIDEO_LOGV(TAG, "GetVideoMetadata: basic info - fps=%.2f, duration=%" PRId64 " us, frames=%" PRId64 ", interval=%" PRId64 " us, dims=%dx%d",
                 metadata->native_fps, metadata->duration_us, metadata->total_frames, metadata->frame_interval_us,
                 metadata->width, metadata->height);
      if (media_codec_) {
        ExtractOutputFormatInfo(metadata);
      }
      AMediaFormat_delete(fmt);
      return true;
    }
    AMediaFormat_delete(fmt);
  }
  VIDEO_LOGW(TAG, "GetVideoMetadata: no video track found, using defaults");
  return false;
}

bool VideoDecoder::ExtractOutputFormatInfo(VideoMetadata* metadata) {
  if (!metadata || !media_codec_) {
    return false;
  }

  AMediaFormat* output_format = AMediaCodec_getOutputFormat(media_codec_);
  if (!output_format) {
    return false;
  }

  VIDEO_LOGV(TAG, "ExtractOutputFormatInfo: output format = %s",
             AMediaFormat_toString(output_format));

  // Extract output format information
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_COLOR_FORMAT, &metadata->color_format);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_STRIDE, &metadata->stride);

  // Extract frame rate from output format if available
  int32_t frame_rate = -1;
  if (AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_FRAME_RATE, &frame_rate) && frame_rate > 0) {
    VIDEO_LOGV(TAG, "ExtractOutputFormatInfo: detected output frame rate=%d fps", frame_rate);
    metadata->native_fps = static_cast<float>(frame_rate);
    metadata->frame_interval_us = static_cast<int64_t>(1000000.0f / frame_rate);

    // Update global metadata with the detected frame rate
    // This is the first time we have accurate frame rate information
    if (native_fps_ == -1.0f) {
      native_fps_ = static_cast<float>(frame_rate);
      frame_interval_us_ = static_cast<int64_t>(1000000.0f / frame_rate);
      VIDEO_LOGV(TAG, "ExtractOutputFormatInfo: updated global metadata - fps=%.2f, interval=%" PRId64 " us", 
                 native_fps_, frame_interval_us_);
    }

    VIDEO_LOGV(TAG, "ExtractOutputFormatInfo: updated fps from output format=%.2f, interval=%" PRId64 " us", 
               metadata->native_fps, metadata->frame_interval_us);
  } else {
    VIDEO_LOGV(TAG, "ExtractOutputFormatInfo: no frame rate found in output format");
  }

#if __ANDROID_API__ >= 28
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_SLICE_HEIGHT, &metadata->slice_height);
#else
  metadata->slice_height = metadata->height;
#endif

#if defined(AMEDIAFORMAT_KEY_CROP_LEFT)
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_LEFT, &metadata->crop_left);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_TOP, &metadata->crop_top);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_RIGHT, &metadata->crop_right);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_BOTTOM, &metadata->crop_bottom);
#else
  // Older NDK levels do not expose the crop keys; fall back to the full frame.
  metadata->crop_left = 0;
  metadata->crop_top = 0;
  metadata->crop_right = metadata->width > 0 ? metadata->width - 1 : 0;
  metadata->crop_bottom = metadata->height > 0 ? metadata->height - 1 : 0;
#endif

  // Calculate output dimensions after cropping
  metadata->output_width = metadata->crop_right - metadata->crop_left + 1;
  metadata->output_height = metadata->crop_bottom - metadata->crop_top + 1;

  // Calculate YUV format information
  metadata->yuv_format_info = ImageUtils::DetectYUVFormatFromMediaCodec(
      metadata->color_format,
      metadata->stride,
      metadata->slice_height,
      metadata->width,
      metadata->height,
      metadata->crop_left,
      metadata->crop_top,
      metadata->crop_right,
      metadata->crop_bottom);

  metadata->format_info_ready = true;
  VIDEO_LOGV(TAG, "ExtractOutputFormatInfo: output format ready - stride=%d, slice=%d, crop=[%d,%d,%d,%d], output=%dx%d, yuv_format=%d",
             metadata->stride, metadata->slice_height, metadata->crop_left, metadata->crop_top, 
             metadata->crop_right, metadata->crop_bottom, metadata->output_width, metadata->output_height,
             static_cast<int>(metadata->yuv_format_info.format));

  AMediaFormat_delete(output_format);
  return true;
}

std::vector<int64_t> VideoDecoder::CalculateSampleIndices(const VideoMetadata& metadata, 
                                                          int max_frames, 
                                                          float target_fps,
                                                          float skip_secs) {
  std::vector<int64_t> indices;
  
  if (metadata.total_frames <= 0) {
    VIDEO_LOGW(TAG, "CalculateSampleIndices: invalid total_frames=%" PRId64, metadata.total_frames);
    return indices;
  }
  
  if (metadata.duration_us <= 0) {
    VIDEO_LOGW(TAG, "CalculateSampleIndices: invalid duration=%" PRId64 " us", metadata.duration_us);
    return indices;
  }
  
  // Convert duration to seconds
  double duration_seconds = static_cast<double>(metadata.duration_us) / 1000000.0;
  
  // Step 1: Estimate how many frames we'd sample at target_fps
  int estimated_frames = static_cast<int>(round(target_fps * duration_seconds));
  
  // Step 2: Determine desired frames
  int desired_frames = std::min(estimated_frames, max_frames);
  if (desired_frames < 1) {
    desired_frames = 1;
  }
  
  // Step 3: Calculate sampling range with skip logic
  int64_t start_idx = 0;
  int64_t end_idx = metadata.total_frames - 1;
  
  if (skip_secs > 0 && (duration_seconds - 2 * skip_secs) > (max_frames / target_fps)) {
    start_idx = static_cast<int64_t>(skip_secs * metadata.native_fps);
    end_idx = metadata.total_frames - static_cast<int64_t>(skip_secs * metadata.native_fps);
  }
  
  // Ensure valid range
  start_idx = std::max<int64_t>(0, start_idx);
  end_idx = std::min<int64_t>(end_idx, metadata.total_frames - 1);
  
  if (start_idx >= end_idx) {
    start_idx = 0;
    end_idx = metadata.total_frames - 1;
  }
  
  // Step 4: Uniform sampling using linear interpolation
  if (desired_frames == 1) {
    // Single frame: take middle
    indices.push_back((start_idx + end_idx) / 2);
  } else {
    for (int i = 0; i < desired_frames; ++i) {
      double ratio = static_cast<double>(i) / (desired_frames - 1);
      int64_t idx = start_idx + static_cast<int64_t>(ratio * (end_idx - start_idx));
      indices.push_back(idx);
    }
  }
  
  // Remove duplicates and sort
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  
  VIDEO_LOGV(TAG, "CalculateSampleIndices: total_frames=%" PRId64 ", duration=%.2fs, desired=%d, range=[%" PRId64 ",%" PRId64 "], result=%zu indices",
             metadata.total_frames, duration_seconds, desired_frames, start_idx, end_idx, indices.size());
  
  return indices;
}

// Helper function to determine if we should capture the current frame based on target_fps
bool VideoDecoder::ShouldCaptureFrame(int64_t current_pts_us, int64_t last_captured_pts_us, float target_fps) {
  if (last_captured_pts_us < 0) {
    // First frame, always capture
    return true;
  }
  
  const int64_t target_interval_us = static_cast<int64_t>(1000000.0f / target_fps);
  const int64_t time_since_last_capture = current_pts_us - last_captured_pts_us;
  
  // Capture if enough time has passed since last capture
  return time_since_last_capture >= target_interval_us;
}

// Helper function to determine if we should capture the current frame based on frame index (Video-style)
bool VideoDecoder::ShouldCaptureFrameByIndex(int64_t current_frame_index, const std::vector<int64_t>& sample_indices) {
  // Binary search to check if current_frame_index is in sample_indices
  return std::binary_search(sample_indices.begin(), sample_indices.end(), current_frame_index);
}

// Calculate sample indices using Video-style sampling (similar to video_sample_indices_fn)
std::vector<int64_t> VideoDecoder::CalculateVideoSampleIndices(const VideoMetadata& metadata, 
                                                                 int max_frames, 
                                                                 float target_fps,
                                                                 float skip_secs) {
  std::vector<int64_t> indices;
  
  int64_t total_num_frames = metadata.total_frames;
  if (total_num_frames <= 0) {
    VIDEO_LOGW(TAG, "CalculateVideoSampleIndices: invalid total_num_frames=%" PRId64, total_num_frames);
    return indices;
  }
  
  float native_fps = metadata.native_fps > 0 ? metadata.native_fps : 30.0f;
  double duration_seconds = static_cast<double>(metadata.duration_us) / 1000000.0;
  
  if (duration_seconds <= 0) {
    VIDEO_LOGW(TAG, "CalculateVideoSampleIndices: invalid duration_seconds=%.2f", duration_seconds);
    return indices;
  }
  
  // Step 1: Estimate how many frames we'd sample at target_fps
  int estimated_frames = static_cast<int>(round(target_fps * duration_seconds));
  
  // Step 2: Determine desired frames
  int desired_frames = std::min(estimated_frames, max_frames);
  if (desired_frames < 1) {
    desired_frames = 1;
  }
  
  // Step 3: Calculate sampling range with skip logic (Video-style)
  int64_t start_idx = 0;
  int64_t end_idx = total_num_frames - 1;
  
  if (skip_secs > 0 && (duration_seconds - 2 * skip_secs) > (max_frames / target_fps)) {
    start_idx = static_cast<int64_t>(skip_secs * native_fps);
    end_idx = total_num_frames - static_cast<int64_t>(skip_secs * native_fps);
  }
  
  // Ensure valid range
  start_idx = std::max<int64_t>(0, start_idx);
  end_idx = std::min<int64_t>(end_idx, total_num_frames - 1);
  
  if (start_idx >= end_idx) {
    start_idx = 0;
    end_idx = total_num_frames - 1;
  }
  
  // Step 4: Uniform sampling using linear interpolation (Video-style)
  if (desired_frames == 1) {
    // Single frame: take middle
    indices.push_back((start_idx + end_idx) / 2);
  } else {
    for (int i = 0; i < desired_frames; ++i) {
      double ratio = static_cast<double>(i) / (desired_frames - 1);
      int64_t idx = start_idx + static_cast<int64_t>(ratio * (end_idx - start_idx));
      indices.push_back(idx);
    }
  }
  
  // Remove duplicates and sort (Video-style)
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  
  VIDEO_LOGV(TAG, "CalculateVideoSampleIndices: total_frames=%" PRId64 ", duration=%.2fs, desired=%d, range=[%" PRId64 ",%" PRId64 "], result=%zu indices",
             total_num_frames, duration_seconds, desired_frames, start_idx, end_idx, indices.size());
  
  return indices;
}

int VideoDecoder::DecodeWithFps(int max_frames, float target_fps, 
                               std::vector<MNN::Express::VARP>* out_tensors,
                               std::vector<int64_t>* out_timestamps,
                               float skip_secs,
                               FrameDebugCallback callback) {
  if (!media_codec_ || !media_extractor_) {
    VIDEO_LOGE(TAG, "VIDEO_DECODE: decoder not properly initialized");
    return -1;
  }

  if (!out_tensors || !out_timestamps) {
    VIDEO_LOGE(TAG, "VIDEO_DECODE: output parameters cannot be null");
    return -1;
  }

  out_tensors->clear();
  out_timestamps->clear();

  // Get video metadata for basic info
  VideoMetadata metadata;
  bool metadata_ok = GetVideoMetadata(&metadata);

  VIDEO_LOGV(TAG, "VIDEO_DECODE: ===== DECODE SESSION START =====");
  VIDEO_LOGV(TAG, "VIDEO_DECODE: target_fps=%.2f, max_frames=%d, skip_secs=%.1f", 
             target_fps, max_frames, skip_secs);

  int frames_captured = 0;
  int64_t last_captured_pts_us = -1;
  int64_t current_frame_index = 0;  // Frame index counter
  int64_t first_frame_pts_us = -1;  // First frame timestamp for fixed logic
  std::vector<int64_t> sample_indices;  // Video-style sample indices
  bool indices_calculated = false;  // Whether we've calculated indices after first frame
  int consecutive_failures = 0;  // Track consecutive decode failures to prevent infinite loops
  const int max_consecutive_failures = 10;  // Maximum failures before giving up
  
  // Calculate start time considering skip_secs
  int64_t start_time_us = 0;
  if (skip_secs > 0 && metadata_ok && metadata.duration_us > 0) {
    start_time_us = static_cast<int64_t>(skip_secs * 1000000.0f);
  }
  
  // Start decoding from the beginning or skip position
  int64_t current_decode_timestamp_us = start_time_us;
  const int64_t decode_step_us = 33333; // ~30fps decode step for smooth scanning

  while (frames_captured < max_frames) {
    VIDEO_LOGV(TAG, "VIDEO_DECODE: ----- DECODE ATTEMPT AT %" PRId64 " us (frame_idx=%" PRId64 ") -----", 
               current_decode_timestamp_us, current_frame_index);

    std::vector<uint8_t> yuv_data;
    ImageUtils::YUVFormatInfo format_info;
    int64_t pts_us = 0;
    long native_ms = 0;
    bool saw_eos = false;

    // Get next frame from codec (no RGB conversion yet)
    bool decode_ok = GetNextFrame(&yuv_data, &format_info, &pts_us, &native_ms, &saw_eos);

    if (saw_eos) {
      VIDEO_LOGV(TAG, "VIDEO_DECODE: EOS reached, captured %d frames", frames_captured);
      break;
    }

    if (!decode_ok || yuv_data.empty()) {
      consecutive_failures++;
      VIDEO_LOGV(TAG, "VIDEO_DECODE: FAILED to decode frame at timestamp=%" PRId64 " us (frame_idx=%" PRId64 "), consecutive_failures=%d", 
                 current_decode_timestamp_us, current_frame_index, consecutive_failures);
      
      // Check for too many consecutive failures
      if (consecutive_failures >= max_consecutive_failures) {
        VIDEO_LOGW(TAG, "VIDEO_DECODE: Too many consecutive failures (%d), ending decode session", consecutive_failures);
        break;
      }
      
      // Still increment frame index even on decode failure to avoid infinite loop
      current_frame_index++;
      current_decode_timestamp_us += decode_step_us;
      continue;
    }
    VIDEO_LOGV(TAG, "VIDEO_DECODE: DECODED frame %" PRId64 " - pts=%" PRId64 " us, yuv_size=%zu bytes", 
      current_frame_index, pts_us, yuv_data.size());
    // Record first frame timestamp for fixed logic
    if (current_frame_index == 0) {
      first_frame_pts_us = pts_us;
      last_captured_pts_us = 0;
      VIDEO_LOGV(TAG, "VIDEO_DECODE: First frame pts recorded: %" PRId64 " us", first_frame_pts_us);
    }

    // Reset failure counter on successful decode
    consecutive_failures = 0;
    VIDEO_LOGV(TAG, "VIDEO_DECODE test 1");
    // Calculate sample indices after first frame (when we have accurate metadata)
    if (!indices_calculated && metadata_ok && current_frame_index == 0) {
      // Update metadata with actual frame information if needed
      if (metadata.total_frames <= 0 && metadata.duration_us > 0 && metadata.native_fps > 0) {
        metadata.total_frames = static_cast<int64_t>((metadata.duration_us / 1000000.0) * metadata.native_fps);
        VIDEO_LOGV(TAG, "VIDEO_DECODE: Estimated total_frames=%" PRId64 " from duration and fps", metadata.total_frames);
      }
      sample_indices = CalculateVideoSampleIndices(metadata, max_frames, target_fps, skip_secs);
      indices_calculated = true;
      
      // Print indices for debugging
      std::string indices_str;
      for (size_t i = 0; i < sample_indices.size(); i++) {
        if (i > 0) indices_str += ",";
        indices_str += std::to_string(sample_indices[i]);
      }
      VIDEO_LOGV(TAG, "VIDEO_DECODE: Video sample indices: [%s]", indices_str.c_str());
    }
    // Determine capture decision using both methods for cross-validation
    bool should_capture = 
        ShouldCaptureFrameByIndex(current_frame_index, sample_indices);
    if (should_capture) {
      // Only now convert YUV to RGB since we decided to capture this frame
      std::vector<uint8_t> rgb_data;
      if (!ConvertYuvToRgb(yuv_data, format_info, &rgb_data)) {
        VIDEO_LOGE(TAG, "VIDEO_DECODE: FAILED to convert YUV to RGB");
        current_decode_timestamp_us += decode_step_us;
        continue;
      }

      // Convert RGB data to MNN tensor
      MNN::Express::VARP tensor = mls::CreateTensorFromRgb(
          rgb_data.data(), video_metadata_.width, video_metadata_.height);
      
      if (!tensor.get()) {
        VIDEO_LOGE(TAG, "VIDEO_DECODE: FAILED to create tensor from RGB data");
        current_decode_timestamp_us += decode_step_us;
        continue;
      }

      // Store the captured frame
      out_tensors->push_back(tensor);
      out_timestamps->push_back(pts_us);
      last_captured_pts_us = pts_us;

      VIDEO_LOGV(TAG, "VIDEO_DECODE: FRAME %d CAPTURED - pts=%" PRId64 " us, interval_since_last=%" PRId64 " us, rgb_size=%zu", 
                 frames_captured, pts_us, 
                 frames_captured > 0 ? (pts_us - (frames_captured > 1 ? (*out_timestamps)[frames_captured-1] : 0)) : 0,
                 rgb_data.size());

      // Call the debug callback if provided
      if (callback) {
        callback(tensor, pts_us, native_ms, current_decode_timestamp_us, "DynamicCapture", 
                 video_metadata_.width, video_metadata_.height);
      }

      frames_captured++;
    } else {
      VIDEO_LOGV(TAG, "VIDEO_DECODE: SKIPPED frame %" PRId64 " - pts=%" PRId64 " us, method=%s", 
                 current_frame_index, pts_us, indices_calculated ? "index" : "timestamp");
    }
    current_frame_index++;
    current_decode_timestamp_us += decode_step_us;
    VIDEO_LOGV(TAG, "VIDEO_DECODE: frame captured %d max frames %d", frames_captured, max_frames);
  }

  VIDEO_LOGV(TAG, "VIDEO_DECODE: ===== DECODE SESSION END =====");
  VIDEO_LOGV(TAG, "VIDEO_DECODE: SUMMARY - requested=%d, captured=%d, success_rate=%.1f%%", 
             max_frames, frames_captured, (frames_captured * 100.0f / max_frames));
  
  return frames_captured;
}

namespace mls {

} // namespace mls
