#include "video_decoder.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"

#include "../mls_log.h"
#include "byte_buffer_decoder.hpp"
#include "surface_decoder.hpp"
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

VideoDecoder* VideoDecoder::CreateSurfaceDecoder(int fd, off64_t offset, off64_t length) {
  VIDEO_LOGV(TAG,"CreateSurfaceDecoder: fd=%d, offset=%lld, length=%lld", fd,
       (long long)offset, (long long)length);
  VideoDecoder* decoder = new SurfaceDecoder();
  if (!decoder->OpenFromFd(fd, offset, length)) {
    VIDEO_LOGE(TAG,"CreateSurfaceDecoder: OpenFromFd failed");
    delete decoder;
    return nullptr;
  }
  VIDEO_LOGV(TAG,"CreateSurfaceDecoder: success");
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
  video_width_ = video_height_ = 0;
  mime_type_.clear();
}

int VideoDecoder::DecodeByteBufferWithFps(
    const char* asset_path,
    const char* out_dir,
    int max_seconds,
    int mode,
    const char* strategy,
    const char* csv_path,
    float target_fps,
    FrameDebugCallback callback) {
  (void)out_dir;
  (void)mode;
  (void)csv_path;
  int afd = open(asset_path, O_RDONLY);
  if (afd < 0) {
    VIDEO_LOGE(TAG,"DecodeByteBufferWithFps: open file failed %s", asset_path);
    return -1;
  }

  struct stat st;
  if (fstat(afd, &st) != 0) {
    VIDEO_LOGE(TAG,"DecodeByteBufferWithFps: fstat failed for %s", asset_path);
    close(afd);
    return -1;
  }

  VIDEO_LOGV(TAG,"DecodeByteBufferWithFps: file size = %lld bytes",
       static_cast<long long>(st.st_size));

  VideoDecoder* decoder = VideoDecoder::CreateFromFd(afd, 0, st.st_size);
  close(afd);
  if (!decoder) {
    VIDEO_LOGE(TAG,"DecodeByteBufferWithFps: create decoder failed");
    return -1;
  }

  if (!decoder->SelectVideoTrack()) {
    VIDEO_LOGE(TAG,"DecodeByteBufferWithFps: no video track found");
    delete decoder;
    return -1;
  }

  if (!decoder->Configure()) {
    VIDEO_LOGE(TAG,"DecodeByteBufferWithFps: configure failed");
    delete decoder;
    return -1;
  }

  int saved = 0;
  // Calculate frame interval in microseconds (1 second = 1,000,000 microseconds)
  const int64_t frame_interval_us = static_cast<int64_t>(1000000.0f / target_fps);
  int64_t target_timestamp_us = 0;

  while (saved < max_seconds) {
    std::vector<uint8_t> rgb;
    int64_t pts = 0;
    long native_ms = 0;
    bool eos = false;

    int64_t tolerance_us = std::max<int64_t>(frame_interval_us / 2, 20000);
    if (decoder->DecodeFrame(target_timestamp_us, tolerance_us, &rgb, &pts,
                             &native_ms, &eos)) {
      if (callback) {
        MNN::Express::VARP tensor;
        if (!rgb.empty()) {
          tensor = mls::CreateTensorFromRgb(rgb.data(), decoder->width(),
                                            decoder->height());
        }
        callback(tensor, pts, native_ms, target_timestamp_us, strategy,
                 decoder->width(), decoder->height());
      }
      ++saved;
      target_timestamp_us += frame_interval_us;
    }

    if (eos) {
      break;
    }
  }

  delete decoder;
  return saved;
}

bool VideoDecoder::DecodeFrameToTensor(int64_t target_timestamp_us,
                                       int64_t tolerance_us,
                                       MNN::Express::VARP* out_tensor,
                                       int64_t* out_pts_us,
                                       long* native_ms,
                                       bool* out_eos) {
  if (!out_tensor) {
    VIDEO_LOGE(TAG,"DecodeFrameToTensor: out_tensor pointer is null");
    return false;
  }

  std::vector<uint8_t> rgb_data;
  if (!DecodeFrame(target_timestamp_us, tolerance_us, &rgb_data, out_pts_us,
                   native_ms, out_eos)) {
    return false;
  }

  if (rgb_data.empty()) {
    *out_tensor = nullptr;
    return true;
  }

  auto tensor = mls::CreateTensorFromRgb(rgb_data.data(), video_width_, video_height_);
  if (tensor.get() == nullptr) {
    VIDEO_LOGE(TAG,"DecodeFrameToTensor: failed to create tensor from RGB data");
    return false;
  }

  *out_tensor = tensor;
  return true;
}

int VideoDecoder::DecodeWithFps(int max_frames, float target_fps, 
                               std::vector<MNN::Express::VARP>* out_tensors,
                               std::vector<int64_t>* out_timestamps,
                               FrameDebugCallback callback) {
  if (!media_codec_ || !media_extractor_) {
    VIDEO_LOGE(TAG,"SMART_DECODE: decoder not properly initialized");
    return -1;
  }

  if (!out_tensors || !out_timestamps) {
    VIDEO_LOGE(TAG,"SMART_DECODE: output parameters cannot be null");
    return -1;
  }

  out_tensors->clear();
  out_timestamps->clear();

  int frames_decoded = 0;
  
  // Calculate target sampling interval
  const int64_t target_frame_interval_us = static_cast<int64_t>(1000000.0f / target_fps);
  
  // Get original video frame interval from video metadata
  int64_t original_frame_interval_us = 33333; // ~30fps default
  
  // Try to get actual frame rate from video format if available
  if (media_extractor_) {
    int tracks = AMediaExtractor_getTrackCount(media_extractor_);
    for (int i = 0; i < tracks; ++i) {
      AMediaFormat* fmt = AMediaExtractor_getTrackFormat(media_extractor_, i);
      const char* mime = nullptr;
      if (AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime) &&
          mime && strncmp(mime, "video/", 6) == 0) {
        float video_fps = 30.0f; // default
        if (AMediaFormat_getFloat(fmt, AMEDIAFORMAT_KEY_FRAME_RATE, &video_fps) && video_fps > 0) {
          original_frame_interval_us = static_cast<int64_t>(1000000.0f / video_fps);
          VIDEO_LOGV(TAG,"SMART_DECODE: detected video fps=%.2f, interval=%lld us", video_fps, original_frame_interval_us);
        }
        AMediaFormat_delete(fmt);
        break;
      }
      AMediaFormat_delete(fmt);
    }
  }
  
  int64_t target_timestamp_us = 0;
  bool saw_eos = false;

  VIDEO_LOGV(TAG,"SMART_DECODE: ===== DECODE SESSION START =====");
  VIDEO_LOGV(TAG,"SMART_DECODE: target_fps=%.2f, max_frames=%d", target_fps, max_frames);
  VIDEO_LOGV(TAG,"SMART_DECODE: target_interval=%lld us, original_interval=%lld us", 
       target_frame_interval_us, original_frame_interval_us);

  while (frames_decoded < max_frames && !saw_eos) {
    VIDEO_LOGV(TAG,"SMART_DECODE: ----- FRAME %d SEARCH START -----", frames_decoded);
    
    MNN::Express::VARP tensor;
    int64_t pts_us = 0;
    long native_ms = 0;
    
    // Smart tolerance calculation
    float sampling_ratio = (float)target_frame_interval_us / original_frame_interval_us;
    int64_t base_tolerance_us;
    
    if (sampling_ratio > 3.0f) {
      // Very low sampling rate: use larger tolerance to find nearest frames
      base_tolerance_us = std::min<int64_t>(target_frame_interval_us / 2, 100000);
      VIDEO_LOGV(TAG,"SMART_DECODE: low sampling mode, ratio=%.2f, tolerance=%lld us", 
           sampling_ratio, base_tolerance_us);
    } else if (sampling_ratio > 1.5f) {
      // Low sampling rate: use moderate tolerance
      base_tolerance_us = std::max<int64_t>(original_frame_interval_us, 50000);
      VIDEO_LOGV(TAG,"SMART_DECODE: standard sampling mode, ratio=%.2f, tolerance=%lld us", 
           sampling_ratio, base_tolerance_us);
    } else {
      // High sampling rate: use standard tolerance
      base_tolerance_us = std::max<int64_t>(original_frame_interval_us / 2, 20000);
      VIDEO_LOGV(TAG,"SMART_DECODE: high sampling mode, ratio=%.2f, tolerance=%lld us", 
           sampling_ratio, base_tolerance_us);
    }

    VIDEO_LOGV(TAG,"SMART_DECODE: target_timestamp=%lld us", target_timestamp_us);

    // Multi-stage frame finding with improved strategy for low sampling rates
    bool found = false;
    int stage = 1;
    
    // For low sampling rates, we need a different approach
    if (sampling_ratio > 2.0f) {
      // Low sampling rate: find the closest frame within a reasonable window
      int64_t search_window = target_frame_interval_us / 2; // Half the target interval
      int64_t best_pts = -1;
      int64_t best_diff = LLONG_MAX;
      MNN::Express::VARP best_tensor;
      long best_native_ms = 0;
      
      VIDEO_LOGV(TAG,"SMART_DECODE: low sampling strategy - search window=%lld us", search_window);
      
      // Search backwards first (earlier frames)
      for (int64_t offset = 0; offset <= search_window && !saw_eos; offset += 10000) {
        int64_t search_target = target_timestamp_us - offset;
        if (search_target < 0) break;
        
        VIDEO_LOGV(TAG,"SMART_DECODE: searching backwards, offset=-%lld us, target=%lld us", 
             offset, search_target);
        
        bool decode_ok = DecodeFrameToTensor(search_target, base_tolerance_us, &tensor,
                                           &pts_us, &native_ms, &saw_eos);
        
        if (decode_ok && tensor.get()) {
          int64_t diff = abs(pts_us - target_timestamp_us);
          VIDEO_LOGV(TAG,"SMART_DECODE: found frame at pts=%lld us, diff=%lld us", pts_us, diff);
          
          if (diff < best_diff) {
            best_diff = diff;
            best_pts = pts_us;
            best_tensor = tensor;
            best_native_ms = native_ms;
          }
          break; // Take the first valid frame we find
        }
      }
      
      // If no frame found backwards, search forwards
      if (best_pts == -1 && !saw_eos) {
        for (int64_t offset = 10000; offset <= search_window && !saw_eos; offset += 10000) {
          int64_t search_target = target_timestamp_us + offset;
          
          VIDEO_LOGV(TAG,"SMART_DECODE: searching forwards, offset=+%lld us, target=%lld us", 
               offset, search_target);
          
          bool decode_ok = DecodeFrameToTensor(search_target, base_tolerance_us, &tensor,
                                             &pts_us, &native_ms, &saw_eos);
          
          if (decode_ok && tensor.get()) {
            int64_t diff = abs(pts_us - target_timestamp_us);
            VIDEO_LOGV(TAG,"SMART_DECODE: found frame at pts=%lld us, diff=%lld us", pts_us, diff);
            
            if (diff < best_diff) {
              best_diff = diff;
              best_pts = pts_us;
              best_tensor = tensor;
              best_native_ms = native_ms;
            }
            break; // Take the first valid frame we find
          }
        }
      }
      
      if (best_pts != -1) {
        found = true;
        tensor = best_tensor;
        pts_us = best_pts;
        native_ms = best_native_ms;
        stage = 1;
        VIDEO_LOGV(TAG,"SMART_DECODE: low sampling SUCCESS - pts=%lld us, diff=%lld us", 
             pts_us, best_diff);
      }
    } else {
      // Standard sampling rate: use original multi-stage approach
      // Stage 1: Precise matching with base tolerance
      VIDEO_LOGV(TAG,"SMART_DECODE: stage 1 - precise matching, tolerance=%lld us", base_tolerance_us);
      bool decode_ok = DecodeFrameToTensor(target_timestamp_us, base_tolerance_us, &tensor,
                                           &pts_us, &native_ms, &saw_eos);
      
      if (decode_ok && tensor.get()) {
        found = true;
        VIDEO_LOGV(TAG,"SMART_DECODE: stage 1 SUCCESS - pts=%lld us, diff=%lld us", 
             pts_us, pts_us - target_timestamp_us);
      } else if (!saw_eos) {
        // Stage 2: Expanded tolerance (2x)
        stage = 2;
        int64_t expanded_tolerance = std::min<int64_t>(base_tolerance_us * 2, 200000);
        VIDEO_LOGV(TAG,"SMART_DECODE: stage 2 - expanded tolerance=%lld us", expanded_tolerance);
        
        decode_ok = DecodeFrameToTensor(target_timestamp_us, expanded_tolerance, &tensor,
                                       &pts_us, &native_ms, &saw_eos);
        
        if (decode_ok && tensor.get()) {
          found = true;
          VIDEO_LOGV(TAG,"SMART_DECODE: stage 2 SUCCESS - pts=%lld us, diff=%lld us", 
               pts_us, pts_us - target_timestamp_us);
        }
      }
    }

    if (saw_eos) {
      VIDEO_LOGV(TAG,"SMART_DECODE: EOS reached at frame %d", frames_decoded);
      break;
    }

    if (!found || !tensor.get()) {
      VIDEO_LOGV(TAG,"SMART_DECODE: FAILED to find frame after all stages, skipping target=%lld us", 
           target_timestamp_us);
      target_timestamp_us += target_frame_interval_us;
      continue;
    }

    // Success: store the frame
    out_tensors->push_back(tensor);
    out_timestamps->push_back(pts_us);

    VIDEO_LOGV(TAG,"SMART_DECODE: FRAME %d CAPTURED - pts=%lld us, target=%lld us, stage=%d, native_time=%ld ms", 
         frames_decoded, pts_us, target_timestamp_us, stage, native_ms);

    // Call the debug callback if provided
    if (callback) {
      callback(tensor, pts_us, native_ms, target_timestamp_us, "SmartDecode", 
               video_width_, video_height_);
    }

    frames_decoded++;
    target_timestamp_us += target_frame_interval_us;
    
    VIDEO_LOGV(TAG,"SMART_DECODE: next target=%lld us", target_timestamp_us);
  }

  VIDEO_LOGV(TAG,"SMART_DECODE: ===== DECODE SESSION END =====");
  VIDEO_LOGV(TAG,"SMART_DECODE: SUMMARY - requested=%d, decoded=%d, success_rate=%.1f%%", 
       max_frames, frames_decoded, (frames_decoded * 100.0f / max_frames));
  
  return frames_decoded;
}

namespace mls {

} // namespace mls
