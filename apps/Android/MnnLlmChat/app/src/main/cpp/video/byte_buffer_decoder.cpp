#include "byte_buffer_decoder.hpp"

#include <cstring>
#include <ctime>
#include <memory>
#include <inttypes.h>
#include <unistd.h>

#include "MNN/ImageProcess.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"

#include "image_utils.hpp"
#include "video_utils.hpp"

#define TAG "ByteBufferDecoder"

namespace mls {
MNN::Express::VARP CreateTensorFromRgb(const uint8_t* rgb_data,
                                       int width,
                                       int height);
}

namespace {

MNN::CV::ImageFormat ToImageProcessFormat(
    ImageUtils::YUVFormat format) {
  switch (format) {
    case ImageUtils::YUVFormat::NV12:
      return MNN::CV::YUV_NV12;
    case ImageUtils::YUVFormat::NV21:
      return MNN::CV::YUV_NV21;
    case ImageUtils::YUVFormat::I420:
      return MNN::CV::YUV_I420;
    default:
      return MNN::CV::YUV_NV21;
  }
}

bool ConvertYuvToRgbWithImageProcess(const std::vector<uint8_t>& yuv_data,
                                     const ImageUtils::YUVFormatInfo& format_info,
                                     std::vector<uint8_t>* out_rgb) {
  if (!out_rgb) {
    return false;
  }

  if (yuv_data.empty() || !format_info.isValid) {
    out_rgb->clear();
    return false;
  }

  int dst_width = format_info.outputWidth;
  int dst_height = format_info.outputHeight;
  if (dst_width <= 0 || dst_height <= 0) {
    VIDEO_LOGE(TAG,"ConvertYuvToRgbWithImageProcess: invalid dimensions %dx%d", dst_width,
         dst_height);
    return false;
  }

  int stride = format_info.stride > 0 ? format_info.stride : dst_width;
  int slice_height = format_info.sliceHeight > 0 ? format_info.sliceHeight : dst_height;

  MNN::CV::ImageProcess::Config config;
  config.sourceFormat = ToImageProcessFormat(format_info.format);
  config.destFormat = MNN::CV::RGB;

  std::unique_ptr<MNN::CV::ImageProcess> processor(
      MNN::CV::ImageProcess::create(config));
  if (!processor) {
    VIDEO_LOGE(TAG,"ConvertYuvToRgbWithImageProcess: failed to create ImageProcess");
    return false;
  }

  out_rgb->resize(static_cast<size_t>(dst_width) * dst_height * 3);
  auto status = processor->convert(yuv_data.data(), dst_width, slice_height,
                                   stride, out_rgb->data(), dst_width,
                                   dst_height, 3, dst_width * 3,
                                   halide_type_of<uint8_t>());
  if (status != MNN::NO_ERROR) {
    VIDEO_LOGE(TAG,"ConvertYuvToRgbWithImageProcess: ImageProcess convert failed, status=%d",
         static_cast<int>(status));
    out_rgb->clear();
    return false;
  }

  return true;
}

}  // namespace

ByteBufferDecoder::ByteBufferDecoder() = default;

ByteBufferDecoder::~ByteBufferDecoder() { Teardown(); }

bool ByteBufferDecoder::Configure() {
  return ConfigureByteBuffer();
}

bool ByteBufferDecoder::ConfigureByteBuffer() {
  if (!media_extractor_ || mime_type_.empty()) {
    VIDEO_LOGE(TAG,"ConfigureByteBuffer: extractor or mime type not set");
    return false;
  }

  AMediaFormat* fmt = AMediaExtractor_getTrackFormat(
      media_extractor_,
      AMediaExtractor_getSampleTrackIndex(media_extractor_));
  if (!fmt) {
    VIDEO_LOGE(TAG,"ConfigureByteBuffer: failed to get track format");
    return false;
  }

  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_COLOR_FORMAT,
                        ImageUtils::kColorFormatYUV420Flexible);

  media_codec_ = AMediaCodec_createDecoderByType(mime_type_.c_str());
  if (!media_codec_) {
    VIDEO_LOGE(TAG,"ConfigureByteBuffer: failed to create decoder for %s",
         mime_type_.c_str());
    AMediaFormat_delete(fmt);
    return false;
  }

  if (AMediaCodec_configure(media_codec_, fmt, nullptr, nullptr, 0) != AMEDIA_OK) {
    VIDEO_LOGE(TAG,"ConfigureByteBuffer: codec configuration failed");
    AMediaFormat_delete(fmt);
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
    return false;
  }

  AMediaFormat_delete(fmt);
  LogCodecName("ByteBuffer");

  if (AMediaCodec_start(media_codec_) != AMEDIA_OK) {
    VIDEO_LOGE(TAG,"ConfigureByteBuffer: failed to start codec");
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
    return false;
  }

  // Reset EOS tracking
  input_eos_ = false;
  output_eos_ = false;
  
  // Initialize video metadata after successful configuration
  // This will get basic information from track format
  if (!GetVideoMetadata(&video_metadata_)) {
    VIDEO_LOGW(TAG, "ConfigureByteBuffer: failed to get initial video metadata");
  } else {
    VIDEO_LOGV(TAG, "ConfigureByteBuffer: initial metadata - dims=%dx%d, fps=%.2f, duration=%" PRId64 " us",
               video_metadata_.width, video_metadata_.height, video_metadata_.native_fps, video_metadata_.duration_us);
  }
  
  return true;
}

bool ByteBufferDecoder::UpdateOutputFormatInfo() {
  if (!media_codec_) {
    return false;
  }
  
  // Update video metadata with the latest output format information
  // This ensures we have the most accurate format details including YUV format info
  if (!GetVideoMetadata(&video_metadata_)) {
    VIDEO_LOGW(TAG, "UpdateOutputFormatInfo: failed to update video metadata");
    return false;
  }
  
  VIDEO_LOGV(TAG, "UpdateOutputFormatInfo: metadata updated - fps=%.2f, stride=%d, slice=%d, crop=[%d,%d,%d,%d], output=%dx%d, yuv_format=%d",
             video_metadata_.native_fps, video_metadata_.stride, video_metadata_.slice_height,
             video_metadata_.crop_left, video_metadata_.crop_top, video_metadata_.crop_right, video_metadata_.crop_bottom,
             video_metadata_.output_width, video_metadata_.output_height, static_cast<int>(video_metadata_.yuv_format_info.format));
  
  return video_metadata_.format_info_ready;
}

// Feed input data to codec (separated from YUV decoding)
bool ByteBufferDecoder::FeedInputToCodec(bool* out_eos) {
  if (!media_codec_ || !media_extractor_ || !out_eos) {
    return false;
  }

  if (*out_eos) {
    return true; // Already at end of stream
  }

  ssize_t in_idx = AMediaCodec_dequeueInputBuffer(media_codec_, 10000);
  VIDEO_LOGV(TAG, "FeedInputToCodec: dequeueInputBuffer returned %zd", in_idx);
  if (in_idx >= 0) {
    size_t in_size = 0;
    uint8_t* in = AMediaCodec_getInputBuffer(media_codec_, in_idx, &in_size);
    if (!in || in_size == 0) {
      VIDEO_LOGW(TAG, "FeedInputToCodec: input buffer null or zero, size=%zu", in_size);
      // Return buffer to codec to avoid stalling
      AMediaCodec_queueInputBuffer(media_codec_, in_idx, 0, 0, 0, 0);
      return false;
    }

    ssize_t ss = AMediaExtractor_readSampleData(media_extractor_, in, in_size);
    int64_t pts = AMediaExtractor_getSampleTime(media_extractor_);

    if (ss < 0) {
      ss = 0;
      *out_eos = true;
    }

    AMediaCodec_queueInputBuffer(media_codec_, in_idx, 0, ss, pts,
                                 *out_eos ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
    VIDEO_LOGV(TAG, "FeedInputToCodec: queued size=%zd, pts=%" PRId64 ", eos=%d", ss, pts, *out_eos);
    if (!*out_eos) {
      AMediaExtractor_advance(media_extractor_);
    }
    return true;
  }
  return false;
}

// Simple function to get next available YUV frame from codec
bool ByteBufferDecoder::GetNextYuvFrame(std::vector<uint8_t>* out_yuv,
                                        ImageUtils::YUVFormatInfo* format_info,
                                        int64_t* out_pts_us,
                                        long* native_ms,
                                        bool* out_eos) {
  if (!media_codec_ || !out_yuv || !format_info || !out_eos) {
    VIDEO_LOGE(TAG, "GetNextYuvFrame: invalid parameters");
    return false;
  }

  timespec t0{};
  if (native_ms) {
    clock_gettime(CLOCK_MONOTONIC, &t0);
  }

  // Non-blocking poll with limited retries to be robust around startup/drain
  const int kMaxPolls = 100;
  for (int tries = 0; tries < kMaxPolls; ++tries) {
    AMediaCodecBufferInfo info{};
    ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(media_codec_, &info, 0 /* non-blocking */);
    if (out_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
      // Backoff a little to avoid busy spin
      usleep(2000); // 2ms
      continue;
    }

    if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
      VIDEO_LOGV(TAG, "GetNextYuvFrame: output format changed, updating info");
      UpdateOutputFormatInfo();
      continue;
    }

    if (out_idx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
      // With NDK API, we still use getOutputBuffer per index; just log and continue.
      VIDEO_LOGV(TAG, "GetNextYuvFrame: output buffers changed");
      continue;
    }

    if (out_idx < 0) {
      VIDEO_LOGV(TAG, "GetNextYuvFrame: dequeueOutputBuffer returned code=%zd", out_idx);
      continue;
    }

    // Valid buffer
    *out_eos = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0;
    if (out_pts_us) {
      *out_pts_us = info.presentationTimeUs;
    }

    bool success = true;
    if (info.size > 0) {
      size_t out_size = 0;
      uint8_t* out = AMediaCodec_getOutputBuffer(media_codec_, out_idx, &out_size);
      if (!out || out_size < static_cast<size_t>(info.offset + info.size)) {
        VIDEO_LOGE(TAG, "GetNextYuvFrame: invalid output buffer");
        success = false;
      } else {
        out_yuv->resize(info.size);
        memcpy(out_yuv->data(), out + info.offset, info.size);

        if (video_metadata_.format_info_ready && video_metadata_.yuv_format_info.isValid) {
          *format_info = video_metadata_.yuv_format_info;
        } else {
          *format_info = ImageUtils::DetectYUVFormatWithFallback(
              out_yuv->data(),
              video_metadata_.width,
              video_metadata_.height,
              info.size);
        }

        if (!format_info->isValid) {
          VIDEO_LOGE(TAG, "GetNextYuvFrame: unable to determine YUV format for frame");
          success = false;
        }
      }
    } else {
      out_yuv->clear();
      format_info->isValid = false;
      success = false;
    }

    if (native_ms) {
      timespec t1{};
      clock_gettime(CLOCK_MONOTONIC, &t1);
      *native_ms = (t1.tv_sec - t0.tv_sec) * 1000 +
                   (t1.tv_nsec - t0.tv_nsec) / 1000000;
    }

    AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
    return success;
  }

  return false;
}

bool ByteBufferDecoder::ConvertYuvToRgb(const std::vector<uint8_t>& yuv_data,
                                        const ImageUtils::YUVFormatInfo& format_info,
                                        std::vector<uint8_t>* out_rgb) {
  if (!out_rgb) {
    VIDEO_LOGE(TAG, "ConvertYuvToRgb: invalid output parameter");
    return false;
  }

  if (yuv_data.empty() || !format_info.isValid) {
    out_rgb->clear();
    return true;
  }

  if (!ConvertYuvToRgbWithImageProcess(yuv_data, format_info, out_rgb)) {
    VIDEO_LOGE(TAG, "ConvertYuvToRgb: YUV to RGB conversion failed via ImageProcess");
    return false;
  }

  VIDEO_LOGV(TAG, "ConvertYuvToRgb: converted frame to RGB via ImageProcess, size=%zu",
             out_rgb->size());
  return true;
}

bool ByteBufferDecoder::GetNextFrame(std::vector<uint8_t>* out_yuv,
                                     ImageUtils::YUVFormatInfo* format_info,
                                     int64_t* out_pts_us,
                                     long* native_ms,
                                     bool* out_eos) {
  if (!media_codec_ || !out_yuv || !format_info || !out_eos) {
    VIDEO_LOGE(TAG, "GetNextFrame: invalid parameters");
    return false;
  }

  timespec t0{};
  if (native_ms) {
    clock_gettime(CLOCK_MONOTONIC, &t0);
  }

  // We consider out_eos as OUTPUT EOS only. Input EOS is tracked separately.
  *out_eos = output_eos_;

  // Try repeatedly to feed input and dequeue output until we either get a frame
  // or observe OUTPUT EOS from the codec. This makes EOS handling robust.
  const int kMaxOuterLoops = 200;          // Safeguard against infinite loops per call
  const int kMaxDrainPollsAfterInputEos = 200; // Extra polls after input EOS to drain pending frames
  int drain_polls = 0;

  for (int loop = 0; loop < kMaxOuterLoops; ++loop) {
    // Feed more input if available and not yet EOS on input side
    if (!input_eos_) {
      bool feed_ok = FeedInputToCodec(&input_eos_);
      VIDEO_LOGV(TAG, "GetNextFrame: FeedInputToCodec ok=%d, input_eos=%d", feed_ok, input_eos_);
    }

    // Dequeue output non-blocking and handle special return codes in-loop
    AMediaCodecBufferInfo info{};
    ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(media_codec_, &info, 0 /* non-blocking */);
    if (out_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
      // No output available right now. If input has ended, keep draining more aggressively.
      if (input_eos_) {
        if (++drain_polls >= kMaxDrainPollsAfterInputEos) {
          // Assume drained if we've polled many times after input EOS.
          output_eos_ = true;
          *out_eos = true;
          VIDEO_LOGV(TAG, "GetNextFrame: assume drained after %d polls post input EOS", drain_polls);
          return false;
        }
        usleep(1000); // 1ms backoff while draining
        continue; // keep polling output
      }
      // Otherwise, go back to feed more input
      usleep(1000); // backoff to avoid busy spin
      continue;
    }

    if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
      VIDEO_LOGV(TAG, "GetNextFrame: output format changed, updating info");
      UpdateOutputFormatInfo();
      continue; // Try dequeue again
    }

    if (out_idx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
      VIDEO_LOGV(TAG, "GetNextFrame: output buffers changed");
      continue; // No special handling needed for NDK; retry
    }

    if (out_idx < 0) {
      // Other negative codes (buffers changed etc.) - continue loop
      VIDEO_LOGV(TAG, "GetNextFrame: dequeueOutputBuffer returned code=%zd", out_idx);
      continue;
    }

    // Valid output buffer
    output_eos_ = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0;
    *out_eos = output_eos_;
    if (out_pts_us) {
      *out_pts_us = info.presentationTimeUs;
    }

    bool success = true;
    if (info.size > 0) {
      size_t out_size = 0;
      uint8_t* out = AMediaCodec_getOutputBuffer(media_codec_, out_idx, &out_size);
      if (!out || out_size < static_cast<size_t>(info.offset + info.size)) {
        VIDEO_LOGE(TAG, "GetNextFrame: invalid output buffer");
        success = false;
      } else {
        out_yuv->resize(info.size);
        memcpy(out_yuv->data(), out + info.offset, info.size);

        if (video_metadata_.format_info_ready && video_metadata_.yuv_format_info.isValid) {
          *format_info = video_metadata_.yuv_format_info;
        } else {
          *format_info = ImageUtils::DetectYUVFormatWithFallback(
              out_yuv->data(),
              video_metadata_.width,
              video_metadata_.height,
              info.size);
        }

        if (!format_info->isValid) {
          VIDEO_LOGE(TAG, "GetNextFrame: unable to determine YUV format for frame");
          success = false;
        }
      }
    } else {
      // size == 0 frame: nothing to output; if EOS, we signal completion.
      out_yuv->clear();
      format_info->isValid = false;
      success = false;
    }

    if (native_ms) {
      timespec t1{};
      clock_gettime(CLOCK_MONOTONIC, &t1);
      *native_ms = (t1.tv_sec - t0.tv_sec) * 1000 +
                   (t1.tv_nsec - t0.tv_nsec) / 1000000;
    }

    AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);

    if (success) {
      return true; // Produced a frame
    }

    if (output_eos_) {
      // Drained final buffer (possibly empty) after EOS
      return false;
    }

    // Otherwise continue feeding/dequeuing
  }

  // Loop exhausted without producing a frame; return false (not EOS unless flagged)
  return false;
}
