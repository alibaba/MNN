#include "byte_buffer_decoder.hpp"

#include <android/log.h>
#include <cstring>
#include <ctime>

#include "image_utils.hpp"

#define TAG "ByteBufferDecoder"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

ByteBufferDecoder::ByteBufferDecoder() = default;

ByteBufferDecoder::~ByteBufferDecoder() { Teardown(); }

bool ByteBufferDecoder::Configure() {
  return ConfigureByteBuffer();
}

bool ByteBufferDecoder::ConfigureByteBuffer() {
  if (!media_extractor_ || mime_type_.empty()) {
    LOGE("ConfigureByteBuffer: extractor or mime type not set");
    return false;
  }

  AMediaFormat* fmt = AMediaExtractor_getTrackFormat(
      media_extractor_,
      AMediaExtractor_getSampleTrackIndex(media_extractor_));
  if (!fmt) {
    LOGE("ConfigureByteBuffer: failed to get track format");
    return false;
  }

  AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_COLOR_FORMAT,
                        ImageUtils::kColorFormatYUV420Flexible);

  media_codec_ = AMediaCodec_createDecoderByType(mime_type_.c_str());
  if (!media_codec_) {
    LOGE("ConfigureByteBuffer: failed to create decoder for %s",
         mime_type_.c_str());
    AMediaFormat_delete(fmt);
    return false;
  }

  if (AMediaCodec_configure(media_codec_, fmt, nullptr, nullptr, 0) != AMEDIA_OK) {
    LOGE("ConfigureByteBuffer: codec configuration failed");
    AMediaFormat_delete(fmt);
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
    return false;
  }

  AMediaFormat_delete(fmt);
  LogCodecName("ByteBuffer");

  if (AMediaCodec_start(media_codec_) != AMEDIA_OK) {
    LOGE("ConfigureByteBuffer: failed to start codec");
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
    return false;
  }

  output_format_info_ = {};
  format_info_updated_ = false;
  return true;
}

bool ByteBufferDecoder::UpdateOutputFormatInfo() {
  if (!media_codec_) {
    return false;
  }

  AMediaFormat* output_format = AMediaCodec_getOutputFormat(media_codec_);
  if (!output_format) {
    LOGE("UpdateOutputFormatInfo: failed to get output format");
    return false;
  }

  LOGV("UpdateOutputFormatInfo: output format = %s",
       AMediaFormat_toString(output_format));

  int32_t color_format = 0;
  int32_t stride = 0;
  int32_t slice_height = 0;
  int32_t crop_left = 0;
  int32_t crop_top = 0;
  int32_t crop_right = 0;
  int32_t crop_bottom = 0;

  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_COLOR_FORMAT,
                        &color_format);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_STRIDE, &stride);
#if __ANDROID_API__ >= 28
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_SLICE_HEIGHT,
                        &slice_height);
#else
  slice_height = video_height_;
#endif
#if defined(AMEDIAFORMAT_KEY_CROP_LEFT)
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_LEFT, &crop_left);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_TOP, &crop_top);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_RIGHT, &crop_right);
  AMediaFormat_getInt32(output_format, AMEDIAFORMAT_KEY_CROP_BOTTOM, &crop_bottom);
#else
  // Older NDK levels do not expose the crop keys; fall back to the full frame.
  crop_left = 0;
  crop_top = 0;
  crop_right = video_width_ > 0 ? video_width_ - 1 : 0;
  crop_bottom = video_height_ > 0 ? video_height_ - 1 : 0;
#endif

  output_format_info_ = ImageUtils::DetectYUVFormatFromMediaCodec(
      color_format,
      stride,
      slice_height,
      video_width_,
      video_height_,
      crop_left,
      crop_top,
      crop_right,
      crop_bottom);

  format_info_updated_ = output_format_info_.isValid;
  AMediaFormat_delete(output_format);
  return format_info_updated_;
}

bool ByteBufferDecoder::DecodeFrame(int64_t next_target_us,
                                        std::vector<uint8_t>* out_rgb,
                                        int64_t* out_pts_us,
                                        long* native_ms,
                                        bool* out_eos) {
  if (!media_codec_ || !media_extractor_) {
    LOGE("DecodeFrame: codec or extractor not ready");
    return false;
  }
  if (!out_rgb || !out_eos) {
    LOGE("DecodeFrame: invalid output parameters");
    return false;
  }

  timespec t0;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  if (!*out_eos) {
    ssize_t in_idx = AMediaCodec_dequeueInputBuffer(media_codec_, 10000);
    if (in_idx >= 0) {
      size_t in_size = 0;
      uint8_t* in = AMediaCodec_getInputBuffer(media_codec_, in_idx, &in_size);
      ssize_t ss = AMediaExtractor_readSampleData(media_extractor_, in, in_size);
      int64_t pts = AMediaExtractor_getSampleTime(media_extractor_);

      if (ss < 0) {
        ss = 0;
        *out_eos = true;
      }

      AMediaCodec_queueInputBuffer(media_codec_, in_idx, 0, ss, pts,
                                   *out_eos ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
      if (!*out_eos) {
        AMediaExtractor_advance(media_extractor_);
      }
    }
  }

  AMediaCodecBufferInfo info;
  ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(media_codec_, &info, 10000);
  if (out_idx < 0) {
    if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
      LOGV("DecodeFrame: output format changed, updating info");
      if (!UpdateOutputFormatInfo()) {
        LOGE("DecodeFrame: failed to handle format change");
        return false;
      }
      out_idx = AMediaCodec_dequeueOutputBuffer(media_codec_, &info, 10000);
    } else {
      LOGV("DecodeFrame: no output buffer available (idx=%zd)", out_idx);
      return false;
    }
  }

  if (out_idx < 0) {
    return false;
  }

  *out_eos = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0;
  int64_t pts = info.presentationTimeUs;
  if (out_pts_us) {
    *out_pts_us = pts;
  }

  const bool capture = (pts + 20000 >= next_target_us);
  if (!capture) {
    AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
    return false;
  }

  if (info.size > 0) {
    size_t out_size = 0;
    uint8_t* out = AMediaCodec_getOutputBuffer(media_codec_, out_idx, &out_size);
    if (!out || out_size < static_cast<size_t>(info.offset + info.size)) {
      LOGE("DecodeFrame: invalid output buffer");
      AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
      return false;
    }

    std::vector<uint8_t> yuv(info.size);
    memcpy(yuv.data(), out + info.offset, info.size);

    ImageUtils::YUVFormatInfo format_info;
    if (format_info_updated_ && output_format_info_.isValid) {
      format_info = output_format_info_;
    } else {
      format_info = ImageUtils::DetectYUVFormatWithFallback(
          yuv.data(),
          video_width_,
          video_height_,
          info.size);
    }

        if (!format_info.isValid) {
      LOGE("DecodeFrame: unable to determine YUV format for frame");
      AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
      return false;
    }

    if (!ImageUtils::YUVToRgb(yuv.data(), format_info, *out_rgb)) {
      LOGE("DecodeFrame: YUV to RGB conversion failed");
      AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
      return false;
    }

    LOGV("DecodeFrame: converted frame to RGB, size=%zu", out_rgb->size());
  } else {
    out_rgb->clear();
  }

  if (native_ms) {
    timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *native_ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
  }
  AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
  return true;
}
