#include "surface_decoder.hpp"

#include <ctime>

#include "video_utils.hpp"

#define TAG "SurfaceDecoder"

SurfaceDecoder::SurfaceDecoder() = default;

SurfaceDecoder::~SurfaceDecoder() {
  ClearSurface();
  Teardown();
}

bool SurfaceDecoder::Configure() {
  return ConfigureSurface();
}

bool SurfaceDecoder::ConfigureSurface() {
  if (!media_extractor_ || mime_type_.empty()) {
    VIDEO_LOGE(TAG,"ConfigureSurface: extractor or mime type not set");
    return false;
  }

  AMediaFormat* fmt = AMediaExtractor_getTrackFormat(
      media_extractor_,
      AMediaExtractor_getSampleTrackIndex(media_extractor_));
  if (!fmt) {
    VIDEO_LOGE(TAG,"ConfigureSurface: failed to get track format");
    return false;
  }

  media_codec_ = AMediaCodec_createDecoderByType(mime_type_.c_str());
  if (!media_codec_) {
    VIDEO_LOGE(TAG,"ConfigureSurface: failed to create decoder for %s",
         mime_type_.c_str());
    AMediaFormat_delete(fmt);
    return false;
  }

  if (AMediaCodec_configure(media_codec_, fmt, native_window_, nullptr, 0) != AMEDIA_OK) {
    VIDEO_LOGE(TAG,"ConfigureSurface: codec configuration failed");
    AMediaFormat_delete(fmt);
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
    return false;
  }

  AMediaFormat_delete(fmt);
  LogCodecName("Surface");

  if (AMediaCodec_start(media_codec_) != AMEDIA_OK) {
    VIDEO_LOGE(TAG,"ConfigureSurface: failed to start codec");
    AMediaCodec_delete(media_codec_);
    media_codec_ = nullptr;
    return false;
  }

  surface_configured_ = true;
  return true;
}

bool SurfaceDecoder::SetSurface(ANativeWindow* window) {
  if (surface_configured_) {
    VIDEO_LOGE(TAG,"SetSurface: decoder already configured");
    return false;
  }

  native_window_ = window;
  return true;
}

void SurfaceDecoder::ClearSurface() {
  native_window_ = nullptr;
  surface_configured_ = false;
}

bool SurfaceDecoder::DecodeFrame(int64_t target_timestamp_us,
                                     int64_t tolerance_us,
                                     std::vector<uint8_t>* out_rgb,
                                     int64_t* out_pts_us,
                                     long* native_ms,
                                     bool* out_eos) {
  if (!media_codec_ || !media_extractor_) {
    VIDEO_LOGE(TAG,"DecodeFrame: codec or extractor not ready");
    return false;
  }
  if (!out_eos) {
    VIDEO_LOGE(TAG,"DecodeFrame: out_eos pointer is null");
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
      VIDEO_LOGV(TAG,"DecodeFrame: output format changed");
      AMediaFormat* format = AMediaCodec_getOutputFormat(media_codec_);
      if (format) {
        VIDEO_LOGV(TAG,"DecodeFrame: new format: %s", AMediaFormat_toString(format));
        AMediaFormat_delete(format);
      }
      out_idx = AMediaCodec_dequeueOutputBuffer(media_codec_, &info, 10000);
    } else {
      VIDEO_LOGV(TAG,"DecodeFrame: no output buffer available (idx=%zd)", out_idx);
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

  const bool capture = (target_timestamp_us <= 0) ||
                       (pts + tolerance_us >= target_timestamp_us);
  if (!capture) {
    AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
    return false;
  }

  if (out_rgb) {
    out_rgb->clear();
  }

  if (native_ms) {
    timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *native_ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
  }

  AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, true);
  return true;
}
