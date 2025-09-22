#include "byte_buffer_decoder.hpp"

#include <cstring>
#include <ctime>
#include <memory>

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

bool ConvertYuvToTensorWithImageProcess(const std::vector<uint8_t>& yuv_data,
                                        const ImageUtils::YUVFormatInfo& format_info,
                                        MNN::Express::VARP* out_tensor) {
  if (!out_tensor) {
    return false;
  }

  if (yuv_data.empty() || !format_info.isValid) {
    *out_tensor = nullptr;
    return false;
  }

  int dst_width = format_info.outputWidth;
  int dst_height = format_info.outputHeight;
  if (dst_width <= 0 || dst_height <= 0) {
    VIDEO_LOGE(TAG,"ConvertYuvToTensorWithImageProcess: invalid dimensions %dx%d", dst_width,
         dst_height);
    *out_tensor = nullptr;
    return false;
  }

  int stride = format_info.stride > 0 ? format_info.stride : dst_width;
  int slice_height = format_info.sliceHeight > 0 ? format_info.sliceHeight : dst_height;
  VIDEO_LOGV(TAG,"ConvertYuvToTensor: dims=%dx%d stride=%d slice=%d bytes=%zu format=%d",
       dst_width, dst_height, stride, slice_height, yuv_data.size(),
       static_cast<int>(format_info.format));

  MNN::CV::ImageProcess::Config config;
  config.sourceFormat = ToImageProcessFormat(format_info.format);
  config.destFormat = MNN::CV::RGB;

  std::unique_ptr<MNN::CV::ImageProcess> processor(
      MNN::CV::ImageProcess::create(config));
  if (!processor) {
    VIDEO_LOGE(TAG,"ConvertYuvToTensorWithImageProcess: failed to create ImageProcess");
    *out_tensor = nullptr;
    return false;
  }

  auto tensor = MNN::Express::_Input({dst_height, dst_width, 3},
                                     MNN::Express::NHWC,
                                     halide_type_of<uint8_t>());
  auto buffer = tensor->writeMap<uint8_t>();
  if (!buffer) {
    VIDEO_LOGE(TAG,"ConvertYuvToTensorWithImageProcess: failed to map tensor memory");
    *out_tensor = nullptr;
    return false;
  }

  auto status = processor->convert(yuv_data.data(), dst_width, slice_height,
                                   stride, buffer, dst_width, dst_height, 3,
                                   dst_width * 3, halide_type_of<uint8_t>());
  if (status != MNN::NO_ERROR) {
    VIDEO_LOGE(TAG,"ConvertYuvToTensorWithImageProcess: convert failed, status=%d",
         static_cast<int>(status));
    *out_tensor = nullptr;
    return false;
  }

  VIDEO_LOGV(TAG,"ConvertYuvToTensor: conversion succeeded, tensor=%p", tensor.get());

  *out_tensor = tensor;
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
    VIDEO_LOGE(TAG,"UpdateOutputFormatInfo: failed to get output format");
    return false;
  }

  VIDEO_LOGV(TAG,"UpdateOutputFormatInfo: output format = %s",
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

bool ByteBufferDecoder::DecodeFrameToYuv(int64_t target_timestamp_us,
                                         int64_t tolerance_us,
                                         std::vector<uint8_t>* out_yuv,
                                         ImageUtils::YUVFormatInfo* format_info,
                                         int64_t* out_pts_us,
                                         long* native_ms,
                                         bool* out_eos) {
  if (!media_codec_ || !media_extractor_) {
    VIDEO_LOGE(TAG,"DecodeFrameToYuv: codec or extractor not ready");
    return false;
  }
  if (!out_yuv || !format_info || !out_eos) {
    VIDEO_LOGE(TAG,"DecodeFrameToYuv: invalid output parameters");
    return false;
  }

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

  while (true) {
    timespec t0{};
    clock_gettime(CLOCK_MONOTONIC, &t0);

    AMediaCodecBufferInfo info{};
    ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(media_codec_, &info, 10000);
    if (out_idx < 0) {
      if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
        VIDEO_LOGV(TAG,"DecodeFrameToYuv: output format changed, updating info");
        if (!UpdateOutputFormatInfo()) {
          VIDEO_LOGE(TAG,"DecodeFrameToYuv: failed to handle format change");
          return false;
        }
        continue;
      }
      VIDEO_LOGV(TAG,"DecodeFrameToYuv: no output buffer available (idx=%zd)", out_idx);
      return false;
    }

    *out_eos = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0;
    int64_t pts = info.presentationTimeUs;
    if (out_pts_us) {
      *out_pts_us = pts;
    }

    bool capture = (target_timestamp_us <= 0) || *out_eos ||
                   (pts + tolerance_us >= target_timestamp_us);
    if (!capture) {
      VIDEO_LOGV(TAG,"DecodeFrameToYuv: skip frame pts=%lld target=%lld tolerance=%lld",
           (long long)pts, (long long)target_timestamp_us,
           (long long)tolerance_us);
      AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
      continue;
    }

    bool success = true;
    if (info.size > 0) {
      size_t out_size = 0;
      uint8_t* out = AMediaCodec_getOutputBuffer(media_codec_, out_idx, &out_size);
      if (!out || out_size < static_cast<size_t>(info.offset + info.size)) {
        VIDEO_LOGE(TAG,"DecodeFrameToYuv: invalid output buffer");
        success = false;
      } else {
        out_yuv->resize(info.size);
        memcpy(out_yuv->data(), out + info.offset, info.size);
        VIDEO_LOGV(TAG,"DecodeFrameToYuv: copied %d bytes (offset=%d, capacity=%zu)",
             info.size, info.offset, out_size);

        if (format_info_updated_ && output_format_info_.isValid) {
          *format_info = output_format_info_;
        } else {
          *format_info = ImageUtils::DetectYUVFormatWithFallback(
              out_yuv->data(),
              video_width_,
              video_height_,
              info.size);
        }

        if (!format_info->isValid) {
          VIDEO_LOGE(TAG,"DecodeFrameToYuv: unable to determine YUV format for frame");
          success = false;
        } else {
          VIDEO_LOGV(TAG,"DecodeFrameToYuv: format=%d stride=%d slice=%d output=%dx%d",
               static_cast<int>(format_info->format), format_info->stride,
               format_info->sliceHeight, format_info->outputWidth,
               format_info->outputHeight);
        }
      }
    } else {
      out_yuv->clear();
      format_info->isValid = false;
    }

    if (native_ms) {
      timespec t1{};
      clock_gettime(CLOCK_MONOTONIC, &t1);
      *native_ms = (t1.tv_sec - t0.tv_sec) * 1000 +
                   (t1.tv_nsec - t0.tv_nsec) / 1000000;
    }

    AMediaCodec_releaseOutputBuffer(media_codec_, out_idx, false);
    VIDEO_LOGV(TAG,"DecodeFrameToYuv: return %d yuv_size=%zu", success ? 1 : 0,
         out_yuv->size());
    return success;
  }
}

bool ByteBufferDecoder::DecodeFrame(int64_t target_timestamp_us,
                                        int64_t tolerance_us,
                                        std::vector<uint8_t>* out_rgb,
                                        int64_t* out_pts_us,
                                        long* native_ms,
                                        bool* out_eos) {
  if (!out_rgb || !out_eos) {
    VIDEO_LOGE(TAG,"DecodeFrame: invalid output parameters");
    return false;
  }

  ImageUtils::YUVFormatInfo format_info;
  std::vector<uint8_t> yuv_data;
  if (!DecodeFrameToYuv(target_timestamp_us, tolerance_us, &yuv_data, &format_info,
                        out_pts_us, native_ms, out_eos)) {
    return false;
  }

  if (yuv_data.empty() || !format_info.isValid) {
    out_rgb->clear();
    return true;
  }

  if (!ConvertYuvToRgbWithImageProcess(yuv_data, format_info, out_rgb)) {
    VIDEO_LOGE(TAG,"DecodeFrame: YUV to RGB conversion failed via ImageProcess");
    return false;
  }

  VIDEO_LOGV(TAG,"DecodeFrame: converted frame to RGB via ImageProcess, size=%zu",
       out_rgb->size());
  return true;
}

bool ByteBufferDecoder::DecodeFrameToTensor(int64_t target_timestamp_us,
                                            int64_t tolerance_us,
                                            MNN::Express::VARP* out_tensor,
                                            int64_t* out_pts_us,
                                            long* native_ms,
                                            bool* out_eos) {
  if (!out_tensor || !out_eos) {
    VIDEO_LOGE(TAG,"DecodeFrameToTensor: invalid output parameters");
    return false;
  }

  ImageUtils::YUVFormatInfo format_info;
  std::vector<uint8_t> yuv_data;
  if (!DecodeFrameToYuv(target_timestamp_us, tolerance_us, &yuv_data, &format_info,
                        out_pts_us, native_ms, out_eos)) {
    return false;
  }

  if (yuv_data.empty() || !format_info.isValid) {
    *out_tensor = nullptr;
    VIDEO_LOGV(TAG,"DecodeFrameToTensor: empty YUV buffer or invalid format (valid=%d)",
         format_info.isValid);
    return true;
  }

  if (!ConvertYuvToTensorWithImageProcess(yuv_data, format_info, out_tensor)) {
    VIDEO_LOGE(TAG,"DecodeFrameToTensor: ImageProcess conversion to tensor failed");
    return false;
  }

  if (out_tensor && out_tensor->get()) {
    auto info = (*out_tensor)->getInfo();
    if (info && info->dim.size() >= 3) {
      VIDEO_LOGV(TAG,"DecodeFrameToTensor: tensor ready dims=%d,%d,%d",
           info->dim[0], info->dim[1], info->dim[2]);
    }
  }

  return true;
}
