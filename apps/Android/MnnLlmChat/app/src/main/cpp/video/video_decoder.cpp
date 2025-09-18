#include "video_decoder.hpp"

#include <android/log.h>
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

#ifdef __ANDROID__
namespace mls {
MNN::Express::VARP RawRgbToVar(const uint8_t* rgb_data, int width, int height);
}
#endif

#define TAG "VideoDecoder"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#ifdef LOGE
#undef LOGE
#endif
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

VideoDecoder::VideoDecoder() = default;

VideoDecoder::~VideoDecoder() { Teardown(); }

VideoDecoder* VideoDecoder::CreateFromFd(int fd, off64_t offset, off64_t length) {
  LOGV("CreateFromFd: fd=%d, offset=%lld, length=%lld", fd,
       (long long)offset, (long long)length);
  VideoDecoder* decoder = new ByteBufferDecoder();
  if (!decoder->OpenFromFd(fd, offset, length)) {
    LOGE("CreateFromFd: OpenFromFd failed");
    delete decoder;
    return nullptr;
  }
  LOGV("CreateFromFd: success");
  return decoder;
}

VideoDecoder* VideoDecoder::CreateByteBufferDecoder(int fd, off64_t offset, off64_t length) {
  LOGV("CreateByteBufferDecoder: fd=%d, offset=%lld, length=%lld", fd,
       (long long)offset, (long long)length);
  VideoDecoder* decoder = new ByteBufferDecoder();
  if (!decoder->OpenFromFd(fd, offset, length)) {
    LOGE("CreateByteBufferDecoder: OpenFromFd failed");
    delete decoder;
    return nullptr;
  }
  LOGV("CreateByteBufferDecoder: success");
  return decoder;
}

VideoDecoder* VideoDecoder::CreateSurfaceDecoder(int fd, off64_t offset, off64_t length) {
  LOGV("CreateSurfaceDecoder: fd=%d, offset=%lld, length=%lld", fd,
       (long long)offset, (long long)length);
  VideoDecoder* decoder = new SurfaceDecoder();
  if (!decoder->OpenFromFd(fd, offset, length)) {
    LOGE("CreateSurfaceDecoder: OpenFromFd failed");
    delete decoder;
    return nullptr;
  }
  LOGV("CreateSurfaceDecoder: success");
  return decoder;
}

bool VideoDecoder::OpenFromFd(int fd, off64_t offset, off64_t length) {
  Teardown();

  media_extractor_ = AMediaExtractor_new();
  if (!media_extractor_) {
    LOGE("OpenFromFd: AMediaExtractor_new() failed");
    return false;
  }

  media_status_t result = AMediaExtractor_setDataSourceFd(media_extractor_,
                                                          fd, offset, length);
  if (result != AMEDIA_OK) {
    LOGE("OpenFromFd: AMediaExtractor_setDataSourceFd failed with status %d",
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
      LOGV("%s using decoder %s for %s", prefix, cname, mime_type_.c_str());
      free(cname);
      return;
    }
  }
  LOGV("%s using decoder for %s", prefix, mime_type_.c_str());
}

bool VideoDecoder::StepFeedInput(int mode, int64_t next_target_us, bool* saw_input_eos) {
  if (!media_codec_ || !media_extractor_) {
    return false;
  }

  if (!*saw_input_eos && mode == 1) {
    int64_t st = AMediaExtractor_getSampleTime(media_extractor_);
    while (st >= 0 && st + 100000 < next_target_us) {
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
    float fps,
    FrameDebugCallback callback) {
  (void)out_dir;
  (void)mode;
  (void)csv_path;
  int afd = open(asset_path, O_RDONLY);
  if (afd < 0) {
    LOGE("DecodeByteBufferWithFps: open file failed %s", asset_path);
    return -1;
  }

  struct stat st;
  if (fstat(afd, &st) != 0) {
    LOGE("DecodeByteBufferWithFps: fstat failed for %s", asset_path);
    close(afd);
    return -1;
  }

  LOGV("DecodeByteBufferWithFps: file size = %lld bytes",
       static_cast<long long>(st.st_size));

  VideoDecoder* decoder = VideoDecoder::CreateFromFd(afd, 0, st.st_size);
  close(afd);
  if (!decoder) {
    LOGE("DecodeByteBufferWithFps: create decoder failed");
    return -1;
  }

  if (!decoder->SelectVideoTrack()) {
    LOGE("DecodeByteBufferWithFps: no video track found");
    delete decoder;
    return -1;
  }

  if (!decoder->Configure()) {
    LOGE("DecodeByteBufferWithFps: configure failed");
    delete decoder;
    return -1;
  }

  int saved = 0;
  const int64_t step_us = static_cast<int64_t>(1000000.0f / fps);
  int64_t next_target_us = 0;

  while (saved < max_seconds) {
    std::vector<uint8_t> rgb;
    int64_t pts = 0;
    long native_ms = 0;
    bool eos = false;

    if (decoder->DecodeFrame(next_target_us, &rgb, &pts, &native_ms, &eos)) {
      if (callback) {
        callback(rgb, pts, native_ms, next_target_us, strategy,
                 decoder->width(), decoder->height());
      }
      ++saved;
      next_target_us += step_us;
    }

    if (eos) {
      break;
    }
  }

  delete decoder;
  return saved;
}

namespace mls {

} // namespace mls
