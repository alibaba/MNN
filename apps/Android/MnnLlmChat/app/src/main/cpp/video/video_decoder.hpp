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

namespace MNN {
namespace Express {
class VARP;
} // namespace Express
} // namespace MNN

class VideoDecoder {
 public:
  // Debug callback receives RGB pixels ready for inspection/saving.
  using FrameDebugCallback = std::function<void(
      const std::vector<uint8_t>& rgb,
      int64_t pts,
      long native_ms,
      int64_t target_us,
      const char* strategy,
      int width,
      int height)>;

  VideoDecoder();
  virtual ~VideoDecoder();

  static VideoDecoder* CreateFromFd(int fd, off64_t offset, off64_t length);
  static VideoDecoder* CreateByteBufferDecoder(int fd, off64_t offset, off64_t length);
  static VideoDecoder* CreateSurfaceDecoder(int fd, off64_t offset, off64_t length);

  bool OpenFromFd(int fd, off64_t offset, off64_t length);
  bool SelectVideoTrack();
  virtual bool Configure() = 0;

  static int DecodeByteBufferWithFps(
      const char* asset_path,
      const char* out_dir,
      int max_seconds,
      int mode,
      const char* strategy,
      const char* csv_path,
      float fps,
      FrameDebugCallback callback = nullptr);

  // Implementations fill out_rgb with packed RGB data when available
  // (surface decoder leaves it empty).
  virtual bool DecodeFrame(int64_t next_target_us,
                           std::vector<uint8_t>* out_rgb,
                           int64_t* out_pts_us,
                           long* native_ms,
                           bool* out_eos) = 0;

  int width() const { return video_width_; }
  int height() const { return video_height_; }
  const std::string& mime_type() const { return mime_type_; }

 protected:
  void Teardown();
  void LogCodecName(const char* prefix);
  bool StepFeedInput(int mode, int64_t next_target_us, bool* saw_input_eos);

  AMediaExtractor* media_extractor_ = nullptr;
  AMediaCodec* media_codec_ = nullptr;
  int video_width_ = 0;
  int video_height_ = 0;
  std::string mime_type_;
};

#endif  // VIDEO_DECODER_HPP_
