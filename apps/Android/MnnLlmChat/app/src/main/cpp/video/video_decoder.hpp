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
  // Debug callback receives MNN tensor ready for inspection/saving.
  using FrameDebugCallback = std::function<void(
      MNN::Express::VARP tensor,
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
      float target_fps,
      FrameDebugCallback callback = nullptr);

  // Simplified interface for VideoProcessor - returns MNN tensors directly
  int DecodeWithFps(int max_frames, float target_fps, 
                   std::vector<MNN::Express::VARP>* out_tensors,
                   std::vector<int64_t>* out_timestamps,
                   FrameDebugCallback callback = nullptr);

  // Implementations fill out_rgb with packed RGB data when available
  // (surface decoder leaves it empty).
  virtual bool DecodeFrame(int64_t target_timestamp_us,
                           int64_t tolerance_us,
                           std::vector<uint8_t>* out_rgb,
                           int64_t* out_pts_us,
                           long* native_ms,
                           bool* out_eos) = 0;

  virtual bool DecodeFrameToTensor(int64_t target_timestamp_us,
                                   int64_t tolerance_us,
                                   MNN::Express::VARP* out_tensor,
                                   int64_t* out_pts_us,
                                   long* native_ms,
                                   bool* out_eos);

  int width() const { return video_width_; }
  int height() const { return video_height_; }
  const std::string& mime_type() const { return mime_type_; }

 protected:
  void Teardown();
  void LogCodecName(const char* prefix);
  bool StepFeedInput(int mode, int64_t target_timestamp_us, bool* saw_input_eos);

  AMediaExtractor* media_extractor_ = nullptr;
  AMediaCodec* media_codec_ = nullptr;
  int video_width_ = 0;
  int video_height_ = 0;
  std::string mime_type_;
};

#endif  // VIDEO_DECODER_HPP_
