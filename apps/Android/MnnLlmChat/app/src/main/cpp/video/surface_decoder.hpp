#ifndef SURFACE_DECODER_HPP_
#define SURFACE_DECODER_HPP_

#include <android/native_window.h>

#include "video_decoder.hpp"

class SurfaceDecoder : public VideoDecoder {
 public:
  SurfaceDecoder();
  ~SurfaceDecoder() override;

  bool Configure() override;
  bool DecodeFrame(int64_t target_timestamp_us,
                   int64_t tolerance_us,
                   std::vector<uint8_t>* out_rgb,
                   int64_t* out_pts_us,
                   long* native_ms,
                   bool* out_eos) override;

  bool SetSurface(ANativeWindow* window);
  void ClearSurface();

 private:
  bool ConfigureSurface();

  ANativeWindow* native_window_ = nullptr;
  bool surface_configured_ = false;
};

#endif  // SURFACE_DECODER_HPP_
