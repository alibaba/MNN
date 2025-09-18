#ifndef BYTE_BUFFER_DECODER_HPP_
#define BYTE_BUFFER_DECODER_HPP_

#include "image_utils.hpp"
#include "video_decoder.hpp"

class ByteBufferDecoder : public VideoDecoder {
 public:
  ByteBufferDecoder();
  ~ByteBufferDecoder() override;

  bool Configure() override;
  bool DecodeFrame(int64_t next_target_us,
                   std::vector<uint8_t>* out_rgb,
                   int64_t* out_pts_us,
                   long* native_ms,
                   bool* out_eos) override;

 private:
  bool ConfigureByteBuffer();
  bool UpdateOutputFormatInfo();

  ImageUtils::YUVFormatInfo output_format_info_;
  bool format_info_updated_ = false;
};

#endif  // BYTE_BUFFER_DECODER_HPP_
