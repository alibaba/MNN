#ifndef BYTE_BUFFER_DECODER_HPP_
#define BYTE_BUFFER_DECODER_HPP_

#include "image_utils.hpp"
#include "video_decoder.hpp"

class ByteBufferDecoder : public VideoDecoder {
 public:
  ByteBufferDecoder();
  ~ByteBufferDecoder() override;

  bool Configure() override;
  bool DecodeFrame(int64_t target_timestamp_us,
                   int64_t tolerance_us,
                   std::vector<uint8_t>* out_rgb,
                   int64_t* out_pts_us,
                   long* native_ms,
                   bool* out_eos) override;

  bool DecodeFrameToTensor(int64_t target_timestamp_us,
                           int64_t tolerance_us,
                           MNN::Express::VARP* out_tensor,
                           int64_t* out_pts_us,
                           long* native_ms,
                           bool* out_eos) override;

 private:
  bool ConfigureByteBuffer();
  bool UpdateOutputFormatInfo();
  bool DecodeFrameToYuv(int64_t target_timestamp_us,
                        int64_t tolerance_us,
                        std::vector<uint8_t>* out_yuv,
                        ImageUtils::YUVFormatInfo* format_info,
                        int64_t* out_pts_us,
                        long* native_ms,
                        bool* out_eos);

  ImageUtils::YUVFormatInfo output_format_info_;
  bool format_info_updated_ = false;
};

#endif  // BYTE_BUFFER_DECODER_HPP_
