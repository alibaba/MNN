#ifndef BYTE_BUFFER_DECODER_HPP_
#define BYTE_BUFFER_DECODER_HPP_

#include "image_utils.hpp"
#include "video_decoder.hpp"

class ByteBufferDecoder : public VideoDecoder {
 public:
  ByteBufferDecoder();
  ~ByteBufferDecoder() override;

  bool Configure() override;
  
  bool ConvertYuvToRgb(const std::vector<uint8_t>& yuv_data,
                       const ImageUtils::YUVFormatInfo& format_info,
                       std::vector<uint8_t>* out_rgb) override;

  bool GetNextFrame(std::vector<uint8_t>* out_yuv,
                    ImageUtils::YUVFormatInfo* format_info,
                    int64_t* out_pts_us,
                    long* native_ms,
                    bool* out_eos) override;

  float GetDetectedFps() const override { return detected_fps_; }

 private:
  bool ConfigureByteBuffer();
  bool UpdateOutputFormatInfo();
  
  // Feed input data to codec (separated from YUV decoding)
  bool FeedInputToCodec(bool* out_eos);
  
  // Simple function to get next available YUV frame from codec
  bool GetNextYuvFrame(std::vector<uint8_t>* out_yuv,
                       ImageUtils::YUVFormatInfo* format_info,
                       int64_t* out_pts_us,
                       long* native_ms,
                       bool* out_eos);

  float detected_fps_ = -1.0f;  // Detected FPS from output format

  // Stream state tracking to robustly drain codec
  bool input_eos_ = false;   // End-of-stream seen on input side
  bool output_eos_ = false;  // End-of-stream seen on output side
};

#endif  // BYTE_BUFFER_DECODER_HPP_
