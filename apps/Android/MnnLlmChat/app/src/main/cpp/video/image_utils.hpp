#ifndef IMAGE_UTILS_HPP_
#define IMAGE_UTILS_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
namespace Express {
class VARP;
} // namespace Express
} // namespace MNN

class ImageUtils {
 public:
  enum class YUVFormat {
    UNKNOWN = 0,
    I420,
    NV12,
    NV21,
    AUTO
  };

  static constexpr int32_t kColorFormatYUV420Flexible = 0x7F420888;
  static constexpr int32_t kColorFormatYUV420Planar = 0x13;
  static constexpr int32_t kColorFormatYUV420SemiPlanar = 0x15;
  static constexpr int32_t kColorFormatYUV420PackedPlanar = 0x14;
  static constexpr int32_t kColorFormatYUV420PackedSemiPlanar = 0x27;

  struct YUVFormatInfo {
    YUVFormat format = YUVFormat::UNKNOWN;
    bool isValid = false;
    int outputWidth = 0;
    int outputHeight = 0;
    int totalSize = 0;
    int ySize = 0;
    int uvSize = 0;
    int stride = 0;
    int sliceHeight = 0;
  };

  static YUVFormatInfo detectYUVFormat(const uint8_t* data,
                                       int width,
                                       int height,
                                       int size);

  static YUVFormatInfo detectYUVFormatFromMediaCodec(
      int32_t colorFormat,
      int32_t stride,
      int32_t sliceHeight,
      int32_t width,
      int32_t height,
      int32_t cropLeft = 0,
      int32_t cropTop = 0,
      int32_t cropRight = 0,
      int32_t cropBottom = 0);

  static YUVFormatInfo detectYUVFormatWithFallback(
      const uint8_t* yuvData,
      int width,
      int height,
      int dataSize,
      int32_t colorFormat = 0,
      int32_t stride = 0,
      int32_t sliceHeight = 0);

  static const char* colorFormatToString(int32_t colorFormat);
  static bool isFlexibleColorFormat(int32_t colorFormat);
  static bool isVideoMimeType(const char* mimeType);

  // Legacy Pascal-case wrappers for existing call sites
  static inline YUVFormatInfo DetectYUVFormat(const uint8_t* data,
                                              int width,
                                              int height,
                                              int size) {
    return detectYUVFormat(data, width, height, size);
  }

  static inline YUVFormatInfo DetectYUVFormatFromMediaCodec(
      int32_t colorFormat,
      int32_t stride,
      int32_t sliceHeight,
      int32_t width,
      int32_t height,
      int32_t cropLeft = 0,
      int32_t cropTop = 0,
      int32_t cropRight = 0,
      int32_t cropBottom = 0) {
    return detectYUVFormatFromMediaCodec(colorFormat, stride, sliceHeight,
                                         width, height, cropLeft, cropTop,
                                         cropRight, cropBottom);
  }

  static inline YUVFormatInfo DetectYUVFormatWithFallback(
      const uint8_t* yuvData,
      int width,
      int height,
      int dataSize,
      int32_t colorFormat = 0,
      int32_t stride = 0,
      int32_t sliceHeight = 0) {
    return detectYUVFormatWithFallback(yuvData, width, height, dataSize,
                                       colorFormat, stride, sliceHeight);
  }
};

#endif  // IMAGE_UTILS_HPP_
