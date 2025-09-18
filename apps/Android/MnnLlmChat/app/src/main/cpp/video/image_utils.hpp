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

  static bool saveFrameAsImage(const uint8_t* yuvData,
                               int width,
                               int height,
                               const char* basePath,
                               bool saveJPEG = true,
                               bool savePNG = true,
                               bool saveBMP = true,
                               YUVFormat format = YUVFormat::AUTO);

  static bool yuvToRgb(const uint8_t* yuvData,
                       const YUVFormatInfo& formatInfo,
                       std::vector<uint8_t>& rgbData);

  static bool yuvToRgbWithStride(const uint8_t* yuvData,
                                 const YUVFormatInfo& formatInfo,
                                 std::vector<uint8_t>& rgbData);

  static bool saveTensorAsJPG(const MNN::Express::VARP& tensor,
                              const char* filename);
  static bool tensorToRgb(const MNN::Express::VARP& tensor,
                          int height,
                          int width,
                          int channels,
                          std::vector<uint8_t>& rgbData);

  static bool saveAsJPEG(const uint8_t* rgbData,
                         int width,
                         int height,
                         const char* filename);

  static bool saveAsBMP(const uint8_t* rgbData,
                        int width,
                        int height,
                        const char* filename);

  static bool saveRawYUVData(const uint8_t* yuvData,
                             int width,
                             int height,
                             int size,
                             const char* filename);

  static bool createTestPattern(int width,
                                int height,
                                const char* filename);

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

  static inline bool SaveFrameAsImage(const uint8_t* yuvData,
                                      int width,
                                      int height,
                                      const char* basePath,
                                      bool saveJPEG = true,
                                      bool savePNG = true,
                                      bool saveBMP = true,
                                      YUVFormat format = YUVFormat::AUTO) {
    return saveFrameAsImage(yuvData, width, height, basePath, saveJPEG,
                            savePNG, saveBMP, format);
  }

  static inline bool YUVToRgb(const uint8_t* yuvData,
                              const YUVFormatInfo& formatInfo,
                              std::vector<uint8_t>& rgbData) {
    return yuvToRgb(yuvData, formatInfo, rgbData);
  }

  static inline bool YUVToRgbWithStride(const uint8_t* yuvData,
                                        const YUVFormatInfo& formatInfo,
                                        std::vector<uint8_t>& rgbData) {
    return yuvToRgbWithStride(yuvData, formatInfo, rgbData);
  }

  static inline bool SaveTensorAsJPG(const MNN::Express::VARP& tensor,
                                     const char* filename) {
    return saveTensorAsJPG(tensor, filename);
  }

  static inline bool TensorToRgb(const MNN::Express::VARP& tensor,
                                 int height,
                                 int width,
                                 int channels,
                                 std::vector<uint8_t>& rgbData) {
    return tensorToRgb(tensor, height, width, channels, rgbData);
  }

  static inline bool SaveAsJPEG(const uint8_t* rgbData,
                                int width,
                                int height,
                                const char* filename) {
    return saveAsJPEG(rgbData, width, height, filename);
  }

  static inline bool SaveAsBMP(const uint8_t* rgbData,
                               int width,
                               int height,
                               const char* filename) {
    return saveAsBMP(rgbData, width, height, filename);
  }

  static inline bool SaveRawYUVData(const uint8_t* yuvData,
                                    int width,
                                    int height,
                                    int size,
                                    const char* filename) {
    return saveRawYUVData(yuvData, width, height, size, filename);
  }

  static inline bool CreateTestPattern(int width,
                                       int height,
                                       const char* filename) {
    return createTestPattern(width, height, filename);
  }

 private:
  static void i420ToRgb(const uint8_t* yuvData,
                        int width,
                        int height,
                        std::vector<uint8_t>& rgbData);
  static void nv12ToRgb(const uint8_t* yuvData,
                        int width,
                        int height,
                        std::vector<uint8_t>& rgbData);
  static void nv21ToRgb(const uint8_t* yuvData,
                        int width,
                        int height,
                        std::vector<uint8_t>& rgbData);
};

#endif  // IMAGE_UTILS_HPP_
