#include "video/image_utils.hpp"
#include <android/log.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "MNN/expr/Expr.hpp"

// Add stb_image support for proper JPEG/PNG saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TAG "ImageUtils"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

ImageUtils::YUVFormatInfo ImageUtils::detectYUVFormat(const uint8_t* data, int width, int height, int size) {
    YUVFormatInfo info;
    
    if (!data || width <= 0 || height <= 0 || size <= 0) {
        return info;
    }
    
    // Calculate expected sizes for different formats
    int expectedI420Size = width * height * 3 / 2;  // Y + U/4 + V/4
    int expectedNV12Size = width * height * 3 / 2;  // Y + UV/2
    int expectedNV21Size = width * height * 3 / 2;  // Y + VU/2
    
    LOGV("detectYUVFormat: size=%d, expected I420/NV12/NV21=%d, dimensions=%dx%d", 
         size, expectedI420Size, width, height);
    
    // First, try to match by size
    if (size == expectedI420Size || size == expectedNV12Size || size == expectedNV21Size) {
        info.isValid = true;
        info.outputWidth = width;
        info.outputHeight = height;
        info.totalSize = size;
        info.ySize = width * height;
        info.uvSize = width * height / 2;
        info.stride = width;
        info.sliceHeight = height;
        
        // Try to detect format by analyzing the data
        // This is a heuristic approach and may not be 100% accurate
        
        // For now, assume NV12 as it's the most common format on Android
        info.format = YUVFormat::NV12;
        
        LOGV("detectYUVFormat: detected format=NV12 (assumed), size matches expected");
        return info;
    }
    
    // If size doesn't match exactly, try with some tolerance for padding
    int tolerance = width * 16; // Allow for some padding
    if (size >= expectedI420Size - tolerance && size <= expectedI420Size + tolerance) {
        info.isValid = true;
        info.outputWidth = width;
        info.outputHeight = height;
        info.totalSize = size;
        info.ySize = width * height;
        info.uvSize = size - info.ySize;
        info.stride = width;
        info.sliceHeight = height;
        info.format = YUVFormat::NV12; // Default assumption
        
        LOGV("detectYUVFormat: detected format=NV12 (with padding), size=%d vs expected=%d", 
             size, expectedI420Size);
        return info;
    }
    
    LOGE("detectYUVFormat: failed to detect format, size=%d doesn't match expected=%d", 
         size, expectedI420Size);
    return info;
}

ImageUtils::YUVFormatInfo ImageUtils::detectYUVFormatFromMediaCodec(
    int32_t colorFormat, int32_t stride, int32_t sliceHeight, 
    int32_t width, int32_t height,
    int32_t cropLeft, int32_t cropTop, 
    int32_t cropRight, int32_t cropBottom) {
    
    YUVFormatInfo info;
    
    if (width <= 0 || height <= 0) {
        LOGE("detectYUVFormatFromMediaCodec: invalid dimensions %dx%d", width, height);
        return info;
    }
    
    // Use actual dimensions or cropped dimensions
    int outputWidth = (cropRight > cropLeft) ? (cropRight - cropLeft + 1) : width;
    int outputHeight = (cropBottom > cropTop) ? (cropBottom - cropTop + 1) : height;
    
    // Use stride and sliceHeight if available, otherwise use width/height
    int actualStride = (stride > 0) ? stride : width;
    int actualSliceHeight = (sliceHeight > 0) ? sliceHeight : height;
    
    LOGV("detectYUVFormatFromMediaCodec: colorFormat=0x%x (%s), stride=%d, sliceHeight=%d", 
         colorFormat, ImageUtils::colorFormatToString(colorFormat), actualStride, actualSliceHeight);
    LOGV("detectYUVFormatFromMediaCodec: dimensions=%dx%d, output=%dx%d", 
         width, height, outputWidth, outputHeight);
    
    info.outputWidth = outputWidth;
    info.outputHeight = outputHeight;
    info.stride = actualStride;
    info.sliceHeight = actualSliceHeight;
    
    switch (colorFormat) {
        case ImageUtils::kColorFormatYUV420Flexible:
        case ImageUtils::kColorFormatYUV420SemiPlanar:
        case ImageUtils::kColorFormatYUV420PackedSemiPlanar: {
            // NV12 format (most common on Android)
            info.format = YUVFormat::NV12;
            info.ySize = actualStride * actualSliceHeight;
            info.uvSize = actualStride * actualSliceHeight / 2;
            info.totalSize = info.ySize + info.uvSize;
            info.isValid = true;
            LOGV("detectYUVFormatFromMediaCodec: detected NV12, ySize=%d, uvSize=%d, totalSize=%d", 
                 info.ySize, info.uvSize, info.totalSize);
            break;
        }
        
        case ImageUtils::kColorFormatYUV420Planar:
        case ImageUtils::kColorFormatYUV420PackedPlanar: {
            // I420 format
            info.format = YUVFormat::I420;
            info.ySize = actualStride * actualSliceHeight;
            info.uvSize = actualStride * actualSliceHeight / 2;
            info.totalSize = info.ySize + info.uvSize;
            info.isValid = true;
            LOGV("detectYUVFormatFromMediaCodec: detected I420, ySize=%d, uvSize=%d, totalSize=%d", 
                 info.ySize, info.uvSize, info.totalSize);
            break;
        }
        
        default: {
            // For unknown formats, assume NV12 as fallback
            info.format = YUVFormat::NV12;
            info.ySize = actualStride * actualSliceHeight;
            info.uvSize = actualStride * actualSliceHeight / 2;
            info.totalSize = info.ySize + info.uvSize;
            info.isValid = true;
            LOGV("detectYUVFormatFromMediaCodec: unknown format 0x%x, assuming NV12", colorFormat);
            break;
        }
    }
    
    return info;
}

bool ImageUtils::saveFrameAsImage(const uint8_t* yuvData, int width, int height,
                                const char* basePath, bool saveJPEG, 
                                bool savePNG, bool saveBMP,
                                YUVFormat format) {
    if (!yuvData || !basePath || width <= 0 || height <= 0) {
        LOGE("saveFrameAsImage: invalid parameters");
        return false;
    }
    
    // Convert YUV to RGB first
    std::vector<uint8_t> rgbData;
    YUVFormatInfo formatInfo;
    
    if (format == YUVFormat::AUTO) {
        // Auto-detect format
        int expectedSize = width * height * 3 / 2;
        formatInfo = detectYUVFormat(yuvData, width, height, expectedSize);
    } else {
        // Use specified format
        formatInfo.format = format;
        formatInfo.isValid = true;
        formatInfo.outputWidth = width;
        formatInfo.outputHeight = height;
        formatInfo.ySize = width * height;
        formatInfo.uvSize = width * height / 2;
        formatInfo.totalSize = formatInfo.ySize + formatInfo.uvSize;
    }
    
    if (!formatInfo.isValid) {
        LOGE("saveFrameAsImage: invalid YUV format");
        return false;
    }
    
    if (!yuvToRgb(yuvData, formatInfo, rgbData)) {
        LOGE("saveFrameAsImage: YUV to RGB conversion failed");
        return false;
    }
    
    bool success = true;
    
    // Save as JPEG
    if (saveJPEG) {
        std::string jpegPath = std::string(basePath) + ".jpg";
        if (!saveAsJPEG(rgbData.data(), width, height, jpegPath.c_str())) {
            LOGE("saveFrameAsImage: failed to save JPEG: %s", jpegPath.c_str());
            success = false;
        } else {
            LOGV("saveFrameAsImage: saved JPEG: %s", jpegPath.c_str());
        }
    }
    
    // Save as JPEG
    if (savePNG) {
        std::string jpegPath = std::string(basePath) + ".jpg";
        if (!saveAsJPEG(rgbData.data(), width, height, jpegPath.c_str())) {
            LOGE("saveFrameAsImage: failed to save JPEG: %s", jpegPath.c_str());
            success = false;
        } else {
            LOGV("saveFrameAsImage: saved JPEG: %s", jpegPath.c_str());
        }
    }
    
    // Save as BMP
    if (saveBMP) {
        std::string bmpPath = std::string(basePath) + ".bmp";
        if (!saveAsBMP(rgbData.data(), width, height, bmpPath.c_str())) {
            LOGE("saveFrameAsImage: failed to save BMP: %s", bmpPath.c_str());
            success = false;
        } else {
            LOGV("saveFrameAsImage: saved BMP: %s", bmpPath.c_str());
        }
    }
    
    return success;
}

bool ImageUtils::yuvToRgb(const uint8_t* yuvData, const YUVFormatInfo& formatInfo, 
                         std::vector<uint8_t>& rgbData) {
    if (!yuvData || !formatInfo.isValid) {
        return false;
    }
    
    switch (formatInfo.format) {
        case YUVFormat::I420:
            i420ToRgb(yuvData, formatInfo.outputWidth, formatInfo.outputHeight, rgbData);
            break;
        case YUVFormat::NV12:
            nv12ToRgb(yuvData, formatInfo.outputWidth, formatInfo.outputHeight, rgbData);
            break;
        case YUVFormat::NV21:
            nv21ToRgb(yuvData, formatInfo.outputWidth, formatInfo.outputHeight, rgbData);
            break;
        default:
            LOGE("yuvToRgb: unsupported YUV format");
            return false;
    }
    
    return !rgbData.empty();
}

void ImageUtils::i420ToRgb(const uint8_t* yuvData, int width, int height, std::vector<uint8_t>& rgbData) {
    rgbData.resize(width * height * 3);
    
    const uint8_t* yPlane = yuvData;
    const uint8_t* uPlane = yuvData + width * height;
    const uint8_t* vPlane = yuvData + width * height + width * height / 4;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int yIndex = y * width + x;
            int uvIndex = (y / 2) * (width / 2) + (x / 2);
            
            int Y = yPlane[yIndex];
            int U = uPlane[uvIndex] - 128;
            int V = vPlane[uvIndex] - 128;
            
            // YUV to RGB conversion
            int R = Y + (1.370705 * V);
            int G = Y - (0.337633 * U) - (0.698001 * V);
            int B = Y + (1.732446 * U);
            
            // Clamp values
            R = std::max(0, std::min(255, R));
            G = std::max(0, std::min(255, G));
            B = std::max(0, std::min(255, B));
            
            int rgbIndex = yIndex * 3;
            rgbData[rgbIndex] = R;
            rgbData[rgbIndex + 1] = G;
            rgbData[rgbIndex + 2] = B;
        }
    }
}

void ImageUtils::nv12ToRgb(const uint8_t* yuvData, int width, int height, std::vector<uint8_t>& rgbData) {
    rgbData.resize(width * height * 3);
    
    const uint8_t* yPlane = yuvData;
    const uint8_t* uvPlane = yuvData + width * height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int yIndex = y * width + x;
            int uvIndex = ((y / 2) * width + (x & ~1));
            
            int Y = yPlane[yIndex];
            int U = uvPlane[uvIndex] - 128;     // U component
            int V = uvPlane[uvIndex + 1] - 128; // V component
            
            // YUV to RGB conversion
            int R = Y + (1.370705 * V);
            int G = Y - (0.337633 * U) - (0.698001 * V);
            int B = Y + (1.732446 * U);
            
            // Clamp values
            R = std::max(0, std::min(255, R));
            G = std::max(0, std::min(255, G));
            B = std::max(0, std::min(255, B));
            
            int rgbIndex = yIndex * 3;
            rgbData[rgbIndex] = R;
            rgbData[rgbIndex + 1] = G;
            rgbData[rgbIndex + 2] = B;
        }
    }
}

void ImageUtils::nv21ToRgb(const uint8_t* yuvData, int width, int height, std::vector<uint8_t>& rgbData) {
    rgbData.resize(width * height * 3);
    
    const uint8_t* yPlane = yuvData;
    const uint8_t* vuPlane = yuvData + width * height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int yIndex = y * width + x;
            int vuIndex = ((y / 2) * width + (x & ~1));
            
            int Y = yPlane[yIndex];
            int V = vuPlane[vuIndex] - 128;     // V component
            int U = vuPlane[vuIndex + 1] - 128; // U component
            
            // YUV to RGB conversion
            int R = Y + (1.370705 * V);
            int G = Y - (0.337633 * U) - (0.698001 * V);
            int B = Y + (1.732446 * U);
            
            // Clamp values
            R = std::max(0, std::min(255, R));
            G = std::max(0, std::min(255, G));
            B = std::max(0, std::min(255, B));
            
            int rgbIndex = yIndex * 3;
            rgbData[rgbIndex] = R;
            rgbData[rgbIndex + 1] = G;
            rgbData[rgbIndex + 2] = B;
        }
    }
}

bool ImageUtils::saveAsJPEG(const uint8_t* rgbData, int width, int height, const char* filename) {
    if (!rgbData || !filename || width <= 0 || height <= 0) {
        LOGE("saveAsJPEG: invalid parameters");
        return false;
    }
    
    // Use stb_image_write for proper JPEG encoding
    int result = stbi_write_jpg(filename, width, height, 3, rgbData, 90);
    if (result) {
        LOGV("saveAsJPEG: successfully saved JPEG: %s", filename);
        return true;
    } else {
        LOGE("saveAsJPEG: failed to save JPEG: %s", filename);
        return false;
    }
}



bool ImageUtils::saveAsBMP(const uint8_t* rgbData, int width, int height, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        LOGE("saveAsBMP: failed to open file %s", filename);
        return false;
    }
    
    // BMP header
    uint32_t fileSize = 54 + width * height * 3;
    uint32_t dataOffset = 54;
    uint32_t imageSize = width * height * 3;
    
    // File header (14 bytes)
    fwrite("BM", 1, 2, file);                    // Signature
    fwrite(&fileSize, 4, 1, file);              // File size
    uint32_t reserved = 0;
    fwrite(&reserved, 4, 1, file);              // Reserved
    fwrite(&dataOffset, 4, 1, file);            // Data offset
    
    // Info header (40 bytes)
    uint32_t headerSize = 40;
    fwrite(&headerSize, 4, 1, file);            // Header size
    fwrite(&width, 4, 1, file);                 // Width
    fwrite(&height, 4, 1, file);                // Height
    uint16_t planes = 1;
    fwrite(&planes, 2, 1, file);                // Planes
    uint16_t bitsPerPixel = 24;
    fwrite(&bitsPerPixel, 2, 1, file);          // Bits per pixel
    uint32_t compression = 0;
    fwrite(&compression, 4, 1, file);           // Compression
    fwrite(&imageSize, 4, 1, file);             // Image size
    uint32_t xPelsPerMeter = 0;
    fwrite(&xPelsPerMeter, 4, 1, file);         // X pixels per meter
    uint32_t yPelsPerMeter = 0;
    fwrite(&yPelsPerMeter, 4, 1, file);         // Y pixels per meter
    uint32_t colorsUsed = 0;
    fwrite(&colorsUsed, 4, 1, file);            // Colors used
    uint32_t colorsImportant = 0;
    fwrite(&colorsImportant, 4, 1, file);       // Important colors
    
    // Write image data (BMP stores bottom-to-top, BGR format)
    std::vector<uint8_t> bgrLine(width * 3);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int rgbIndex = (y * width + x) * 3;
            int bgrIndex = x * 3;
            bgrLine[bgrIndex] = rgbData[rgbIndex + 2];     // B
            bgrLine[bgrIndex + 1] = rgbData[rgbIndex + 1]; // G
            bgrLine[bgrIndex + 2] = rgbData[rgbIndex];     // R
        }
        fwrite(bgrLine.data(), 1, width * 3, file);
    }
    
    fclose(file);
    LOGV("saveAsBMP: saved BMP file: %s", filename);
    return true;
}

// Save MNN tensor as JPG image
bool ImageUtils::saveTensorAsJPG(MNN::Express::VARP tensor, const char* filename) {
    if (!tensor.get() || !filename) {
        LOGE("saveTensorAsJPG: invalid parameters");
        return false;
    }
    
    // Get tensor info
    auto info = tensor->getInfo();
    if (!info) {
        LOGE("saveTensorAsJPG: failed to get tensor info");
        return false;
    }
    
    // Check tensor dimensions
    auto dims = info->dim;
    if (dims.size() < 2) {
        LOGE("saveTensorAsJPG: tensor must have at least 2 dimensions");
        return false;
    }
    
    int height, width, channels;
    
    // Handle different tensor layouts (NHWC, NCHW, etc.)
    if (dims.size() == 3) {
        // Assume HWC format
        height = dims[0];
        width = dims[1];
        channels = dims[2];
    } else if (dims.size() == 4) {
        // Assume NHWC format, take first batch
        height = dims[1];
        width = dims[2];
        channels = dims[3];
    } else {
        LOGE("saveTensorAsJPG: unsupported tensor dimensions: %zu", dims.size());
        return false;
    }
    
    LOGV("saveTensorAsJPG: tensor dimensions: %dx%dx%d", height, width, channels);
    
    // Convert tensor to RGB data
    std::vector<uint8_t> rgbData;
    if (!tensorToRgb(tensor, height, width, channels, rgbData)) {
        LOGE("saveTensorAsJPG: failed to convert tensor to RGB");
        return false;
    }
    
    // Save as JPG using proper JPEG encoding
    bool result = saveAsJPEG(rgbData.data(), width, height, filename);
    if (result) {
        LOGV("saveTensorAsJPG: successfully saved JPEG: %s", filename);
    } else {
        LOGE("saveTensorAsJPG: failed to save JPEG: %s", filename);
    }
    return result;
}

// Convert MNN tensor to RGB data
bool ImageUtils::tensorToRgb(MNN::Express::VARP tensor, int height, int width, int channels, 
                           std::vector<uint8_t>& rgbData) {
    if (!tensor.get()) {
        LOGE("tensorToRgb: tensor is null");
        return false;
    }
    
    // Get tensor info
    auto info = tensor->getInfo();
    if (!info) {
        LOGE("tensorToRgb: failed to get tensor info");
        return false;
    }
    
    auto dims = info->dim;
    LOGV("tensorToRgb: tensor dimensions: [%s]", 
         [&dims]() {
             std::string dimStr;
             for (size_t i = 0; i < dims.size(); i++) {
                 if (i > 0) dimStr += ", ";
                 dimStr += std::to_string(dims[i]);
             }
             return dimStr;
         }().c_str());
    
    // Validate dimensions
    if (dims.size() < 2) {
        LOGE("tensorToRgb: tensor must have at least 2 dimensions, got %zu", dims.size());
        return false;
    }
    
    // Calculate expected size
    int expectedSize = height * width * channels;
    LOGV("tensorToRgb: expected size: %d (H=%d, W=%d, C=%d)", expectedSize, height, width, channels);
    
    // Try to read tensor data as uint8 first
    auto ptr = tensor->readMap<uint8_t>();
    if (ptr) {
        LOGV("tensorToRgb: reading as uint8 data");
        rgbData.resize(expectedSize);
        
        // Check if tensor size matches expected size
        int tensorSize = 1;
        for (auto dim : dims) {
            tensorSize *= dim;
        }
        
        if (tensorSize == expectedSize) {
            memcpy(rgbData.data(), ptr, expectedSize);
            LOGV("tensorToRgb: successfully copied %d bytes of uint8 data", expectedSize);
            return true;
        } else {
            LOGV("tensorToRgb: tensor size (%d) doesn't match expected size (%d), trying to reshape", 
                 tensorSize, expectedSize);
            
            // Try to reshape or copy what we can
            int copySize = std::min(tensorSize, expectedSize);
            memcpy(rgbData.data(), ptr, copySize);
            
            // Fill remaining with zeros if needed
            if (copySize < expectedSize) {
                memset(rgbData.data() + copySize, 0, expectedSize - copySize);
            }
            
            LOGV("tensorToRgb: copied %d bytes, filled %d bytes with zeros", 
                 copySize, expectedSize - copySize);
            return true;
        }
    }
    
    // Try to read as float data
    auto floatPtr = tensor->readMap<float>();
    if (floatPtr) {
        LOGV("tensorToRgb: reading as float data");
        rgbData.resize(expectedSize);
        
        int tensorSize = 1;
        for (auto dim : dims) {
            tensorSize *= dim;
        }
        
        int copySize = std::min(tensorSize, expectedSize);
        
        // Convert float to uint8
        for (int i = 0; i < copySize; i++) {
            float val = floatPtr[i];
            // Handle different float ranges
            if (val <= 1.0f && val >= 0.0f) {
                // Assume normalized [0,1] range
                rgbData[i] = static_cast<uint8_t>(val * 255.0f);
            } else if (val <= 255.0f && val >= 0.0f) {
                // Assume [0,255] range
                rgbData[i] = static_cast<uint8_t>(val);
            } else {
                // Clamp to [0,255]
                rgbData[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
            }
        }
        
        // Fill remaining with zeros if needed
        if (copySize < expectedSize) {
            memset(rgbData.data() + copySize, 0, expectedSize - copySize);
        }
        
        LOGV("tensorToRgb: converted %d float values to uint8", copySize);
        return true;
    }
    
    LOGE("tensorToRgb: failed to read tensor data as uint8 or float");
    return false;
}
