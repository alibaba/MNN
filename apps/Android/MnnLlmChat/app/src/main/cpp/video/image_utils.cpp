#include "image_utils.hpp"
#include <android/log.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "MNN/expr/Expr.hpp"

// Add stb_image support for proper JPEG/PNG saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

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
        case kColorFormatYUV420Flexible:
        case kColorFormatYUV420SemiPlanar:
        case kColorFormatYUV420PackedSemiPlanar: {
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
        
        case kColorFormatYUV420Planar:
        case kColorFormatYUV420PackedPlanar: {
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

// 统一的格式检测方法：优先使用decoder信息，备用格式检测
ImageUtils::YUVFormatInfo ImageUtils::detectYUVFormatWithFallback(
    const uint8_t* yuvData, int width, int height, int dataSize,
    int32_t colorFormat, int32_t stride, int32_t sliceHeight) {
    
    YUVFormatInfo info;
    
    // 优先使用MediaCodec的格式信息
    if (colorFormat != 0) {
        info = detectYUVFormatFromMediaCodec(colorFormat, stride, sliceHeight, width, height);
        if (info.isValid) {
            LOGV("detectYUVFormatWithFallback: using MediaCodec format info: %s, stride=%d, sliceHeight=%d", 
                 info.format == YUVFormat::NV12 ? "NV12" :
                 info.format == YUVFormat::I420 ? "I420" :
                 info.format == YUVFormat::NV21 ? "NV21" : "UNKNOWN",
                 info.stride, info.sliceHeight);
            return info;
        }
    }
    
    // 备用方案：使用数据检测
    LOGV("detectYUVFormatWithFallback: MediaCodec format not available, using data detection");
    info = detectYUVFormat(yuvData, width, height, dataSize);
    
    if (!info.isValid) {
        // 最后的备用方案：假设NV12格式
        LOGV("detectYUVFormatWithFallback: data detection failed, assuming NV12");
        info.format = YUVFormat::NV12;
        info.isValid = true;
        info.outputWidth = width;
        info.outputHeight = height;
        info.ySize = width * height;
        info.uvSize = width * height / 2;
        info.totalSize = info.ySize + info.uvSize;
        info.stride = width;
        info.sliceHeight = height;
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
    
    LOGV("saveFrameAsImage: saving %dx%d frame, format=%d", width, height, static_cast<int>(format));
    
    // Convert YUV to RGB first
    std::vector<uint8_t> rgbData;
    YUVFormatInfo formatInfo;
    
    if (format == YUVFormat::AUTO) {
        // Auto-detect format
        int expectedSize = width * height * 3 / 2;
        formatInfo = detectYUVFormat(yuvData, width, height, expectedSize);
        LOGV("saveFrameAsImage: auto-detected format=%d, valid=%d", 
             static_cast<int>(formatInfo.format), formatInfo.isValid);
    } else {
        // Use specified format
        formatInfo.format = format;
        formatInfo.isValid = true;
        formatInfo.outputWidth = width;
        formatInfo.outputHeight = height;
        formatInfo.ySize = width * height;
        formatInfo.uvSize = width * height / 2;
        formatInfo.totalSize = formatInfo.ySize + formatInfo.uvSize;
        formatInfo.stride = width;
        formatInfo.sliceHeight = height;
        LOGV("saveFrameAsImage: using specified format=%d", static_cast<int>(format));
    }
    
    if (!formatInfo.isValid) {
        LOGE("saveFrameAsImage: invalid YUV format");
        return false;
    }
    
    LOGV("saveFrameAsImage: format info - format=%d, size=%dx%d, ySize=%d, uvSize=%d, totalSize=%d", 
         static_cast<int>(formatInfo.format), formatInfo.outputWidth, formatInfo.outputHeight,
         formatInfo.ySize, formatInfo.uvSize, formatInfo.totalSize);
    
    if (!yuvToRgb(yuvData, formatInfo, rgbData)) {
        LOGE("saveFrameAsImage: YUV to RGB conversion failed");
        return false;
    }
    
    LOGV("saveFrameAsImage: YUV to RGB conversion successful, RGB size=%zu", rgbData.size());
    
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
        LOGE("yuvToRgb: invalid parameters - yuvData=%p, formatInfo.isValid=%d", 
             yuvData, formatInfo.isValid);
        return false;
    }
    
    LOGV("yuvToRgb: converting format=%d, size=%dx%d, stride=%d, sliceHeight=%d", 
         static_cast<int>(formatInfo.format), formatInfo.outputWidth, formatInfo.outputHeight,
         formatInfo.stride, formatInfo.sliceHeight);
    
    // Check if we need to handle stride/padding
    bool hasStridePadding = (formatInfo.stride != formatInfo.outputWidth) || 
                           (formatInfo.sliceHeight != formatInfo.outputHeight);
    
    if (hasStridePadding) {
        LOGV("yuvToRgb: detected stride padding - stride=%d vs width=%d, sliceHeight=%d vs height=%d",
             formatInfo.stride, formatInfo.outputWidth, formatInfo.sliceHeight, formatInfo.outputHeight);
        
        // Convert with stride handling
        return yuvToRgbWithStride(yuvData, formatInfo, rgbData);
    }
    
    // Standard conversion without stride
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
            LOGE("yuvToRgb: unsupported YUV format: %d", static_cast<int>(formatInfo.format));
            return false;
    }
    
    return !rgbData.empty();
}

bool ImageUtils::yuvToRgbWithStride(const uint8_t* yuvData, const YUVFormatInfo& formatInfo, 
                                   std::vector<uint8_t>& rgbData) {
    if (!yuvData || !formatInfo.isValid) {
        LOGE("yuvToRgbWithStride: invalid parameters");
        return false;
    }
    
    LOGV("yuvToRgbWithStride: handling stride conversion - stride=%d, sliceHeight=%d, output=%dx%d",
         formatInfo.stride, formatInfo.sliceHeight, formatInfo.outputWidth, formatInfo.outputHeight);
    
    rgbData.resize(formatInfo.outputWidth * formatInfo.outputHeight * 3);
    
    int yStride = formatInfo.stride;
    int uvStride = formatInfo.stride;
    int ySliceHeight = formatInfo.sliceHeight;
    int uvSliceHeight = formatInfo.sliceHeight / 2;
    
    const uint8_t* y = yuvData;
    const uint8_t* uv = yuvData + yStride * ySliceHeight;
    
    LOGV("yuvToRgbWithStride: Y plane size=%d, UV plane size=%d", 
         yStride * ySliceHeight, uvStride * uvSliceHeight);
    
    // Convert with stride handling
    for (int i = 0; i < formatInfo.outputHeight; i++) {
        for (int j = 0; j < formatInfo.outputWidth; j++) {
            int yVal = y[i * yStride + j];
            
            int uVal, vVal;
            if (formatInfo.format == YUVFormat::NV12) {
                // NV12: UV interleaved
                int chromaX = j / 2;
                int chromaY = i / 2;
                int uvIndex = chromaY * uvStride + chromaX * 2;
                uVal = uv[uvIndex];
                vVal = uv[uvIndex + 1];
            } else if (formatInfo.format == YUVFormat::NV21) {
                // NV21: VU interleaved
                int chromaX = j / 2;
                int chromaY = i / 2;
                int vuIndex = chromaY * uvStride + chromaX * 2;
                vVal = uv[vuIndex];
                uVal = uv[vuIndex + 1];
            } else if (formatInfo.format == YUVFormat::I420) {
                // I420: separate U and V planes
                int chromaX = j / 2;
                int chromaY = i / 2;
                int uIndex = chromaY * (uvStride / 2) + chromaX;
                int vIndex = chromaY * (uvStride / 2) + chromaX;
                uVal = uv[uIndex];
                vVal = uv[uvStride * uvSliceHeight / 2 + vIndex];
            } else {
                LOGE("yuvToRgbWithStride: unsupported format with stride: %d", 
                     static_cast<int>(formatInfo.format));
                return false;
            }
            
            // Log first few conversions for debugging
            if (i < 2 && j < 2) {
                LOGV("yuvToRgbWithStride: pixel[%d,%d] YUV=(%d,%d,%d)", i, j, yVal, uVal, vVal);
            }
            
            // YUV to RGB conversion
            int c = yVal - 16;
            int d = uVal - 128;
            int e = vVal - 128;
            
            int c298 = 298 * c;
            int r = (c298 + 409 * e + 128) >> 8;
            int g = (c298 - 100 * d - 208 * e + 128) >> 8;
            int b = (c298 + 516 * d + 128) >> 8;
            
            // Clamp values
            r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
            g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
            b = (b < 0) ? 0 : ((b > 255) ? 255 : b);
            
            int idx = (i * formatInfo.outputWidth + j) * 3;
            rgbData[idx] = static_cast<uint8_t>(r);
            rgbData[idx + 1] = static_cast<uint8_t>(g);
            rgbData[idx + 2] = static_cast<uint8_t>(b);
            
            // Log first few final RGB values for debugging
            if (i < 2 && j < 2) {
                LOGV("yuvToRgbWithStride: pixel[%d,%d] final RGB=(%d,%d,%d)", i, j, r, g, b);
            }
        }
    }
    
    LOGV("yuvToRgbWithStride: conversion completed, RGB size=%zu", rgbData.size());
    return true;
}

void ImageUtils::i420ToRgb(const uint8_t* yuvData, int width, int height, std::vector<uint8_t>& rgbData) {
    if (!yuvData || width <= 0 || height <= 0) {
        LOGE("Invalid parameters for i420ToRgb: yuvData=%p, width=%d, height=%d", yuvData, width, height);
        return;
    }
    
    LOGV("i420ToRgb: converting %dx%d I420 to RGB", width, height);
    
    rgbData.resize(width * height * 3);
    
    int ySize = width * height;
    int uvSize = width * height / 4;
    
    LOGV("i420ToRgb: ySize=%d, uvSize=%d, total expected size=%d", ySize, uvSize, ySize + 2 * uvSize);
    
    const uint8_t* y = yuvData;
    const uint8_t* u = yuvData + ySize;
    const uint8_t* v = yuvData + ySize + uvSize;
    
    // Log first few YUV values for debugging
    LOGV("i420ToRgb: first Y values: %d, %d, %d, %d", y[0], y[1], y[2], y[3]);
    LOGV("i420ToRgb: first U values: %d, %d, %d, %d", u[0], u[1], u[2], u[3]);
    LOGV("i420ToRgb: first V values: %d, %d, %d, %d", v[0], v[1], v[2], v[3]);
    
    // Optimized YUV to RGB conversion with reduced calculations
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int yVal = y[i * width + j];
            int uVal = u[(i/2) * (width/2) + (j/2)];
            int vVal = v[(i/2) * (width/2) + (j/2)];
            
            // Log first few conversions for debugging
            if (i < 2 && j < 2) {
                LOGV("i420ToRgb: pixel[%d,%d] YUV=(%d,%d,%d)", i, j, yVal, uVal, vVal);
            }
            
            // YUV to RGB conversion (optimized integer arithmetic)
            int c = yVal - 16;
            int d = uVal - 128;
            int e = vVal - 128;
            
            // Pre-calculate common terms
            int c298 = 298 * c;
            int r = (c298 + 409 * e + 128) >> 8;
            int g = (c298 - 100 * d - 208 * e + 128) >> 8;
            int b = (c298 + 516 * d + 128) >> 8;
            
            // Log first few RGB calculations for debugging
            if (i < 2 && j < 2) {
                LOGV("i420ToRgb: pixel[%d,%d] RGB calc: c=%d, d=%d, e=%d -> r=%d, g=%d, b=%d", 
                     i, j, c, d, e, r, g, b);
            }
            
            // Clamp values (optimized)
            r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
            g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
            b = (b < 0) ? 0 : ((b > 255) ? 255 : b);
            
            int idx = (i * width + j) * 3;
            rgbData[idx] = static_cast<uint8_t>(r);
            rgbData[idx + 1] = static_cast<uint8_t>(g);
            rgbData[idx + 2] = static_cast<uint8_t>(b);
            
            // Log first few final RGB values for debugging
            if (i < 2 && j < 2) {
                LOGV("i420ToRgb: pixel[%d,%d] final RGB=(%d,%d,%d)", i, j, r, g, b);
            }
        }
    }
    
    LOGV("i420ToRgb: conversion completed, RGB size=%zu", rgbData.size());
}

void ImageUtils::nv12ToRgb(const uint8_t* yuvData, int width, int height, std::vector<uint8_t>& rgbData) {
    if (!yuvData || width <= 0 || height <= 0) {
        LOGE("Invalid parameters for nv12ToRgb");
        return;
    }
    
    LOGV("nv12ToRgb: converting %dx%d NV12 to RGB", width, height);
    
    rgbData.resize(width * height * 3);
    
    int ySize = width * height;
    const uint8_t* y = yuvData;
    const uint8_t* uv = yuvData + ySize;
    
    // Log first few YUV values for debugging
    LOGV("nv12ToRgb: first Y values: %d, %d, %d, %d", y[0], y[1], y[2], y[3]);
    LOGV("nv12ToRgb: first UV values: %d, %d, %d, %d", uv[0], uv[1], uv[2], uv[3]);
    
    // Log more Y values to check for variation
    LOGV("nv12ToRgb: Y values at different positions: [0,0]=%d, [0,10]=%d, [10,0]=%d, [10,10]=%d", 
         y[0], y[10], y[10*width], y[10*width+10]);
    
    // Log UV values at different positions
    LOGV("nv12ToRgb: UV values at different positions: [0,0]=%d,%d, [0,10]=%d,%d, [10,0]=%d,%d, [10,10]=%d,%d",
         uv[0], uv[1], uv[10], uv[11], uv[10*width], uv[10*width+1], uv[10*width+10], uv[10*width+11]);
    
    // Optimized NV12 to RGB conversion
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int yVal = y[i * width + j];
            
            // Get UV values from interleaved UV plane
            // NV12 format: UVUVUV... (U and V are interleaved)
            int chromaX = j / 2;
            int chromaY = i / 2;
            int uvIndex = chromaY * width + chromaX * 2;
            int uVal = uv[uvIndex];
            int vVal = uv[uvIndex + 1];
            
            // Log first few conversions for debugging
            if (i < 2 && j < 2) {
                LOGV("nv12ToRgb: pixel[%d,%d] YUV=(%d,%d,%d)", i, j, yVal, uVal, vVal);
            }
            
            // YUV to RGB conversion (optimized integer arithmetic)
            int c = yVal - 16;
            int d = uVal - 128;
            int e = vVal - 128;
            
            // Pre-calculate common terms
            int c298 = 298 * c;
            int r = (c298 + 409 * e + 128) >> 8;
            int g = (c298 - 100 * d - 208 * e + 128) >> 8;
            int b = (c298 + 516 * d + 128) >> 8;
            
            // Log first few RGB calculations for debugging
            if (i < 2 && j < 2) {
                LOGV("nv12ToRgb: pixel[%d,%d] RGB calc: c=%d, d=%d, e=%d -> r=%d, g=%d, b=%d", 
                     i, j, c, d, e, r, g, b);
            }
            
            // Clamp values (optimized)
            r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
            g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
            b = (b < 0) ? 0 : ((b > 255) ? 255 : b);
            
            int idx = (i * width + j) * 3;
            rgbData[idx] = static_cast<uint8_t>(r);
            rgbData[idx + 1] = static_cast<uint8_t>(g);
            rgbData[idx + 2] = static_cast<uint8_t>(b);
            
            // Log first few final RGB values for debugging
            if (i < 2 && j < 2) {
                LOGV("nv12ToRgb: pixel[%d,%d] final RGB=(%d,%d,%d)", i, j, r, g, b);
            }
        }
    }
    
    LOGV("nv12ToRgb: conversion completed, RGB size=%zu", rgbData.size());
}

// Debug function to save raw YUV data for analysis
bool ImageUtils::saveRawYUVData(const uint8_t* yuvData, int width, int height, int size, const char* filename) {
    if (!yuvData || !filename || size <= 0) {
        LOGE("saveRawYUVData: invalid parameters");
        return false;
    }
    
    LOGV("saveRawYUVData: saving %dx%d YUV data (%d bytes) to: %s", width, height, size, filename);
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        LOGE("saveRawYUVData: failed to open file %s", filename);
        return false;
    }
    
    size_t written = fwrite(yuvData, 1, size, file);
    fclose(file);
    
    if (written == size) {
        LOGV("saveRawYUVData: successfully saved %zu bytes to %s", written, filename);
        return true;
    } else {
        LOGE("saveRawYUVData: failed to write all data, wrote %zu/%d bytes", written, size);
        return false;
    }
}

// Test function to create a simple test pattern
bool ImageUtils::createTestPattern(int width, int height, const char* filename) {
    LOGV("createTestPattern: creating %dx%d test pattern", width, height);
    
    std::vector<uint8_t> rgbData(width * height * 3);
    
    // Create a simple gradient pattern
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            
            // Create a gradient pattern
            rgbData[idx] = static_cast<uint8_t>((j * 255) / width);     // R
            rgbData[idx + 1] = static_cast<uint8_t>((i * 255) / height); // G
            rgbData[idx + 2] = static_cast<uint8_t>(128);                // B
        }
    }
    
    LOGV("createTestPattern: test pattern created, saving to: %s", filename);
    return saveAsJPEG(rgbData.data(), width, height, filename);
}

const char* ImageUtils::colorFormatToString(int32_t colorFormat) {
    switch (colorFormat) {
        case kColorFormatYUV420Flexible:
            return "YUV420Flexible";
        case kColorFormatYUV420Planar:
            return "YUV420Planar";
        case kColorFormatYUV420SemiPlanar:
            return "YUV420SemiPlanar";
        case kColorFormatYUV420PackedPlanar:
            return "YUV420PackedPlanar";
        case kColorFormatYUV420PackedSemiPlanar:
            return "YUV420PackedSemiPlanar";
        case 0x7F000100:
            return "QCOM_YUV420PackedSemiPlanar64x32Tile2m8ka";
        case 0x7FA30C00:
            return "QCOM_YUV420PackedSemiPlanar32m";
        default: {
            static char buffer[32];
            snprintf(buffer, sizeof(buffer), "0x%08X", colorFormat);
            return buffer;
        }
    }
}

bool ImageUtils::isFlexibleColorFormat(int32_t colorFormat) {
    return colorFormat == kColorFormatYUV420Flexible;
}

bool ImageUtils::isVideoMimeType(const char* mimeType) {
    if (!mimeType) {
        return false;
    }

    static const char* kVideoMimeTypes[] = {
        "video/mp4",
        "video/h264",
        "video/avc",
        "video/hevc",
        "video/h265",
        "video/vp8",
        "video/vp9",
        "video/av01",
        "video/3gpp",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
    };

    for (const char* type : kVideoMimeTypes) {
        if (strcmp(mimeType, type) == 0) {
            return true;
        }
    }

    return strncmp(mimeType, "video/", 6) == 0;
}

void ImageUtils::nv21ToRgb(const uint8_t* yuvData, int width, int height, std::vector<uint8_t>& rgbData) {
    if (!yuvData || width <= 0 || height <= 0) {
        LOGE("Invalid parameters for nv21ToRgb");
        return;
    }
    
    rgbData.resize(width * height * 3);
    
    int ySize = width * height;
    const uint8_t* y = yuvData;
    const uint8_t* vu = yuvData + ySize;
    
    // Optimized NV21 to RGB conversion
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int yVal = y[i * width + j];
            
            // Get VU values from interleaved VU plane (V first, then U)
            int chromaX = j / 2;
            int chromaY = i / 2;
            int vuIndex = chromaY * width + chromaX * 2;
            int vVal = vu[vuIndex];
            int uVal = vu[vuIndex + 1];
            
            // YUV to RGB conversion (optimized integer arithmetic)
            int c = yVal - 16;
            int d = uVal - 128;
            int e = vVal - 128;
            
            // Pre-calculate common terms
            int c298 = 298 * c;
            int r = (c298 + 409 * e + 128) >> 8;
            int g = (c298 - 100 * d - 208 * e + 128) >> 8;
            int b = (c298 + 516 * d + 128) >> 8;
            
            // Clamp values (optimized)
            r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
            g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
            b = (b < 0) ? 0 : ((b > 255) ? 255 : b);
            
            int idx = (i * width + j) * 3;
            rgbData[idx] = static_cast<uint8_t>(r);
            rgbData[idx + 1] = static_cast<uint8_t>(g);
            rgbData[idx + 2] = static_cast<uint8_t>(b);
        }
    }
}

bool ImageUtils::saveAsJPEG(const uint8_t* rgbData, int width, int height, const char* filename) {
    if (!rgbData || !filename || width <= 0 || height <= 0) {
        LOGE("saveAsJPEG: invalid parameters - rgbData=%p, filename=%s, width=%d, height=%d", 
             rgbData, filename, width, height);
        return false;
    }
    
    LOGV("saveAsJPEG: saving %dx%d RGB data to: %s", width, height, filename);
    
    // Log first few RGB values for debugging
    if (width * height >= 4) {
        LOGV("saveAsJPEG: first RGB values: R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, R=%d,G=%d,B=%d",
             rgbData[0], rgbData[1], rgbData[2], rgbData[3], rgbData[4], rgbData[5],
             rgbData[6], rgbData[7], rgbData[8], rgbData[9], rgbData[10], rgbData[11]);
    }
    
    // Use stb_image_write for proper JPEG encoding
    LOGV("saveAsJPEG: calling stbi_write_jpg with quality=90");
    int result = stbi_write_jpg(filename, width, height, 3, rgbData, 90);
    if (result) {
        LOGV("saveAsJPEG: successfully saved JPEG: %s (result=%d)", filename, result);
        return true;
    } else {
        LOGE("saveAsJPEG: failed to save JPEG: %s (result=%d)", filename, result);
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
bool ImageUtils::saveTensorAsJPG(const MNN::Express::VARP& tensor, const char* filename) {
    if (!tensor.get() || !filename) {
        LOGE("saveTensorAsJPG: invalid parameters - tensor=%p, filename=%s", tensor.get(), filename);
        return false;
    }
    
    LOGV("saveTensorAsJPG: starting to save tensor to: %s", filename);
    
    // Get tensor info
    auto info = tensor->getInfo();
    if (!info) {
        LOGE("saveTensorAsJPG: failed to get tensor info");
        return false;
    }
    
    // Check tensor dimensions
    auto dims = info->dim;
    LOGV("saveTensorAsJPG: tensor dimensions: [%s]", 
         [&dims]() {
             std::string dimStr;
             for (size_t i = 0; i < dims.size(); i++) {
                 if (i > 0) dimStr += ", ";
                 dimStr += std::to_string(dims[i]);
             }
             return dimStr;
         }().c_str());
    
    if (dims.size() < 2) {
        LOGE("saveTensorAsJPG: tensor must have at least 2 dimensions, got %zu", dims.size());
        return false;
    }
    
    int height, width, channels;
    
    // Handle different tensor layouts (NHWC, NCHW, etc.)
    if (dims.size() == 3) {
        // Assume HWC format
        height = dims[0];
        width = dims[1];
        channels = dims[2];
        LOGV("saveTensorAsJPG: detected HWC format");
    } else if (dims.size() == 4) {
        // Assume NHWC format, take first batch
        height = dims[1];
        width = dims[2];
        channels = dims[3];
        LOGV("saveTensorAsJPG: detected NHWC format");
    } else {
        LOGE("saveTensorAsJPG: unsupported tensor dimensions: %zu", dims.size());
        return false;
    }
    
    LOGV("saveTensorAsJPG: tensor dimensions: %dx%dx%d", height, width, channels);
    LOGV("saveTensorAsJPG: tensor data type: %d", static_cast<int>(info->type.code));
    
    // Convert tensor to RGB data
    std::vector<uint8_t> rgbData;
    LOGV("saveTensorAsJPG: calling tensorToRgb...");
    if (!tensorToRgb(tensor, height, width, channels, rgbData)) {
        LOGE("saveTensorAsJPG: failed to convert tensor to RGB");
        return false;
    }
    
    LOGV("saveTensorAsJPG: tensorToRgb completed, RGB data size: %zu", rgbData.size());
    
    // Log first few RGB values for debugging
    if (rgbData.size() >= 12) {
        LOGV("saveTensorAsJPG: first RGB values: R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, R=%d,G=%d,B=%d, R=%d,G=%d,B=%d",
             rgbData[0], rgbData[1], rgbData[2], rgbData[3], rgbData[4], rgbData[5],
             rgbData[6], rgbData[7], rgbData[8], rgbData[9], rgbData[10], rgbData[11]);
    }
    
    // Save as JPG using proper JPEG encoding
    LOGV("saveTensorAsJPG: calling saveAsJPEG...");
    bool result = saveAsJPEG(rgbData.data(), width, height, filename);
    if (result) {
        LOGV("saveTensorAsJPG: successfully saved JPEG: %s", filename);
    } else {
        LOGE("saveTensorAsJPG: failed to save JPEG: %s", filename);
    }
    return result;
}

// Convert MNN tensor to RGB data
bool ImageUtils::tensorToRgb(const MNN::Express::VARP& tensor, int height, int width, int channels,
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
    
    // Check if dimensions match expected format
    bool dimensionsMatch = false;
    if (dims.size() == 3) {
        // Check if it's HWC format
        if (dims[0] == height && dims[1] == width && dims[2] == channels) {
            dimensionsMatch = true;
            LOGV("tensorToRgb: tensor is in HWC format");
        }
        // Check if it's CHW format
        else if (dims[0] == channels && dims[1] == height && dims[2] == width) {
            dimensionsMatch = true;
            LOGV("tensorToRgb: tensor is in CHW format");
        }
    }
    
    if (!dimensionsMatch) {
        LOGV("tensorToRgb: tensor dimensions don't match expected format, trying to adapt");
    }
    
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
        
        LOGV("tensorToRgb: tensor total size: %d, expected size: %d", tensorSize, expectedSize);
        
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
