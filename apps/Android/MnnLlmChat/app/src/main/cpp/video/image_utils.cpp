#include "image_utils.hpp"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "MNN/expr/Expr.hpp"
#include "video_utils.hpp"

#define TAG "ImageUtils"

ImageUtils::YUVFormatInfo ImageUtils::detectYUVFormat(const uint8_t* data, int width, int height, int size) {
    YUVFormatInfo info;
    
    if (!data || width <= 0 || height <= 0 || size <= 0) {
        return info;
    }
    
    // Calculate expected sizes for different formats
    int expectedI420Size = width * height * 3 / 2;  // Y + U/4 + V/4
    int expectedNV12Size = width * height * 3 / 2;  // Y + UV/2
    int expectedNV21Size = width * height * 3 / 2;  // Y + VU/2
    
    VIDEO_LOGV(TAG,"detectYUVFormat: size=%d, expected I420/NV12/NV21=%d, dimensions=%dx%d", 
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
        
        VIDEO_LOGV(TAG,"detectYUVFormat: detected format=NV12 (assumed), size matches expected");
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
        
        VIDEO_LOGV(TAG,"detectYUVFormat: detected format=NV12 (with padding), size=%d vs expected=%d", 
             size, expectedI420Size);
        return info;
    }
    
    VIDEO_LOGE(TAG,"detectYUVFormat: failed to detect format, size=%d doesn't match expected=%d", 
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
        VIDEO_LOGE(TAG,"detectYUVFormatFromMediaCodec: invalid dimensions %dx%d", width, height);
        return info;
    }
    
    // Use actual dimensions or cropped dimensions
    int outputWidth = (cropRight > cropLeft) ? (cropRight - cropLeft + 1) : width;
    int outputHeight = (cropBottom > cropTop) ? (cropBottom - cropTop + 1) : height;
    
    // Use stride and sliceHeight if available, otherwise use width/height
    int actualStride = (stride > 0) ? stride : width;
    int actualSliceHeight = (sliceHeight > 0) ? sliceHeight : height;
    
    VIDEO_LOGV(TAG,"detectYUVFormatFromMediaCodec: colorFormat=0x%x (%s), stride=%d, sliceHeight=%d", 
         colorFormat, ImageUtils::colorFormatToString(colorFormat), actualStride, actualSliceHeight);
    VIDEO_LOGV(TAG,"detectYUVFormatFromMediaCodec: dimensions=%dx%d, output=%dx%d", 
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
            VIDEO_LOGV(TAG,"detectYUVFormatFromMediaCodec: detected NV12, ySize=%d, uvSize=%d, totalSize=%d", 
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
            VIDEO_LOGV(TAG,"detectYUVFormatFromMediaCodec: detected I420, ySize=%d, uvSize=%d, totalSize=%d", 
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
            VIDEO_LOGV(TAG,"detectYUVFormatFromMediaCodec: unknown format 0x%x, assuming NV12", colorFormat);
            break;
        }
    }
    
    return info;
}

ImageUtils::YUVFormatInfo ImageUtils::detectYUVFormatWithFallback(
    const uint8_t* yuvData, int width, int height, int dataSize,
    int32_t colorFormat, int32_t stride, int32_t sliceHeight) {
    YUVFormatInfo info;
    if (colorFormat != 0) {
        info = detectYUVFormatFromMediaCodec(colorFormat, stride, sliceHeight, width, height);
        if (info.isValid) {
            VIDEO_LOGV(TAG,"detectYUVFormatWithFallback: using MediaCodec format info: %s, stride=%d, sliceHeight=%d", 
                 info.format == YUVFormat::NV12 ? "NV12" :
                 info.format == YUVFormat::I420 ? "I420" :
                 info.format == YUVFormat::NV21 ? "NV21" : "UNKNOWN",
                 info.stride, info.sliceHeight);
            return info;
        }
    }
    
    VIDEO_LOGV(TAG,"detectYUVFormatWithFallback: MediaCodec format not available, using data detection");
    info = detectYUVFormat(yuvData, width, height, dataSize);
    
    if (!info.isValid) {
        VIDEO_LOGV(TAG,"detectYUVFormatWithFallback: data detection failed, assuming NV12");
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
