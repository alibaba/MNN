#ifdef __ANDROID__

#include "AndroidVideoDecoder.hpp"
#include "log_utils.hpp"
#include <media/NdkMediaFormat.h>
#include <media/NdkImage.h>
#include <libyuv.h>
#include <unistd.h>     // For access() function, open(), close()
#include <sys/stat.h>   // For stat() function
#include <errno.h>      // For errno
#include <fcntl.h>      // For O_RDONLY
#include <chrono>       // For std::chrono
#include <thread>       // For std::this_thread

namespace MNN {

// Utility function to convert AImage to cv::Mat using libyuv
static cv::Mat aimage_to_cvmat(AImage* image) {
    if (!image) {
        return cv::Mat();
    }

    int32_t width, height;
    AImage_getWidth(image, &width);
    AImage_getHeight(image, &height);

    // Get Y plane data
    uint8_t* y_data = nullptr;
    int32_t y_stride = 0;
    AImage_getPlaneData(image, 0, &y_data, &y_stride);

    // Get U plane data
    uint8_t* u_data = nullptr;
    int32_t u_stride = 0;
    int32_t u_pixel_stride = 0;
    AImage_getPlaneData(image, 1, &u_data, &u_stride);
    AImage_getPlanePixelStride(image, 1, &u_pixel_stride);

    // Get V plane data
    uint8_t* v_data = nullptr;
    int32_t v_stride = 0;
    int32_t v_pixel_stride = 0;
    AImage_getPlaneData(image, 2, &v_data, &v_stride);
    AImage_getPlanePixelStride(image, 2, &v_pixel_stride);

    cv::Mat abgr_mat(height, width, CV_8UC4);

    // Use Android420ToABGR which handles various YUV_420_888 layouts
    // This works for both planar and semi-planar formats based on pixel stride
    libyuv::Android420ToABGR(
        y_data, y_stride,
        u_data, u_stride,
        v_data, v_stride,
        u_pixel_stride,  // src_pixel_stride_uv - use U pixel stride for both U and V
        abgr_mat.data, abgr_mat.step,
        width, height);

    return abgr_mat;
}

// Convert direct codec buffer to cv::Mat
static cv::Mat convertDirectBufferToMat(uint8_t* buffer, size_t buffer_size, int width, int height, int color_format) {
    if (!buffer || buffer_size == 0 || width <= 0 || height <= 0) {
        LOG_ERROR("Invalid buffer parameters for direct conversion");
        return cv::Mat();
    }
    
    LOG_DEBUG("Converting direct buffer: %dx%d, format: %d, size: %zu", width, height, color_format, buffer_size);
    
    // Common Android color formats
    // COLOR_FormatYUV420Planar = 19
    // COLOR_FormatYUV420SemiPlanar = 21  
    // COLOR_FormatYUV420PackedPlanar = 20
    
    cv::Mat result;
    
    // Calculate expected YUV420 size
    size_t expected_yuv420_size = width * height * 3 / 2;
    
    if (buffer_size >= expected_yuv420_size) {
        // Assume YUV420 format and convert to BGR
        cv::Mat yuv_mat(height * 3 / 2, width, CV_8UC1, buffer);
        cv::Mat bgr_mat;
        
        try {
            cv::cvtColor(yuv_mat, bgr_mat, cv::COLOR_YUV2BGR_I420);
            LOG_DEBUG("Direct buffer YUV420 conversion successful");
            return bgr_mat;
        } catch (const cv::Exception& e) {
            LOG_DEBUG("YUV420 conversion failed: %s", e.what());
        }
        
        // Try NV21 format
        try {
            cv::cvtColor(yuv_mat, bgr_mat, cv::COLOR_YUV2BGR_NV21);
            LOG_DEBUG("Direct buffer NV21 conversion successful");
            return bgr_mat;
        } catch (const cv::Exception& e) {
            LOG_DEBUG("NV21 conversion failed: %s", e.what());
        }
    }
    
    LOG_DEBUG("Direct buffer conversion failed for format %d", color_format);
    return cv::Mat();
}

AndroidVideoDecoder::AndroidVideoDecoder() : file_descriptor_(-1) {}

AndroidVideoDecoder::~AndroidVideoDecoder() {
    release();
}

// Static callback function for ImageReader
void AndroidVideoDecoder::on_image_available(void* context, AImageReader* reader) {
    AndroidVideoDecoder* decoder = static_cast<AndroidVideoDecoder*>(context);
    if (!decoder) {
        LOG_ERROR("Invalid decoder context in image callback");
        return;
    }

    LOG_DEBUG("ImageReader callback triggered");

    // Drain all available images from the ImageReader to prevent backpressure
    int processed_count = 0;
    int valid_frames = 0;
    while (true) {
        AImage* image = nullptr;
        media_status_t status = AImageReader_acquireNextImage(reader, &image);
        
        if (status != AMEDIA_OK || !image) {
            // No more images available
            if (status != AMEDIA_OK) {
                LOG_DEBUG("AImageReader_acquireNextImage failed with status: %d", status);
            }
            break;
        }

        // Convert AImage to cv::Mat
        cv::Mat frame = aimage_to_cvmat(image);
        AImage_delete(image);
        processed_count++;

        if (!frame.empty()) {
            valid_frames++;
            // Add frame to queue in thread-safe manner
            std::lock_guard<std::mutex> lock(decoder->frame_queue_mutex_);
            decoder->frame_queue_.push(frame);
            
            // Limit queue size to prevent memory issues
            const size_t MAX_QUEUE_SIZE = 10;
            while (decoder->frame_queue_.size() > MAX_QUEUE_SIZE) {
                decoder->frame_queue_.pop();
                LOG_DEBUG("Dropped old frame to maintain queue size");
            }
            
            // Notify waiting threads
            decoder->frame_available_cv_.notify_one();
            
            LOG_DEBUG("Frame added to queue, current queue size: %zu", 
                      decoder->frame_queue_.size());
        } else {
            LOG_DEBUG("Empty frame from AImage conversion");
        }
    }
    
    LOG_DEBUG("Callback processed %d images, %d valid frames", processed_count, valid_frames);
}

void AndroidVideoDecoder::release() {
    if (codec_) {
        AMediaCodec_stop(codec_);
        AMediaCodec_delete(codec_);
        codec_ = nullptr;
    }
    if (image_reader_) {
        AImageReader_delete(image_reader_);
        image_reader_ = nullptr;
    }
    native_window_ = nullptr; // Owned by ImageReader
    if (extractor_) {
        AMediaExtractor_delete(extractor_);
        extractor_ = nullptr;
    }
    if (file_descriptor_ >= 0) {
        close(file_descriptor_);
        file_descriptor_ = -1;
    }
    
    // Clear frame queue
    {
        std::lock_guard<std::mutex> lock(frame_queue_mutex_);
        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }
    }
    frame_available_cv_.notify_all();
    
    is_eos_ = false;
}

bool AndroidVideoDecoder::init(const std::string& video_path) {
    // Check if file exists and is accessible
    if (access(video_path.c_str(), F_OK) != 0) {
        LOG_ERROR("Video file does not exist or is not accessible: %s (errno: %d)", 
                  video_path.c_str(), errno);
        return false;
    }
    
    // Check file permissions
    if (access(video_path.c_str(), R_OK) != 0) {
        LOG_ERROR("Video file is not readable: %s (errno: %d)", 
                  video_path.c_str(), errno);
        return false;
    }
    
    // Get file info for debugging
    struct stat file_stat;
    if (stat(video_path.c_str(), &file_stat) == 0) {
        LOG_DEBUG("Video file info - Size: %ld bytes, Permissions: %o", 
                  file_stat.st_size, file_stat.st_mode & 0777);
    }
    
    // Open file and get file descriptor
    file_descriptor_ = open(video_path.c_str(), O_RDONLY);
    if (file_descriptor_ < 0) {
        LOG_ERROR("Failed to open video file: %s (errno: %d)", video_path.c_str(), errno);
        return false;
    }
    
    extractor_ = AMediaExtractor_new();
    if (!extractor_) {
        LOG_ERROR("Failed to create AMediaExtractor");
        close(file_descriptor_);
        file_descriptor_ = -1;
        return false;
    }
    
    // Use file descriptor instead of file path
    if (AMediaExtractor_setDataSourceFd(extractor_, file_descriptor_, 0, file_stat.st_size) != AMEDIA_OK) {
        LOG_ERROR("Failed to set data source for video: %s", video_path.c_str());
        LOG_ERROR("This could be due to: unsupported format, corrupted file, or Android API compatibility issue");
        close(file_descriptor_);
        file_descriptor_ = -1;
        return false;
    }

    AMediaFormat* track_format = nullptr;
    for (int i = 0; i < AMediaExtractor_getTrackCount(extractor_); ++i) {
        AMediaFormat* format = AMediaExtractor_getTrackFormat(extractor_, i);
        const char* mime = nullptr;
        AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mime);
        if (strncmp(mime, "video/", 6) == 0) {
            AMediaExtractor_selectTrack(extractor_, i);
            track_format = format;
            LOG_DEBUG("Selected video track %d with mime type %s", i, mime);
            break;
        }
        AMediaFormat_delete(format);
    }

    if (!track_format) {
        LOG_ERROR("No video track found in %s", video_path.c_str());
        return false;
    }

    AMediaFormat_getInt32(track_format, AMEDIAFORMAT_KEY_WIDTH, &width_);
    AMediaFormat_getInt32(track_format, AMEDIAFORMAT_KEY_HEIGHT, &height_);
    
    // Get frame rate with proper error checking
    int32_t frame_rate_int = 0;
    if (AMediaFormat_getInt32(track_format, AMEDIAFORMAT_KEY_FRAME_RATE, &frame_rate_int)) {
        fps_ = static_cast<double>(frame_rate_int);
    } else {
        // Try alternative frame rate key
        if (AMediaFormat_getInt32(track_format, AMEDIAFORMAT_KEY_FRAME_RATE, &frame_rate_int)) {
            fps_ = static_cast<double>(frame_rate_int);
        } else {
            // Default to 30 fps if we can't get the actual frame rate
            fps_ = 30.0;
            LOG_DEBUG("Could not get frame rate from media format, defaulting to 30 fps");
        }
    }
    
    int64_t duration_us = 0;
    if (AMediaFormat_getInt64(track_format, AMEDIAFORMAT_KEY_DURATION, &duration_us)) {
        if (fps_ > 0) {
            frame_count_ = static_cast<int>((duration_us / 1000000.0) * fps_);
        }
    } else {
        LOG_DEBUG("Could not get duration from media format");
        frame_count_ = 0;
    }
    
    LOG_DEBUG("Video properties: %dx%d, %.2f fps, %d frames (estimated)", width_, height_, fps_, frame_count_);

    if (AImageReader_new(width_, height_, AIMAGE_FORMAT_YUV_420_888, 4, &image_reader_) != AMEDIA_OK) {
        LOG_ERROR("Failed to create AImageReader");
        AMediaFormat_delete(track_format);
        return false;
    }
    
    AImageReader_getWindow(image_reader_, &native_window_);
    LOG_DEBUG("Created AImageReader: %dx%d, native window: %p", width_, height_, native_window_);
    
    const char* mime = nullptr;
    AMediaFormat_getString(track_format, AMEDIAFORMAT_KEY_MIME, &mime);
    LOG_DEBUG("Creating decoder for mime type: %s", mime);
    
    codec_ = AMediaCodec_createDecoderByType(mime);
    if (!codec_) {
        LOG_ERROR("Failed to create codec for mime type %s", mime);
        AMediaFormat_delete(track_format);
        return false;
    }
    LOG_DEBUG("Codec created successfully");

    LOG_DEBUG("Configuring codec with format...");
    // Try with Surface first, but fallback to direct buffer mode if it fails
    media_status_t config_result = AMediaCodec_configure(codec_, track_format, native_window_, nullptr, 0);
    if (config_result != AMEDIA_OK) {
        LOG_DEBUG("Surface configuration failed (%d), trying direct buffer mode", config_result);
        
        // Clean up the ImageReader since we won't use Surface mode
        if (image_reader_) {
            AImageReader_delete(image_reader_);
            image_reader_ = nullptr;
        }
        native_window_ = nullptr;
        
        // Configure without Surface for direct buffer access
        config_result = AMediaCodec_configure(codec_, track_format, nullptr, nullptr, 0);
        if (config_result != AMEDIA_OK) {
            LOG_ERROR("Both Surface and direct buffer configuration failed");
            AMediaFormat_delete(track_format);
            return false;
        }
        LOG_DEBUG("Codec configured successfully in direct buffer mode");
    } else {
        LOG_DEBUG("Codec configured successfully with Surface");
    }

    LOG_DEBUG("Starting codec...");
    if (AMediaCodec_start(codec_) != AMEDIA_OK) {
        LOG_ERROR("Failed to start codec");
        AMediaFormat_delete(track_format);
        return false;
    }
    LOG_DEBUG("Codec started successfully");

    // Set up ImageReader callback only if we're in Surface mode
    if (image_reader_ && native_window_) {
        AImageReader_ImageListener listener{
            .context = this,
            .onImageAvailable = AndroidVideoDecoder::on_image_available
        };
        
        if (AImageReader_setImageListener(image_reader_, &listener) != AMEDIA_OK) {
            LOG_ERROR("Failed to set ImageReader listener");
            AMediaFormat_delete(track_format);
            return false;
        }
        LOG_DEBUG("ImageReader listener set successfully after codec start");
    } else {
        LOG_DEBUG("Using direct buffer mode, no ImageReader callback needed");
    }

    AMediaFormat_delete(track_format);
    return true;
}

cv::Mat AndroidVideoDecoder::decode_one_frame() {
    if (is_eos_) {
        // Check if there are any remaining frames in the queue
        std::lock_guard<std::mutex> lock(frame_queue_mutex_);
        if (!frame_queue_.empty()) {
            cv::Mat frame = frame_queue_.front();
            frame_queue_.pop();
            LOG_DEBUG("Returned queued frame after EOS, remaining: %zu", frame_queue_.size());
            return frame;
        }
        return cv::Mat();
    }

    // Try to feed multiple input buffers to the decoder to keep pipeline full
    int input_fed = 0;
    for (int i = 0; i < 3; i++) { // Try up to 3 input buffers
        ssize_t in_idx = AMediaCodec_dequeueInputBuffer(codec_, 0); // Non-blocking
        if (in_idx >= 0) {
            size_t buf_size;
            uint8_t* buf = AMediaCodec_getInputBuffer(codec_, in_idx, &buf_size);
            if (!buf) {
                LOG_ERROR("Failed to get input buffer %d", i);
                break;
            }
            
            ssize_t sample_size = AMediaExtractor_readSampleData(extractor_, buf, buf_size);
            if (sample_size < 0) {
                is_eos_ = true;
                sample_size = 0;
                LOG_DEBUG("End of stream reached at input %d", i);
            }
            
            int64_t sample_time = AMediaExtractor_getSampleTime(extractor_);
            uint32_t flags = is_eos_ ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0;
            
            if (AMediaCodec_queueInputBuffer(codec_, in_idx, 0, sample_size, sample_time, flags) != AMEDIA_OK) {
                LOG_ERROR("Failed to queue input buffer %d", i);
                break;
            }
            
            if (!is_eos_) {
                AMediaExtractor_advance(extractor_);
            }
            
            input_fed++;
            LOG_DEBUG("Queued input buffer %d with %zd bytes", i, sample_size);
            
            if (is_eos_) {
                break; // Stop feeding after EOS
            }
        } else if (in_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            // No more input buffers available
            if (i == 0) {
                LOG_DEBUG("No input buffer available yet");
            }
            break;
        } else {
            LOG_ERROR("Unexpected input buffer index: %zd", in_idx);
            return cv::Mat();
        }
    }
    
    if (input_fed > 0) {
        LOG_DEBUG("Fed %d input buffers to codec", input_fed);
    }

    // Drain all available output buffers to prevent backpressure
    int output_count = 0;
    while (true) {
        AMediaCodecBufferInfo info;
        ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(codec_, &info, 10000); // 10ms timeout for first output
        
        if (out_idx >= 0) {
            // Check if we're in direct buffer mode (no Surface)
            if (!native_window_) {
                // Direct buffer mode - get the actual decoded data
                size_t out_size;
                uint8_t* buf = AMediaCodec_getOutputBuffer(codec_, out_idx, &out_size);
                if (buf && out_size > 0) {
                    LOG_DEBUG("Got direct buffer output %zd, size: %zu", out_idx, out_size);
                    
                    // Get the output format to understand the data layout
                    AMediaFormat* out_format = AMediaCodec_getOutputFormat(codec_);
                    if (out_format) {
                        int32_t color_format;
                        if (AMediaFormat_getInt32(out_format, AMEDIAFORMAT_KEY_COLOR_FORMAT, &color_format)) {
                            LOG_DEBUG("Direct buffer color format: %d", color_format);
                        }
                        
                        int32_t width, height;
                        if (AMediaFormat_getInt32(out_format, AMEDIAFORMAT_KEY_WIDTH, &width) &&
                            AMediaFormat_getInt32(out_format, AMEDIAFORMAT_KEY_HEIGHT, &height)) {
                            LOG_DEBUG("Direct buffer dimensions: %dx%d", width, height);
                            
                            // Try to convert the direct buffer to cv::Mat
                            cv::Mat frame = convertDirectBufferToMat(buf, out_size, width, height, color_format);
                            
                            // Release the buffer
                            AMediaCodec_releaseOutputBuffer(codec_, out_idx, false);
                            AMediaFormat_delete(out_format);
                            
                            if (!frame.empty()) {
                                LOG_DEBUG("Successfully converted direct buffer to Mat: %dx%d", frame.cols, frame.rows);
                                return frame;
                            }
                        }
                        AMediaFormat_delete(out_format);
                    }
                }
                
                // Release buffer even if conversion failed
                AMediaCodec_releaseOutputBuffer(codec_, out_idx, false);
                output_count++;
            } else {
                // Surface mode - release to render to ImageReader
                if (AMediaCodec_releaseOutputBuffer(codec_, out_idx, true) != AMEDIA_OK) {
                    LOG_ERROR("Failed to release output buffer");
                    break;
                }
                output_count++;
                LOG_DEBUG("Released output buffer %zd to Surface", out_idx);
                
                // After releasing, give some time for the surface to process
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        } else if (out_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            // No more output available
            if (output_count == 0) {
                LOG_DEBUG("No output available yet");
            }
            break;
        } else if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            LOG_DEBUG("Output format changed");
            continue;
        } else if (out_idx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
            LOG_DEBUG("Output buffers changed");
            continue;
        } else {
            LOG_ERROR("Unexpected output buffer index: %zd", out_idx);
            break;
        }
    }
    
    if (output_count > 0) {
        LOG_DEBUG("Drained %d output buffers", output_count);
    }

    // Only try Surface/ImageReader mode if we have them
    if (image_reader_ && native_window_) {
        // Try to get a frame from the queue (populated by ImageReader callback)
        {
            std::unique_lock<std::mutex> lock(frame_queue_mutex_);
            if (!frame_queue_.empty()) {
                cv::Mat frame = frame_queue_.front();
                frame_queue_.pop();
                LOG_DEBUG("Retrieved frame from queue, remaining: %zu", frame_queue_.size());
                return frame;
            }
            
            // If no frame available immediately, wait a short time for callback to provide one
            if (frame_available_cv_.wait_for(lock, std::chrono::milliseconds(10), 
                                             [this] { return !frame_queue_.empty(); })) {
                cv::Mat frame = frame_queue_.front();
                frame_queue_.pop();
                LOG_DEBUG("Retrieved frame from queue after wait, remaining: %zu", frame_queue_.size());
                return frame;
            }
        }

        // Fallback: manually check ImageReader if callback didn't work
        LOG_DEBUG("Callback queue empty, trying manual ImageReader check");
        int manual_acquired = 0;
        
        // Try both acquireLatestImage and acquireNextImage
        for (int attempt = 0; attempt < 5; attempt++) {
            AImage* image = nullptr;
            media_status_t status;
            
            // Try latest first, then next
            if (attempt == 0) {
                status = AImageReader_acquireLatestImage(image_reader_, &image);
                LOG_DEBUG("Trying acquireLatestImage, status: %d", status);
            } else {
                status = AImageReader_acquireNextImage(image_reader_, &image);
                LOG_DEBUG("Trying acquireNextImage attempt %d, status: %d", attempt, status);
            }
            
            if (status == AMEDIA_OK && image) {
                manual_acquired++;
                cv::Mat frame = aimage_to_cvmat(image);
                AImage_delete(image);
                
                if (!frame.empty()) {
                    LOG_DEBUG("Manually acquired frame %d on attempt %d", manual_acquired, attempt);
                    return frame;
                }
            } else if (status == AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE) {
                LOG_DEBUG("No buffer available on attempt %d", attempt);
                break;
            } else {
                LOG_DEBUG("Failed to acquire image on attempt %d, status: %d", attempt, status);
                // Wait a bit for more frames
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        if (manual_acquired > 0) {
            LOG_DEBUG("Manually processed %d images but got no valid frames", manual_acquired);
        } else {
            LOG_DEBUG("No images available from ImageReader");
        }
    } else {
        LOG_DEBUG("Direct buffer mode - no ImageReader operations needed");
    }

    // Still no frame available - add small delay to prevent tight loop
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return cv::Mat();
}

} // namespace MNN

#endif // __ANDROID__
