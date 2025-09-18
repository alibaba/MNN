#pragma once

#ifdef __ANDROID__

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkImageReader.h>
#include <android/native_window.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

// Forward declaration for AImage
struct AImage;
// Forward declaration for AImageReader
struct AImageReader;

namespace MNN {

class AndroidVideoDecoder {
public:
    AndroidVideoDecoder();
    ~AndroidVideoDecoder();

    bool init(const std::string& video_path);
    cv::Mat decode_one_frame();
    void release();

    double get_fps() const { return fps_; }
    int get_frame_count() const { return frame_count_; }
    bool is_eos() const { return is_eos_; }

private:
    AMediaExtractor* extractor_ = nullptr;
    AMediaCodec* codec_ = nullptr;
    ANativeWindow* native_window_ = nullptr;
    AImageReader* image_reader_ = nullptr;
    int file_descriptor_ = -1;  // File descriptor for video file

    double fps_ = 0.0;
    int frame_count_ = 0;
    int width_ = 0;
    int height_ = 0;
    bool is_eos_ = false;

    // Frame queue for thread-safe communication between callback and decode_one_frame
    std::queue<cv::Mat> frame_queue_;
    std::mutex frame_queue_mutex_;
    std::condition_variable frame_available_cv_;
    
    // Static callback function for ImageReader
    static void on_image_available(void* context, AImageReader* reader);
};

} // namespace MNN

#endif // __ANDROID__
