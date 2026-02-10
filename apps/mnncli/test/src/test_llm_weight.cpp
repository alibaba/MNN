#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include "hf_file_metadata_utils.hpp"
#include "model_file_downloader.hpp"
#include "hf_sha_verifier.hpp"

namespace fs = std::filesystem;
using namespace mnncli;

int main() {
    std::cout << "Testing download and verification with llm.mnn.weight file..." << std::endl;

    // Use the large weight file
    std::string test_url = "https://huggingface.co/taobao-mnn/SmolLM2-135M-Instruct-MNN/resolve/main/llm.mnn.weight";
    std::string cache_path = "/tmp/test_llm_weight";

    try {
        // Create cache directory
        fs::create_directories(cache_path);
        fs::path test_file_path = fs::path(cache_path) / "llm.mnn.weight";

        // Clean up any existing test file
        if (fs::exists(test_file_path)) {
            fs::remove(test_file_path);
        }

        std::cout << "Getting metadata for: " << test_url << std::endl;

        // Get metadata
        std::string error_info;
        HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(test_url, error_info);

        if (!error_info.empty()) {
            std::cerr << "Error getting metadata: " << error_info << std::endl;
            return 1;
        }

        std::cout << "Metadata retrieved successfully" << std::endl;
        std::cout << "  Location: " << metadata.location << std::endl;
        std::cout << "  ETag: " << metadata.etag << std::endl;
        std::cout << "  ETag length: " << metadata.etag.length() << std::endl;
        std::cout << "  Size: " << metadata.size << " bytes" << std::endl;

        // Create download task
        FileDownloadTask task;
        task.etag = metadata.etag;
        task.relativePath = "llm.mnn.weight";
        task.fileMetadata = metadata;
        task.downloadPath = test_file_path;
        task.downloadedSize = 0;

        // Create a simple download listener
        class SimpleDownloadListener : public FileDownloadListener {
        public:
            bool onDownloadDelta(const std::string* fileName, int64_t downloadedBytes,
                                int64_t totalBytes, int64_t delta) override {
                if (totalBytes > 0) {
                    float progress = static_cast<float>(downloadedBytes) / totalBytes * 100;
                    if (downloadedBytes % (1024 * 1024) == 0 || downloadedBytes == totalBytes) {
                        std::cout << "  Progress: " << static_cast<int>(progress) << "% ("
                                  << downloadedBytes / (1024 * 1024) << "MB / " 
                                  << totalBytes / (1024 * 1024) << "MB)" << std::endl;
                    }
                }
                return false; // Don't pause
            }
        };

        SimpleDownloadListener listener;

        // Download the file
        std::cout << "Starting download..." << std::endl;
        ModelFileDownloader downloader;
        downloader.DownloadFile(task, listener);

        std::cout << "Download completed successfully!" << std::endl;

        // Verify the downloaded file exists
        if (!fs::exists(test_file_path)) {
            std::cerr << "Downloaded file does not exist!" << std::endl;
            return 1;
        }

        // Get actual file size
        int64_t actual_size = fs::file_size(test_file_path);
        std::cout << "Actual file size: " << actual_size << " bytes" << std::endl;

        if (actual_size != metadata.size) {
            std::cerr << "File size mismatch! Expected: " << metadata.size
                      << ", Actual: " << actual_size << std::endl;
            return 1;
        }

        // Verify the file using HfShaVerifier
        std::cout << "Verifying file..." << std::endl;
        bool verify_result = HfShaVerifier::verify(metadata.etag, test_file_path);

        if (verify_result) {
            std::cout << "✅ Verification PASSED!" << std::endl;
        } else {
            std::cout << "❌ Verification FAILED!" << std::endl;

            // Let's manually check what hashes we get
            std::string actual_hash;
            if (metadata.etag.length() == 40) {
                actual_hash = HfShaVerifier::gitSha1Hex(test_file_path);
                std::cout << "  Expected (ETag): " << metadata.etag << std::endl;
                std::cout << "  Actual (Git SHA-1): " << actual_hash << std::endl;
            } else if (metadata.etag.length() == 64) {
                actual_hash = HfShaVerifier::sha256Hex(test_file_path);
                std::cout << "  Expected (ETag): " << metadata.etag << std::endl;
                std::cout << "  Actual (SHA-256): " << actual_hash << std::endl;
            }
        }

        // Clean up
        fs::remove(test_file_path);

        return verify_result ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
