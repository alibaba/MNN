#include <iostream>
#include <string>
#include "hf_file_metadata_utils.hpp"
#include "hf_sha_verifier.hpp"
#include "log_utils.hpp"

using namespace mnncli;

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "CDN ETag Comparison Test" << std::endl;
    std::cout << "==========================================" << std::endl;

    mnncli::LogUtils::SetVerbose(true);

    try {
        // Test config.json (known to work)
        std::cout << "\n--- Testing config.json ---" << std::endl;
        std::string config_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/config.json";
        std::string config_error;
        HfFileMetadata config_metadata = HfFileMetadataUtils::GetFileMetadata(config_url, config_error);

        if (config_error.empty()) {
            std::cout << "Config.json ETag: " << config_metadata.etag << std::endl;
            std::cout << "Config.json Location: " << config_metadata.location << std::endl;
            std::cout << "Config.json Size: " << config_metadata.size << std::endl;

            bool is_config_cdn = (config_metadata.location.find("cdn-lfs") != std::string::npos) ||
                                (config_metadata.location.find("cdn.") != std::string::npos) ||
                                (config_metadata.location.find("cas-bridge") != std::string::npos);
            std::cout << "Config.json served from CDN: " << (is_config_cdn ? "YES" : "NO") << std::endl;
        } else {
            std::cout << "Config.json metadata error: " << config_error << std::endl;
        }

        // Test embeddings_bf16.bin (failing)
        std::cout << "\n--- Testing embeddings_bf16.bin ---" << std::endl;
        std::string embedding_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/embeddings_bf16.bin";
        std::string embedding_error;
        HfFileMetadata embedding_metadata = HfFileMetadataUtils::GetFileMetadata(embedding_url, embedding_error);

        if (embedding_error.empty()) {
            std::cout << "Embedding ETag: " << embedding_metadata.etag << std::endl;
            std::cout << "Embedding Location: " << embedding_metadata.location << std::endl;
            std::cout << "Embedding Size: " << embedding_metadata.size << std::endl;

            bool is_embedding_cdn = (embedding_metadata.location.find("cdn-lfs") != std::string::npos) ||
                                   (embedding_metadata.location.find("cdn.") != std::string::npos) ||
                                   (embedding_metadata.location.find("cas-bridge") != std::string::npos);
            std::cout << "Embedding served from CDN: " << (is_embedding_cdn ? "YES" : "NO") << std::endl;
        } else {
            std::cout << "Embedding metadata error: " << embedding_error << std::endl;
        }

        std::cout << "\n==========================================" << std::endl;
        std::cout << "Test completed!" << std::endl;
        std::cout << "==========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}