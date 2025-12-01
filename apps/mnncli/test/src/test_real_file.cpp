#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "hf_sha_verifier.hpp"

namespace fs = std::filesystem;
using namespace mnncli;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <file_path> <expected_etag>" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];
    std::string expected_etag = argv[2];

    fs::path test_file(file_path);

    if (!fs::exists(test_file)) {
        std::cout << "File does not exist: " << file_path << std::endl;
        return 1;
    }

    std::cout << "Testing SHA calculation with real file: " << file_path << std::endl;
    
    // Get file size
    auto file_size = fs::file_size(test_file);
    std::cout << "File size: " << file_size << " bytes" << std::endl;

    // Test Git SHA-1 calculation
    std::string git_sha1_result = HfShaVerifier::gitSha1Hex(test_file);
    std::cout << "Git SHA-1: " << git_sha1_result << std::endl;
    std::cout << "Expected ETag: " << expected_etag << std::endl;
    std::cout << "Match: " << (git_sha1_result == expected_etag ? "true" : "false") << std::endl;

    // Test verification
    bool verify_result = HfShaVerifier::verify(expected_etag, test_file);
    std::cout << "Verification result: " << (verify_result ? "PASS" : "FAIL") << std::endl;

    return 0;
}
