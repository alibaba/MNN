#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "hf_sha_verifier.hpp"

namespace fs = std::filesystem;
using namespace mnncli;

int main() {
    // Test with a small file to verify SHA calculation
    fs::path test_file = "/tmp/test_sha_file.txt";

    // Create a test file
    std::ofstream file(test_file);
    file << "Hello, World!";
    file.close();

    std::cout << "Testing SHA calculation with simple file..." << std::endl;
    std::cout << "File content: Hello, World!" << std::endl;

    // Test SHA-256 calculation
    std::string sha256_result = HfShaVerifier::sha256Hex(test_file);
    std::cout << "SHA-256: " << sha256_result << std::endl;

    // Test Git SHA-1 calculation
    std::string git_sha1_result = HfShaVerifier::gitSha1Hex(test_file);
    std::cout << "Git SHA-1: " << git_sha1_result << std::endl;

    // Clean up
    fs::remove(test_file);

    return 0;
}