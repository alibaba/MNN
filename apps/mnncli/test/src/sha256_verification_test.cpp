#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "hf_sha_verifier.hpp"

namespace fs = std::filesystem;
using namespace mnncli;

int main() {
    // Create a test file with known content
    fs::path test_file = "/tmp/test_sha256_file.txt";

    // Create test file
    std::ofstream file(test_file);
    file << "Hello, World!";
    file.close();

    std::cout << "Testing SHA-256 calculation with simple file..." << std::endl;
    std::cout << "File content: Hello, World!" << std::endl;

    // Test our HfShaVerifier methods
    std::string git_sha1 = HfShaVerifier::gitSha1Hex(test_file);
    std::string sha256 = HfShaVerifier::sha256Hex(test_file);

    std::cout << "Git SHA-1: " << git_sha1 << std::endl;
    std::cout << "SHA-256: " << sha256 << std::endl;

    // Expected SHA-256 for "Hello, World!" is: dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f
    std::cout << "Expected SHA-256 for 'Hello, World!': dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f" << std::endl;

    // Clean up
    fs::remove(test_file);

    return 0;
}