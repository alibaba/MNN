#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

int main() {
    // Create test file exactly like our debug test
    fs::path test_file = "/tmp/test_sha_file.txt";
    std::ofstream file(test_file);
    file << "Hello, World!";
    file.close();

    // Read file content to verify
    std::ifstream t(test_file);
    std::string content((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
    t.close();

    std::cout << "File content: '" << content << "'" << std::endl;
    std::cout << "File size: " << content.length() << " bytes" << std::endl;

    // Manual Git SHA-1 calculation
    std::string prefix = "blob " + std::to_string(content.length());
    std::cout << "Prefix: 'blob " << content.length() << "\\0'" << std::endl;

    // Calculate SHA-1 using OpenSSL EVP API
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    const EVP_MD* md = EVP_sha1();
    unsigned char hash[SHA_DIGEST_LENGTH];
    unsigned int hash_len;

    EVP_DigestInit_ex(mdctx, md, NULL);
    EVP_DigestUpdate(mdctx, prefix.c_str(), prefix.length());
    EVP_DigestUpdate(mdctx, "\0", 1);  // Add null terminator
    EVP_DigestUpdate(mdctx, content.c_str(), content.length());
    EVP_DigestFinal_ex(mdctx, hash, &hash_len);
    EVP_MD_CTX_free(mdctx);

    // Convert to hex
    std::stringstream ss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setfill('0') << std::setw(2) << (int)hash[i];
    }

    std::cout << "Manual Git SHA-1: " << ss.str() << std::endl;

    // Clean up
    fs::remove(test_file);

    return 0;
}