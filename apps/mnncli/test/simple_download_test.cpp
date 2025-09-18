#include <iostream>
#include <fstream>
#include <string>
#include <httplib.h>

int main() {
    std::cout << "Testing ModelScope file download..." << std::endl;
    
    // Test the exact URL we're using
    std::string url = "https://modelscope.cn/api/v1/models/MNN/SmolLM2-135M-Instruct-MNN/repo?FilePath=.gitattributes";
    
    std::cout << "URL: " << url << std::endl;
    
    // Parse URL to get host and path
    size_t protocol_end = url.find("://");
    if (protocol_end == std::string::npos) {
        std::cerr << "Invalid URL format" << std::endl;
        return 1;
    }
    
    size_t host_start = protocol_end + 3;
    size_t path_start = url.find('/', host_start);
    if (path_start == std::string::npos) {
        std::cerr << "Invalid URL format" << std::endl;
        return 1;
    }
    
    std::string host = url.substr(host_start, path_start - host_start);
    std::string path = url.substr(path_start);
    
    std::cout << "Host: " << host << std::endl;
    std::cout << "Path: " << path << std::endl;
    
    // Create HTTP client
    httplib::Client client("https://" + host);
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI/1.0");
    headers.emplace("Accept", "*/*");
    
    std::cout << "Making request..." << std::endl;
    
    // Test with a simple GET request first
    auto res = client.Get(path, headers);
    
    if (!res) {
        std::cerr << "No response received" << std::endl;
        return 1;
    }
    
    std::cout << "Response status: " << res->status << std::endl;
    std::cout << "Response headers:" << std::endl;
    for (const auto& header : res->headers) {
        std::cout << "  " << header.first << ": " << header.second << std::endl;
    }
    
    std::cout << "Response body length: " << res->body.length() << std::endl;
    std::cout << "First 200 chars: " << res->body.substr(0, 200) << std::endl;
    
    return 0;
}
