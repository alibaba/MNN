//
// Standalone test for ModelScope API
// Compile with: g++ -std=c++17 -I../include -I../../../3rd_party/rapidjson/include -I../../../3rd_party/httplib -lssl -lcrypto -o standalone_test standalone_test.cpp
//

#include <iostream>
#include <string>
#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

int main() {
    std::cout << "🔍 Standalone ModelScope API Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Test the same API call that MsApiClient makes
    std::string host = "modelscope.cn";
    std::string path = "/api/v1/models/MNN/SmolLM2-135M-Instruct-MNN";
    
    std::cout << "Making request to: https://" << host << path << std::endl;
    
    // Make the HTTPS request to ModelScope
    httplib::SSLClient cli(host, 443);
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI-Test/1.0");
    headers.emplace("Accept", "application/json");
    
    std::cout << "Sending request..." << std::endl;
    auto res = cli.Get(path, headers);
    
    if (!res) {
        std::cerr << "❌ No response received" << std::endl;
        return 1;
    }
    
    std::cout << "📡 Response status: " << res->status << std::endl;
    std::cout << "📄 Response headers:" << std::endl;
    for (const auto& header : res->headers) {
        std::cout << "   " << header.first << ": " << header.second << std::endl;
    }
    
    std::cout << "\n📝 Raw response body (first 1000 chars):" << std::endl;
    std::string body = res->body;
    if (body.length() > 1000) {
        body = body.substr(0, 1000) + "...";
    }
    std::cout << body << std::endl;
    
    if (res->status != 200) {
        std::cerr << "❌ API request failed with status " << res->status << std::endl;
        return 1;
    }
    
    // Try to parse the JSON response
    rapidjson::Document doc;
    if (doc.Parse(res->body.c_str()).HasParseError()) {
        std::cerr << "❌ Failed to parse JSON response: " 
                  << rapidjson::GetParseError_En(doc.GetParseError()) << std::endl;
        return 1;
    }
    
    std::cout << "\n✅ JSON parsed successfully" << std::endl;
    
    // Check the structure
    std::cout << "\n🔍 JSON structure analysis:" << std::endl;
    for (rapidjson::Value::ConstMemberIterator it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) {
        std::cout << "   " << it->name.GetString() << ": ";
        if (it->value.IsString()) {
            std::cout << "string = \"" << it->value.GetString() << "\"";
        } else if (it->value.IsInt()) {
            std::cout << "int = " << it->value.GetInt();
        } else if (it->value.IsInt64()) {
            std::cout << "int64 = " << it->value.GetInt64();
        } else if (it->value.IsArray()) {
            std::cout << "array with " << it->value.Size() << " elements";
        } else if (it->value.IsObject()) {
            std::cout << "object with " << it->value.MemberCount() << " members";
        } else {
            std::cout << "other type";
        }
        std::cout << std::endl;
    }
    
    // Check for Data field
    if (doc.HasMember("Data")) {
        std::cout << "\n📊 Data field found:" << std::endl;
        const rapidjson::Value& data = doc["Data"];
        
        if (data.IsObject()) {
            for (rapidjson::Value::ConstMemberIterator it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
                std::cout << "   " << it->name.GetString() << ": ";
                if (it->value.IsString()) {
                    std::cout << "string = \"" << it->value.GetString() << "\"";
                } else if (it->value.IsInt()) {
                    std::cout << "int = " << it->value.GetInt();
                } else if (it->value.IsInt64()) {
                    std::cout << "int64 = " << it->value.GetInt64();
                } else if (it->value.IsArray()) {
                    std::cout << "array with " << it->value.Size() << " elements";
                } else if (it->value.IsObject()) {
                    std::cout << "object with " << it->value.MemberCount() << " members";
                } else {
                    std::cout << "other type";
                }
                std::cout << std::endl;
            }
            
            // Check for Files array
            if (data.HasMember("Files") && data["Files"].IsArray()) {
                const rapidjson::Value& files = data["Files"];
                std::cout << "\n📁 Files array found with " << files.Size() << " elements:" << std::endl;
                
                for (rapidjson::Value::SizeType i = 0; i < files.Size(); i++) {
                    const rapidjson::Value& file = files[i];
                    if (file.IsObject()) {
                        std::cout << "   File " << i << ":" << std::endl;
                        for (rapidjson::Value::ConstMemberIterator it = file.MemberBegin(); it != file.MemberEnd(); ++it) {
                            std::cout << "     " << it->name.GetString() << ": ";
                            if (it->value.IsString()) {
                                std::cout << "\"" << it->value.GetString() << "\"";
                            } else if (it->value.IsInt64()) {
                                std::cout << it->value.GetInt64();
                            } else {
                                std::cout << "other type";
                            }
                            std::cout << std::endl;
                        }
                    }
                }
            } else {
                std::cout << "\n❌ Files array not found or not an array" << std::endl;
            }
        } else {
            std::cout << "\n❌ Data field is not an object" << std::endl;
        }
    } else {
        std::cout << "\n❌ Data field not found" << std::endl;
    }
    
    // Test alternative API endpoints
    std::cout << "\n🔍 Testing alternative API endpoints..." << std::endl;
    
    // Try the repo endpoint
    std::string repo_path = "/api/v1/models/MNN/SmolLM2-135M-Instruct-MNN/repo";
    std::cout << "Testing repo endpoint: " << repo_path << std::endl;
    
    auto repo_res = cli.Get(repo_path, headers);
    if (repo_res && repo_res->status == 200) {
        std::cout << "✅ Repo endpoint works!" << std::endl;
        std::cout << "Response length: " << repo_res->body.length() << " characters" << std::endl;
        
        // Show first 500 chars of repo response
        std::string repo_body = repo_res->body;
        if (repo_body.length() > 500) {
            repo_body = repo_body.substr(0, 500) + "...";
        }
        std::cout << "First 500 chars: " << repo_body << std::endl;
    } else {
        std::cout << "❌ Repo endpoint failed with status: " 
                  << (repo_res ? std::to_string(repo_res->status) : "no response") << std::endl;
    }
    
    // Try a different model to see if the issue is specific to this one
    std::cout << "\n🔍 Testing with a different model..." << std::endl;
    std::string alt_path = "/api/v1/models/damo/nlp_gpt2_text-generation_1.3B";
    std::cout << "Testing alternative model: " << alt_path << std::endl;
    
    auto alt_res = cli.Get(alt_path, headers);
    if (alt_res && alt_res->status == 200) {
        std::cout << "✅ Alternative model works!" << std::endl;
        std::cout << "Response length: " << alt_res->body.length() << " characters" << std::endl;
        
        // Parse this response too
        rapidjson::Document alt_doc;
        if (!alt_doc.Parse(alt_res->body.c_str()).HasParseError()) {
            if (alt_doc.HasMember("Data") && alt_doc["Data"].IsObject()) {
                const rapidjson::Value& alt_data = alt_doc["Data"];
                if (alt_data.HasMember("Files") && alt_data["Files"].IsArray()) {
                    const rapidjson::Value& alt_files = alt_data["Files"];
                    std::cout << "Alternative model has " << alt_files.Size() << " files" << std::endl;
                }
            }
        }
    } else {
        std::cout << "❌ Alternative model failed with status: " 
                  << (alt_res ? std::to_string(alt_res->status) : "no response") << std::endl;
    }
    
    return 0;
}
