#include <MNN/MNNDefine.h>
#define UJSON_USE_RAPIDJSON
#include "../src/tokenizer/jinja.hpp"
#include "../src/ujson.hpp"
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <fstream>
#include <iostream>
#include <sstream>

static int test(const char* testjson) {
    std::ifstream inputFs(testjson);
    if (inputFs.fail()) {
        MNN_ERROR("Open %s error\n", testjson);
        return 0;
    }
    std::ostringstream osString;
    osString << inputFs.rdbuf();
    auto document = jinja::json::parse(osString.str());
    if (!document.is_array()) {
        MNN_ERROR("Invalid json\n");
        return 0;
    }
    int pos = 0;
    for (size_t i = 0; i < document.size(); i++) {
        auto iter = document[i];
        std::string res = iter["res"].get<std::string>();
        std::string chatTemplate = iter["chat_template"].get<std::string>();
        std::string bos, eos;
        if (iter.contains("bos")) bos = iter["bos"].get<std::string>();
        if (iter.contains("eos")) eos = iter["eos"].get<std::string>();

        jinja::json default_ctx = jinja::json::object();
        default_ctx["bos_token"] = bos;
        default_ctx["eos_token"] = eos;
        jinja::Template tmpl(chatTemplate, default_ctx);

        jinja::json messages = iter["messages"];
        jinja::json extra_ctx = jinja::json::object();
        if (iter.contains("extras")) {
            extra_ctx = iter["extras"];
        }
        auto newres = tmpl.apply_chat_template(messages, true, jinja::json::array(), extra_ctx);
        if (res != newres) {
            MNN_ERROR("Error for %d template\n", pos);
            MNN_ERROR("Origin:\n%s\n", res.c_str());
            MNN_ERROR("Compute:\n%s\n", newres.c_str());
            return 0;
        }
        pos++;
    }
    MNN_PRINT("Test %d template, All Right\n", pos);
    return 0;
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        MNN_ERROR("Usage: ./apply_template token_config.json \n");
        MNN_ERROR("Or \n");
        MNN_ERROR("Usage: ./apply_template test.json 1\n");
        return 0;
    }
    if (argc >= 3) {
        MNN_PRINT("Test %s\n", argv[1]);
        test(argv[1]);
        return 0;
    }
    rapidjson::Document resDocument;
    {
        std::ifstream inputFs("result.json");
        bool valid = false;
        if (!inputFs.fail()) {
            std::ostringstream osString;
            osString << inputFs.rdbuf();
            resDocument.Parse(osString.str().c_str());
            if (resDocument.HasParseError()) {
                MNN_ERROR("Invalid json\n");
            } else {
                valid = true;
                MNN_PRINT("Has result.json, append it\n");
            }
        }
        if (!valid) {
            resDocument.SetArray();
            MNN_PRINT("Create new result.json\n");
        }
    }
    for (int i = 1; i < argc; ++i) {
        auto tokenConfigPath = argv[i];
        FUNC_PRINT_ALL(tokenConfigPath, s);
        rapidjson::Document document;
        std::ifstream inputFs(tokenConfigPath);
        std::ostringstream osString;
        if (inputFs.fail()) {
            MNN_ERROR("Open File error\n");
            return 0;
        }
        osString << inputFs.rdbuf();
        document.Parse(osString.str().c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        std::string bosToken, eosToken;
        auto loadtoken = [](const rapidjson::GenericValue<rapidjson::UTF8<>>& value, std::string& dst) {
            if (value.IsString()) { dst = value.GetString(); return; }
            if (value.IsObject() && value.HasMember("content") && value["content"].IsString()) {
                dst = value["content"].GetString();
            }
        };
        if (document.HasMember("bos_token")) loadtoken(document["bos_token"], bosToken);
        if (document.HasMember("eos_token")) loadtoken(document["eos_token"], eosToken);
        std::string templateChat;
        if (document.HasMember("chat_template")) {
            templateChat = document["chat_template"].GetString();
        }
        if (templateChat.empty()) {
            MNN_ERROR("Invalid json with no chat_template\n");
            return 0;
        }

        jinja::json default_ctx = jinja::json::object();
        default_ctx["bos_token"] = bosToken;
        default_ctx["eos_token"] = eosToken;
        jinja::Template tmpl(templateChat, default_ctx);

        jinja::json messages = jinja::json::parse(R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 8 * 12."},
        {"role": "assistant", "content": "96."},
        {"role": "user", "content": "What is 9 * 8?"}
        ])");

        jinja::json extra_ctx = jinja::json::object();
        extra_ctx["enable_thinking"] = false;

        auto res = tmpl.apply_chat_template(messages, true, jinja::json::array(), extra_ctx);
        MNN_PRINT("%s", res.c_str());

        // Write result
        rapidjson::Value v;
        v.SetObject();
        rapidjson::Document rjMessages;
        rjMessages.Parse(R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 8 * 12."},
        {"role": "assistant", "content": "96."},
        {"role": "user", "content": "What is 9 * 8?"}
        ])");
        rapidjson::Value extras;
        extras.SetObject();
        extras.AddMember("enable_thinking", false, resDocument.GetAllocator());
        rapidjson::Value msgCopy;
        msgCopy.CopyFrom(rjMessages, resDocument.GetAllocator());
        v.AddMember("messages", msgCopy, resDocument.GetAllocator());
        v.AddMember("extras", extras, resDocument.GetAllocator());
        {
            rapidjson::Value tv;
            tv.SetString(templateChat.c_str(), resDocument.GetAllocator());
            v.AddMember("chat_template", tv, resDocument.GetAllocator());
        }
        if (!bosToken.empty()) {
            rapidjson::Value tv;
            tv.SetString(bosToken.c_str(), resDocument.GetAllocator());
            v.AddMember("bos", tv, resDocument.GetAllocator());
        }
        if (!eosToken.empty()) {
            rapidjson::Value tv;
            tv.SetString(eosToken.c_str(), resDocument.GetAllocator());
            v.AddMember("eos", tv, resDocument.GetAllocator());
        }
        {
            rapidjson::Value tv;
            tv.SetString(res.c_str(), resDocument.GetAllocator());
            v.AddMember("res", tv, resDocument.GetAllocator());
        }
        resDocument.GetArray().PushBack(v, resDocument.GetAllocator());
    }
    rapidjson::StringBuffer buf;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> bufwriter(buf);
    resDocument.Accept(bufwriter);
    std::ofstream os("result.json");
    os << buf.GetString();

    return 0;
}
