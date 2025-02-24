#include <algorithm>
#include <vector>
#include <cmath>
#include <llm/llm.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <iterator>
#include <random>
#include "dataset.hpp"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

namespace MNN {
namespace Transformer {


// parse file
// csv json

// parse csv
std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

// dialog, turn,
void parse_jsonl(std::string prompt_file, std::vector<std::vector<std::vector<ChatMessage>>>& dialogs) {
    std::ifstream prompt_fs(prompt_file);
    std::string prompt;
    while(std::getline(prompt_fs, prompt)) {
        rapidjson::Document document;
        document.Parse(prompt.c_str());
        std::vector<std::vector<ChatMessage>> cnv;
        if(document.HasMember("conversation")) {
            auto& value = document["conversation"];
            if (value.IsArray()) {
                for (auto& v : value.GetArray()) {
                    if (v.IsObject()) {
                        std::vector<ChatMessage> result;
                        for (auto itr = v.MemberBegin(); itr != v.MemberEnd(); ++itr) {
                            // {"human"/"user": , "assistant": }
                            result.push_back(std::make_pair(itr->name.GetString(), itr->value.GetString()));
                        }
                        cnv.push_back(result);
                    }
                }
            }
        }
        dialogs.push_back(cnv);
    }
}

void write_jsonl(std::string prompt_file, const std::vector<std::vector<std::vector<ChatMessage>>>& dialogs) {
    std::ofstream prompt_fs(prompt_file);
    for(auto& dialog : dialogs) {
        rapidjson::Document document;
        document.SetObject();
        rapidjson::Value conversation(rapidjson::kArrayType);
        conversation.SetArray();
        for (auto& turn : dialog) {
            rapidjson::Value sentence(rapidjson::kObjectType);
            sentence.SetObject();
            for (auto& role : turn) {
                sentence.AddMember(rapidjson::Value(role.first.c_str(), document.GetAllocator()),
                                     rapidjson::Value(role.second.c_str(), document.GetAllocator()), document.GetAllocator());
            }
            conversation.PushBack(sentence, document.GetAllocator());
        }
        document.AddMember("conversation", conversation, document.GetAllocator());
        // write to file
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        prompt_fs << buffer.GetString() << std::endl;
    }
}


// dataset
// wikitext, ShareGPT

std::string getPPLType(std::string dataset_name) {
    if (dataset_name == "wikitext"
        || dataset_name == "plaintext"
        || dataset_name == "rowsplit") {
        return "text";
    } else if (dataset_name == "shareGPT") {
        return "chat";
    } else {
        // default chat
        return "chat";
    }
}

std::vector<std::string> plaintext(std::string prompt_file) {
    // split by line
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    prompts.push_back("");
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r' || prompt.back() == '\n') {
            prompt.pop_back();
        }
        // concatenate.
        prompts.back() += prompt + "\n";
    }
    return prompts;
}

std::vector<std::string> rowsplit(std::string prompt_file) {
    // split by line
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r' || prompt.back() == '\n') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    return prompts;
}

// wikitext
void removeSubstrs(std::string& s, std::string p) {
    std::string::size_type n = p.length();
    for (std::string::size_type i = s.find(p); i != std::string::npos; i = s.find(p))
        s.erase(i, n);
}
std::vector<std::string> wikitext(std::string prompt_file) {
    // split wiki text into " = " first-level column.
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r' || prompt.back() == '\n') {
            prompt.pop_back();
        }
        if (prompt.size() < 4) continue;
        removeSubstrs(prompt, "@-@");
        if ((prompts.size() == 0) \
             || (prompt.size() >= 4 \
                 && prompt.at(0) == ' ' \
                 && prompt.at(1) == '=' \
                 && prompt.at(2) == ' ' \
                 && prompt.at(3) != '=')) {
            // first-level column.
            prompts.push_back(prompt);
        } else {
            // concatenate.
            prompts.back() += "\n" + prompt;
        }
    }
    return prompts;
}

std::string genSampleName(std::string oriName, int sample_size) {
    const size_t last_slash_idx = oriName.rfind('.');
    auto stem = oriName.substr(0, last_slash_idx);
    return stem + "_sample" + std::to_string(sample_size) + ".jsonl";
}

std::vector<std::vector<std::vector<ChatMessage>>> shareGPT(std::string prompt_file, int sample_size) {
    std::vector<std::vector<std::vector<ChatMessage>>> dialogs, dataset;
    parse_jsonl(prompt_file, dialogs);
    // randomly sample a subset
    if (sample_size > 0 && sample_size < dialogs.size()){
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(dialogs.begin(), dialogs.end(), g);
        dataset.insert(dataset.end(), dialogs.begin(), dialogs.begin() + sample_size);
        dialogs = dataset;
        // store dialogs to file
        write_jsonl(genSampleName(prompt_file, sample_size), dialogs);
    }
    return dialogs;
}


} // Transformer
} // MNN
