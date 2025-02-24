#ifndef LLM_DATASET_hpp
#define LLM_DATASET_hpp

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "llm/llm.hpp"

#include <MNN/MNNDefine.h>

namespace MNN {
namespace Transformer {


// parse csv
MNN_PUBLIC std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines);
void parse_jsonl(std::string prompt_file, std::vector<std::vector<std::vector<ChatMessage>>>& dialogs);

std::string getPPLType(std::string dataset_name);
std::vector<std::string> rowsplit(std::string prompt_file);
std::vector<std::string> plaintext(std::string prompt_file);
std::vector<std::string> wikitext(std::string prompt_file);
std::vector<std::vector<std::vector<ChatMessage>>> shareGPT(std::string prompt_file, int sample_size=-1); // -1: no sampling

} // Transformer
} // MNN

#endif // LLM_DATASET_hpp