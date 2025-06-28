// sherpa-mnn/csrc/cppjieba-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <iostream>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "cppjieba/Jieba.hpp"
#include "gtest/gtest.h"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

// Please download dict files form
// https://github.com/csukuangfj/cppjieba/releases/download/sherpa-mnn-2024-04-19/dict.tar.bz2
const char *const kDictPath = "./dict/jieba.dict.utf8";
const char *const kHmmPath = "./dict/hmm_model.utf8";
const char *const kUserDictPath = "./dict/user.dict.utf8";
const char *const kIdfPath = "./dict/idf.utf8";
const char *const kStopWordPath = "./dict/stop_words.utf8";

TEST(CppJieBa, Case1) {
  if (!FileExists(kDictPath)) {
    SHERPA_ONNX_LOGE("%s does not exist. Skipping test", kDictPath);
    return;
  }

  cppjieba::Jieba jieba(kDictPath, kHmmPath, kUserDictPath, kIdfPath,
                        kStopWordPath);

  std::vector<std::string> words;
  std::vector<cppjieba::Word> jiebawords;

  std::string s = "他来到了网易杭研大厦。How are you?";
  std::cout << s << std::endl;
  std::cout << "[demo] Cut With HMM" << std::endl;
  jieba.Cut(s, words, true);
  std::cout << limonp::Join(words.begin(), words.end(), "/") << std::endl;
  /*
  他来到了网易杭研大厦
  [demo] Cut With HMM
  他/来到/了/网易/杭研/大厦
  */
  s = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造";
  std::cout << s << std::endl;
  std::cout << "[demo] CutForSearch" << std::endl;
  jieba.CutForSearch(s, words);
  std::cout << limonp::Join(words.begin(), words.end(), "/") << std::endl;
  /*
  小明硕士毕业于中国科学院计算所，后在日本京都大学深造
  [demo] CutForSearch
  小明/硕士/毕业/于/中国/科学/学院/科学院/中国科学院/计算/计算所/，/后/在/日本/京都/大学/日本京都大学/深造
   */
  std::cout << "[demo] Insert User Word" << std::endl;
  jieba.Cut("男默女泪", words);
  std::cout << limonp::Join(words.begin(), words.end(), "/") << std::endl;
  jieba.InsertUserWord("男默女泪");
  jieba.Cut("男默女泪", words);
  std::cout << limonp::Join(words.begin(), words.end(), "/") << std::endl;
  /*
  [demo] Insert User Word
  男默/女泪
  男默女泪
  */
  std::cout << "[demo] CutForSearch Word With Offset" << std::endl;
  jieba.CutForSearch(s, jiebawords, true);
  std::cout << jiebawords << std::endl;
  /*
[demo] CutForSearch Word With Offset
[{"word": "小明", "offset": 0}, {"word": "硕士", "offset": 6}, {"word": "毕业",
"offset": 12}, {"word": "于", "offset": 18}, {"word": "中国", "offset": 21},
{"word": "科学", "offset": 27}, {"word": "学院", "offset": 30}, {"word":
"科学院", "offset": 27}, {"word": "中国科学院", "offset": 21}, {"word": "计算",
"offset": 36}, {"word": "计算所", "offset": 36}, {"word": "，", "offset": 45},
{"word": "后", "offset": 48}, {"word": "在", "offset": 51}, {"word": "日本",
"offset": 54}, {"word": "京都", "offset": 60}, {"word": "大学", "offset": 66},
{"word": "日本京都大学", "offset": 54}, {"word": " 深造", "offset": 72}]
   */
  // see more test at
  // https://github.com/yanyiwu/cppjieba/blob/master/test/demo.cpp
}

TEST(CppJieBa, Case2) {
  if (!FileExists(kDictPath)) {
    SHERPA_ONNX_LOGE("%s does not exist. Skipping test", kDictPath);
    return;
  }

  cppjieba::Jieba jieba(kDictPath, kHmmPath, kUserDictPath, kIdfPath,
                        kStopWordPath);
  std::string s =
      "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如"
      "涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感"
      "受着生命的奇迹与温柔";
  std::vector<std::string> words;
  bool is_hmm = true;
  jieba.Cut(s, words, is_hmm);
  {
    std::ostringstream os;
    std::string sep = "";
    for (const auto &w : words) {
      os << sep << w;
      sep = "_";
    }

    std::cout << os.str() << "\n";
  }
  /*
当_夜幕降临_，_星光点点_，_伴随_着_微风_拂面_，
_我_在_静谧_中_感受_着_时光_的_流转_，
_思念_如_涟漪_荡漾_，_梦境_如_画卷_展开_，_我_与_自然_融为一体_，
_沉静_在_这_片_宁静_的_美丽_之中_，_感受_着_生命_的_奇迹_与_温柔
   */
  s = "这里有：红的、绿的、蓝的；各种各样的颜色都有!你想要什么呢?测试.";
  std::regex punct_re("：|、|；");
  std::string s2 = std::regex_replace(s, punct_re, "，");

  std::regex punct_re2("[.]");
  s2 = std::regex_replace(s2, punct_re2, "。");

  std::regex punct_re3("[?]");
  s2 = std::regex_replace(s2, punct_re3, "？");

  std::regex punct_re4("[!]");
  s2 = std::regex_replace(s2, punct_re4, "！");
  std::cout << s << "\n" << s2 << "\n";

  words.clear();
  jieba.Cut(s2, words, is_hmm);
  {
    std::ostringstream os;
    std::string sep = "";
    for (const auto &w : words) {
      os << sep << w;
      sep = "_";
    }

    std::cout << os.str() << "\n";
  }
}

}  // namespace sherpa_mnn
