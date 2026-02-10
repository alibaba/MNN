#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

class MNNTTSConfig
{
public:
  explicit MNNTTSConfig(const std::string &config_file_path);
  
  // 支持参数覆盖的构造函数
  MNNTTSConfig(const std::string &config_file_path, 
               const std::map<std::string, std::string> &overrides);

  // 应用参数覆盖
  void applyOverrides(const std::map<std::string, std::string> &overrides);

  // 模板方法的实现必须放在头文件中或者在源文件中模板实例化
  template <typename T>
  T get_value_from_json(const nlohmann::json &j, const std::string &key) const
  {
    if (!j.contains(key))
    {
      throw std::runtime_error("Missing key in config.json: '" + key + "'");
    }
    try
    {
      return j.at(key).get<T>();
    }
    catch (const nlohmann::json::exception &e)
    {
      throw std::runtime_error("Type mismatch for key '" + key + "': " + e.what());
    }
  }

private:
  // 原始的JSON对象，如果需要更灵活的访问
  nlohmann::json raw_config_data_;

public:
  std::string model_type_;
  std::string model_path_;
  std::string asset_folder_;
  std::string cache_folder_;
  int sample_rate_;
};