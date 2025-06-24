#pragma once

#include <string>
#include <tuple>
#include <vector>

typedef std::vector<int16_t> Audio;

class MNNTTSImplBase
{
public:
  // 虚析构函数，确保通过基类指针删除派生类对象时能正确调用派生类的析构函数
  virtual ~MNNTTSImplBase() = default;

  // 示例接口方法
  virtual std::tuple<int, Audio> Process(const std::string &text) = 0;

private:
  int sample_rate_ = 16000;
};