
#ifndef MEMORY_FORMATER_H
#define MEMORY_FORMATER_H

#include "MNN/MNNDefine.h"
#include <vector>

inline void printDims(const std::vector<int>& dims) {
  int num_dims = dims.size();
  MNN_PRINT(" {");
  if (num_dims > 0) MNN_PRINT("%d", dims.at(0));
  for (size_t i = 1; i < num_dims; i++) {
    MNN_PRINT(", %d", dims.at(i));
  }
  MNN_PRINT("}");
}

inline float MNNBF16ToFP32(int16_t s16Value) {
    int32_t s32Value = ((int32_t)s16Value) << 16;
    float* fp32Value = (float*)(&s32Value);
    return *fp32Value;
}

inline void formatPrint(const char* prefix, const float& value, const char* suffix) {
    MNN_PRINT("%s%f%s", prefix, value, suffix);
}
inline void formatPrint(const char* prefix, const double& value, const char* suffix) {
   MNN_PRINT("%s%f%s", prefix, value, suffix);
}

inline void formatPrint(const char* prefix, const uint8_t& value, const char* suffix) {
   MNN_PRINT("%s%d%s", prefix, value, suffix);
}

inline void formatPrint(const char* prefix, const int8_t& value, const char* suffix) {
   MNN_PRINT("%s%d%s", prefix, value, suffix);
}

inline void formatPrint(const char* prefix, const int16_t& value, const char* suffix) {
   MNN_PRINT("%s%f%s", prefix, MNNBF16ToFP32(value), suffix);
}

inline void formatPrint(const char* prefix, const int& value, const char* suffix) {
   MNN_PRINT("%s%d%s", prefix, value, suffix);
}
inline void formatPrint(const char* prefix, const unsigned int& value, const char* suffix) {
   MNN_PRINT("%s%u%s", prefix, value, suffix);
}
inline void formatPrint(const char* prefix, const long int& value, const char* suffix) {
   MNN_PRINT("%s%ld%s", prefix, value, suffix);
}
inline void formatPrint(const char* prefix, const unsigned long& value, const char* suffix) {
   MNN_PRINT("%s%lu%s", prefix, value, suffix);
}

inline void formatPrint(const char* prefix, const long long& value, const char* suffix) {
   MNN_PRINT("%s%lld%s", prefix, value, suffix);
}
inline void formatPrint(const char* prefix, const unsigned long long& value, const char* suffix) {
   MNN_PRINT("%s%llu%s", prefix, value, suffix);
}


template <typename ElementType>
inline void formatMatrix(ElementType* data, std::vector<int> dims) {

  const int MaxLines = 100;

  MNN_PRINT("shape:");
  printDims(dims);
  MNN_PRINT("\n");
  while (dims.size() > 1) {
    if (*(dims.end() - 1) == 1) {
        dims.erase(dims.end() - 1);
    } else {
        break;
    }
  }

  if (dims.size() == 0) {
    formatPrint("scalar:", *data, "\n");
    return;
  }
  int highDim = dims[0];
  const int lines = highDim < MaxLines ? highDim : MaxLines;
  const int tailStart = highDim - MaxLines > lines ? highDim - MaxLines : lines;

  // MNN_PRINT("\n{");
  if (dims.size() == 1) { // output elements in the last dim in a row.
    MNN_PRINT("{");
    for (int i = 0; i < lines; i++) {
      formatPrint("", data[i], ", ");
    }
    if (tailStart > lines) {
      formatPrint(", …skip middle ", tailStart - lines, "…,");
    }
    for (int i = tailStart; i < highDim; i++) {
      formatPrint("", data[i], ", ");
    }
    MNN_PRINT("}");
    return;

  } else {
    dims.erase(dims.begin());

    int step = dims[0];
    for (size_t i = 1; i < dims.size(); i++) {
      step *= dims[i];
    }

    for (int i = 0; i < lines; i++) {
      formatPrint("", i, " th:");
      formatMatrix(data + i * step, dims);
      MNN_PRINT("\n");
    }
    if (tailStart > lines) {
      formatPrint("{…skip middle ", tailStart - lines, " …}\n");
    }

    for (int i = tailStart; i < highDim; i++) {
      formatPrint("", i, " th:");
      formatMatrix(data + i * step, dims);
      MNN_PRINT("\n");
    }

    return;

  }
}

#endif


