#ifndef SIMDHEADER_HPP
#define SIMDHEADER_HPP
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#ifdef MNN_USE_SSE
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__EMSCRIPTEN__)
#include <smmintrin.h>
#else
#include <x86intrin.h>
#endif
#endif
#endif
