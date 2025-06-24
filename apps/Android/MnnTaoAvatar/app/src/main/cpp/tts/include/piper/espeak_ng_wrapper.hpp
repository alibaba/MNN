#pragma once

// This wrapper is intended to safely include espeak-ng's speak_lib.h
// by managing potential macro conflicts with C++ standard library headers.

// Include any C++ standard headers that might be indirectly included by other
// parts of the project (e.g., MNN) and could conflict with espeak-ng macros.
// This helps ensure that the C++ versions are seen by the compiler first.
#include <cctype> // For std::tolower, std::toupper
#include <locale> // Often included with iostream or other C++ std libs

// Undefine the macros from espeak-ng's compat/wctype.h that conflict
// with the C++ standard library functions.
#ifdef tolower
#undef tolower
#endif

#ifdef toupper
#undef toupper
#endif

// Now, safely include the actual espeak-ng header.
// The path should be relative to the include paths configured in CMakeLists.txt.
// Assuming 'third-party/espeak-ng/src/include' is an include directory.
#include "speak_lib.h"
