HALF-PRECISION FLOATING POINT LIBRARY (Version 1.12.0)
------------------------------------------------------

This is a C++ header-only library to provide an IEEE 754 conformant 16-bit 
half-precision floating point type along with corresponding arithmetic 
operators, type conversions and common mathematical functions. It aims for both 
efficiency and ease of use, trying to accurately mimic the behaviour of the 
builtin floating point types at the best performance possible.


INSTALLATION AND REQUIREMENTS
-----------------------------

Comfortably enough, the library consists of just a single header file 
containing all the functionality, which can be directly included by your 
projects, without the neccessity to build anything or link to anything.

Whereas this library is fully C++98-compatible, it can profit from certain 
C++11 features. Support for those features is checked automatically at compile 
(or rather preprocessing) time, but can be explicitly enabled or disabled by 
defining the corresponding preprocessor symbols to either 1 or 0 yourself. This 
is useful when the automatic detection fails (for more exotic implementations) 
or when a feature should be explicitly disabled:

  - 'long long' integer type for mathematical functions returning 'long long' 
    results (enabled for VC++ 2003 and newer, gcc and clang, overridable with 
    'HALF_ENABLE_CPP11_LONG_LONG').

  - Static assertions for extended compile-time checks (enabled for VC++ 2010, 
    gcc 4.3, clang 2.9 and newer, overridable with 'HALF_ENABLE_CPP11_STATIC_ASSERT').

  - Generalized constant expressions (enabled for VC++ 2015, gcc 4.6, clang 3.1 
    and newer, overridable with 'HALF_ENABLE_CPP11_CONSTEXPR').

  - noexcept exception specifications (enabled for VC++ 2015, gcc 4.6, clang 3.0 
    and newer, overridable with 'HALF_ENABLE_CPP11_NOEXCEPT').

  - User-defined literals for half-precision literals to work (enabled for 
    VC++ 2015, gcc 4.7, clang 3.1 and newer, overridable with 
    'HALF_ENABLE_CPP11_USER_LITERALS').

  - Type traits and template meta-programming features from <type_traits> 
    (enabled for VC++ 2010, libstdc++ 4.3, libc++ and newer, overridable with 
    'HALF_ENABLE_CPP11_TYPE_TRAITS').

  - Special integer types from <cstdint> (enabled for VC++ 2010, libstdc++ 4.3, 
    libc++ and newer, overridable with 'HALF_ENABLE_CPP11_CSTDINT').

  - Certain C++11 single-precision mathematical functions from <cmath> for 
    an improved implementation of their half-precision counterparts to work 
    (enabled for VC++ 2013, libstdc++ 4.3, libc++ and newer, overridable with 
    'HALF_ENABLE_CPP11_CMATH').

  - Hash functor 'std::hash' from <functional> (enabled for VC++ 2010, 
    libstdc++ 4.3, libc++ and newer, overridable with 'HALF_ENABLE_CPP11_HASH').

The library has been tested successfully with Visual C++ 2005-2015, gcc 4.4-4.8 
and clang 3.1. Please contact me if you have any problems, suggestions or even 
just success testing it on other platforms.


DOCUMENTATION
-------------

Here follow some general words about the usage of the library and its 
implementation. For a complete documentation of its iterface look at the 
corresponding website http://half.sourceforge.net. You may also generate the 
complete developer documentation from the library's only include file's doxygen 
comments, but this is more relevant to developers rather than mere users (for 
reasons described below).

BASIC USAGE

To make use of the library just include its only header file half.hpp, which 
defines all half-precision functionality inside the 'half_float' namespace. The 
actual 16-bit half-precision data type is represented by the 'half' type. This 
type behaves like the builtin floating point types as much as possible, 
supporting the usual arithmetic, comparison and streaming operators, which 
makes its use pretty straight-forward:

    using half_float::half;
    half a(3.4), b(5);
    half c = a * b;
    c += 3;
    if(c > a)
	    std::cout << c << std::endl;

Additionally the 'half_float' namespace also defines half-precision versions 
for all mathematical functions of the C++ standard library, which can be used 
directly through ADL:

    half a(-3.14159);
    half s = sin(abs(a));
    long l = lround(s);

You may also specify explicit half-precision literals, since the library 
provides a user-defined literal inside the 'half_float::literal' namespace, 
which you just need to import (assuming support for C++11 user-defined literals):

    using namespace half_float::literal;
    half x = 1.0_h;

Furthermore the library provides proper specializations for 
'std::numeric_limits', defining various implementation properties, and 
'std::hash' for hashing half-precision numbers (assuming support for C++11 
'std::hash'). Similar to the corresponding preprocessor symbols from <cmath> 
the library also defines the 'HUGE_VALH' constant and maybe the 'FP_FAST_FMAH' 
symbol.

CONVERSIONS AND ROUNDING

The half is explicitly constructible/convertible from a single-precision float 
argument. Thus it is also explicitly constructible/convertible from any type 
implicitly convertible to float, but constructing it from types like double or 
int will involve the usual warnings arising when implicitly converting those to 
float because of the lost precision. On the one hand those warnings are 
intentional, because converting those types to half neccessarily also reduces 
precision. But on the other hand they are raised for explicit conversions from 
those types, when the user knows what he is doing. So if those warnings keep 
bugging you, then you won't get around first explicitly converting to float 
before converting to half, or use the 'half_cast' described below. In addition 
you can also directly assign float values to halfs.

In contrast to the float-to-half conversion, which reduces precision, the 
conversion from half to float (and thus to any other type implicitly 
convertible from float) is implicit, because all values represetable with 
half-precision are also representable with single-precision. This way the 
half-to-float conversion behaves similar to the builtin float-to-double 
conversion and all arithmetic expressions involving both half-precision and 
single-precision arguments will be of single-precision type. This way you can 
also directly use the mathematical functions of the C++ standard library, 
though in this case you will invoke the single-precision versions which will 
also return single-precision values, which is (even if maybe performing the 
exact same computation, see below) not as conceptually clean when working in a 
half-precision environment.

The default rounding mode for conversions from float to half uses truncation 
(round toward zero, but mapping overflows to infinity) for rounding values not 
representable exactly in half-precision. This is the fastest rounding possible 
and is usually sufficient. But by redefining the 'HALF_ROUND_STYLE' 
preprocessor symbol (before including half.hpp) this default can be overridden 
with one of the other standard rounding modes using their respective constants 
or the equivalent values of 'std::float_round_style' (it can even be 
synchronized with the underlying single-precision implementation by defining it 
to 'std::numeric_limits<float>::round_style'):

  - 'std::round_indeterminate' or -1 for the fastest rounding (default).

  - 'std::round_toward_zero' or 0 for rounding toward zero.

  - std::round_to_nearest' or 1 for rounding to the nearest value.

  - std::round_toward_infinity' or 2 for rounding toward positive infinity.

  - std::round_toward_neg_infinity' or 3 for rounding toward negative infinity.

In addition to changing the overall default rounding mode one can also use the 
'half_cast'. This converts between half and any built-in arithmetic type using 
a configurable rounding mode (or the default rounding mode if none is 
specified). In addition to a configurable rounding mode, 'half_cast' has 
another big difference to a mere 'static_cast': Any conversions are performed 
directly using the given rounding mode, without any intermediate conversion 
to/from 'float'. This is especially relevant for conversions to integer types, 
which don't necessarily truncate anymore. But also for conversions from 
'double' or 'long double' this may produce more precise results than a 
pre-conversion to 'float' using the single-precision implementation's current 
rounding mode would.

    half a = half_cast<half>(4.2);
    half b = half_cast<half,std::numeric_limits<float>::round_style>(4.2f);
    assert( half_cast<int, std::round_to_nearest>( 0.7_h )     == 1 );
    assert( half_cast<half,std::round_toward_zero>( 4097 )     == 4096.0_h );
    assert( half_cast<half,std::round_toward_infinity>( 4097 ) == 4100.0_h );
    assert( half_cast<half,std::round_toward_infinity>( std::numeric_limits<double>::min() ) > 0.0_h );

When using round to nearest (either as default or through 'half_cast') ties are 
by default resolved by rounding them away from zero (and thus equal to the 
behaviour of the 'round' function). But by redefining the 
'HALF_ROUND_TIES_TO_EVEN' preprocessor symbol to 1 (before including half.hpp) 
this default can be changed to the slightly slower but less biased and more 
IEEE-conformant behaviour of rounding half-way cases to the nearest even value.

    #define HALF_ROUND_TIES_TO_EVEN 1
    #include <half.hpp>
    ...
    assert( half_cast<int,std::round_to_nearest>(3.5_h) 
         == half_cast<int,std::round_to_nearest>(4.5_h) );

IMPLEMENTATION

For performance reasons (and ease of implementation) many of the mathematical 
functions provided by the library as well as all arithmetic operations are 
actually carried out in single-precision under the hood, calling to the C++ 
standard library implementations of those functions whenever appropriate, 
meaning the arguments are converted to floats and the result back to half. But 
to reduce the conversion overhead as much as possible any temporary values 
inside of lengthy expressions are kept in single-precision as long as possible, 
while still maintaining a strong half-precision type to the outside world. Only 
when finally assigning the value to a half or calling a function that works 
directly on halfs is the actual conversion done (or never, when further 
converting the result to float.

This approach has two implications. First of all you have to treat the 
library's documentation at http://half.sourceforge.net as a simplified version, 
describing the behaviour of the library as if implemented this way. The actual 
argument and return types of functions and operators may involve other internal 
types (feel free to generate the exact developer documentation from the Doxygen 
comments in the library's header file if you really need to). But nevertheless 
the behaviour is exactly like specified in the documentation. The other 
implication is, that in the presence of rounding errors or over-/underflows 
arithmetic expressions may produce different results when compared to 
converting to half-precision after each individual operation:

    half a = std::numeric_limits<half>::max() * 2.0_h / 2.0_h;       // a = MAX
    half b = half(std::numeric_limits<half>::max() * 2.0_h) / 2.0_h; // b = INF
    assert( a != b );

But this should only be a problem in very few cases. One last word has to be 
said when talking about performance. Even with its efforts in reducing 
conversion overhead as much as possible, the software half-precision 
implementation can most probably not beat the direct use of single-precision 
computations. Usually using actual float values for all computations and 
temproraries and using halfs only for storage is the recommended way. On the 
one hand this somehow makes the provided mathematical functions obsolete 
(especially in light of the implicit conversion from half to float), but 
nevertheless the goal of this library was to provide a complete and 
conceptually clean half-precision implementation, to which the standard 
mathematical functions belong, even if usually not needed.

IEEE CONFORMANCE

The half type uses the standard IEEE representation with 1 sign bit, 5 exponent 
bits and 10 mantissa bits (11 when counting the hidden bit). It supports all 
types of special values, like subnormal values, infinity and NaNs. But there 
are some limitations to the complete conformance to the IEEE 754 standard:

  - The implementation does not differentiate between signalling and quiet 
    NaNs, this means operations on halfs are not specified to trap on 
    signalling NaNs (though they may, see last point).

  - Though arithmetic operations are internally rounded to single-precision 
    using the underlying single-precision implementation's current rounding 
    mode, those values are then converted to half-precision using the default 
    half-precision rounding mode (changed by defining 'HALF_ROUND_STYLE' 
    accordingly). This mixture of rounding modes is also the reason why 
    'std::numeric_limits<half>::round_style' may actually return 
    'std::round_indeterminate' when half- and single-precision rounding modes 
    don't match.

  - Because of internal truncation it may also be that certain single-precision 
    NaNs will be wrongly converted to half-precision infinity, though this is 
    very unlikely to happen, since most single-precision implementations don't 
    tend to only set the lowest bits of a NaN mantissa.

  - The implementation does not provide any floating point exceptions, thus 
    arithmetic operations or mathematical functions are not specified to invoke 
    proper floating point exceptions. But due to many functions implemented in 
    single-precision, those may still invoke floating point exceptions of the 
    underlying single-precision implementation.

Some of those points could have been circumvented by controlling the floating 
point environment using <cfenv> or implementing a similar exception mechanism. 
But this would have required excessive runtime checks giving two high an impact 
on performance for something that is rarely ever needed. If you really need to 
rely on proper floating point exceptions, it is recommended to explicitly 
perform computations using the built-in floating point types to be on the safe 
side. In the same way, if you really need to rely on a particular rounding 
behaviour, it is recommended to either use single-precision computations and 
explicitly convert the result to half-precision using 'half_cast' and 
specifying the desired rounding mode, or synchronize the default half-precision 
rounding mode to the rounding mode of the single-precision implementation (most 
likely 'HALF_ROUND_STYLE=1', 'HALF_ROUND_TIES_TO_EVEN=1'). But this is really 
considered an expert-scenario that should be used only when necessary, since 
actually working with half-precision usually comes with a certain 
tolerance/ignorance of exactness considerations and proper rounding comes with 
a certain performance cost.


CREDITS AND CONTACT
-------------------

This library is developed by CHRISTIAN RAU and released under the MIT License 
(see LICENSE.txt). If you have any questions or problems with it, feel free to 
contact me at rauy@users.sourceforge.net.

Additional credit goes to JEROEN VAN DER ZIJP for his paper on "Fast Half Float 
Conversions", whose algorithms have been used in the library for converting 
between half-precision and single-precision values.
