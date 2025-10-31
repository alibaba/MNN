#pragma once

#include <type_traits>

// Safe comparisons betweeen types that may contain mixture between signed and unsigned types.
// Based on https://www.sandordargo.com/blog/2023/10/11/cpp20-intcmp-utilities

namespace cmp {

namespace detail {
    template <class T, class U, bool same_sign>
    struct eq_impl;
    
    template <class T, class U>
    struct eq_impl<T, U, true> {
        static constexpr bool call(T t, U u) noexcept { return t == u; }
    };
    
    template <class T, class U>
    struct eq_impl<T, U, false> {
        static constexpr bool call(T t, U u) noexcept {
            if (std::is_signed<T>::value) {
                return t >= 0 && static_cast<typename std::make_unsigned<T>::type>(t) == u;
            } else {
                return u >= 0 && static_cast<typename std::make_unsigned<U>::type>(u) == t;
            }
        }
    };
}

template <class T, class U>
constexpr bool eq(T t, U u) noexcept {
    return detail::eq_impl<T, U, std::is_signed<T>::value == std::is_signed<U>::value>::call(t, u);
}

template <class T, class U>
constexpr bool ne(T t, U u) noexcept {
    return !eq(t, u);
}

namespace detail {
    template <class T, class U, bool same_sign>
    struct lt_impl;
    
    template <class T, class U>
    struct lt_impl<T, U, true> {
        static constexpr bool call(T t, U u) noexcept { return t < u; }
    };
    
    template <class T, class U>
    struct lt_impl<T, U, false> {
        static constexpr bool call(T t, U u) noexcept {
            if (std::is_signed<T>::value) {
                return t < 0 || static_cast<typename std::make_unsigned<T>::type>(t) < u;
            } else {
                return u >= 0 && t < static_cast<typename std::make_unsigned<U>::type>(u);
            }
        }
    };
}

template <class T, class U>
constexpr bool lt(T t, U u) noexcept {
    return detail::lt_impl<T, U, std::is_signed<T>::value == std::is_signed<U>::value>::call(t, u);
}

template <class T, class U>
constexpr bool gt(T t, U u) noexcept {
    return lt(u, t);
}

template <class T, class U>
constexpr bool le(T t, U u) noexcept {
    return !lt(u, t);
}

template <class T, class U>
constexpr bool ge(T t, U u) noexcept {
    return !lt(t, u);
}

} // namespace cmp