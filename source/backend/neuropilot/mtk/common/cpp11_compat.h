#pragma once

#include <memory>
#include <type_traits>
#include <string>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

namespace mtk {
namespace cpp11_compat {

// make_unique implementation for C++11
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Type trait helpers for C++11 compatibility
template<typename T>
struct is_signed : std::is_signed<T> {};

template<typename T>
struct is_same : std::is_same<T, T> {};

template<typename Base, typename Derived>
struct is_base_of : std::is_base_of<Base, Derived> {};

template<typename T, typename... Args>
struct is_constructible : std::is_constructible<T, Args...> {};

template<typename From, typename To>
struct is_convertible : std::is_convertible<From, To> {};

// enable_if helper
template<bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

// simple string_view implementation for C++11
class string_view {
public:
    string_view() : data_(nullptr), size_(0) {}
    string_view(const char* data, size_t size) : data_(data), size_(size) {}
    string_view(const char* data) : data_(data), size_(data ? strlen(data) : 0) {}
    string_view(const std::string& str) : data_(str.data()), size_(str.size()) {}
    
    const char* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    const char& operator[](size_t pos) const { return data_[pos]; }
    
private:
    const char* data_;
    size_t size_;
};

// filesystem operations for C++11 compatibility
namespace fs {
    inline bool exists(const std::string& path) {
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
    }
    
    inline bool create_directories(const std::string& path) {
        std::string dir = path;
        if (dir.back() != '/') dir += '/';
        
        for (size_t i = 1; i < dir.size(); ++i) {
            if (dir[i] == '/') {
                std::string subdir = dir.substr(0, i);
                if (!exists(subdir)) {
                    if (mkdir(subdir.c_str(), 0755) != 0 && errno != EEXIST) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    class path {
    public:
        path(const std::string& p) : path_str(p) {}
        
        path operator/(const std::string& other) const {
            std::string result = path_str;
            if (!result.empty() && result.back() != '/') {
                result += '/';
            }
            result += other;
            return path(result);
        }
        
        path stem() const {
            std::string filename = path_str.substr(path_str.find_last_of('/') + 1);
            size_t dot_pos = filename.find_last_of('.');
            if (dot_pos != std::string::npos) {
                filename = filename.substr(0, dot_pos);
            }
            return path(filename);
        }
        
        std::string string() const {
            return path_str;
        }
        
    private:
        std::string path_str;
    };
}

} // namespace cpp11_compat
} // namespace mtk
