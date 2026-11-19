#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mnncli {

// Process-local index of immutable CPU prefix files; live KV files must never be indexed directly.
class PromptCheckpointStore {
public:
    struct Match {
        std::string name;
        size_t tokens = 0;
    };

    PromptCheckpointStore() = default;
    PromptCheckpointStore(const PromptCheckpointStore&) = delete;
    PromptCheckpointStore& operator=(const PromptCheckpointStore&) = delete;
    ~PromptCheckpointStore() { clear(); }

    bool configure(const std::string& directory, size_t slots, size_t bytes) {
        if (directory != directory_ || slots != slots_ || bytes != limit_) clear();
        directory_ = directory;
        slots_ = slots;
        limit_ = bytes;
        std::error_code error;
        std::filesystem::create_directories(directory_, error);
        if (error || slots_ == 0 || limit_ == 0) return false;
        const auto probe = std::filesystem::path(directory_) / (nextName() + ".probe");
        std::ofstream file(probe, std::ios::binary);
        if (!file) return false;
        file.close();
        std::filesystem::remove(probe, error);
        return !error;
    }

    Match find(const std::vector<int>& tokens) {
        Entry* best = nullptr;
        for (auto& entry : entries_) {
            if (entry.tokens.size() < tokens.size() && (!best || entry.tokens.size() > best->tokens.size()) &&
                std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin())) best = &entry;
        }
        if (!best) return {};
        best->used = ++clock_;
        return {best->name, best->tokens.size()};
    }

    std::string nextName() {
        return "mnncli-prefix-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(++clock_);
    }

    bool save(const std::string& working, const std::vector<int>& tokens) {
        if (slots_ == 0 || limit_ == 0 || tokens.empty()) return false;
        std::vector<std::filesystem::path> files;
        if (!filesFor(working, files)) return false;
        size_t bytes = 0;
        bool marker = false;
        for (const auto& file : files) {
            std::error_code error;
            if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(file, error)) || error) return false;
            const auto size = std::filesystem::file_size(file, error);
            if (error || size > limit_ - bytes) return false;
            bytes += size;
            marker |= file.filename() == working + "_0_sync.k";
        }
        if (!marker) return false;
        const auto name = nextName();
        std::vector<std::filesystem::path> copies;
        copies.reserve(files.size());
        for (const auto& file : files) {
            std::error_code error;
            const auto target = std::filesystem::path(directory_) /
                (name + file.filename().string().substr(working.size()));
            copies.push_back(target);
            std::filesystem::copy_file(file, target, std::filesystem::copy_options::none, error);
            if (error) {
                removeFiles(copies);
                return false;
            }
        }
        // Commit only complete snapshots; a failed copy must not evict usable state.
        while (!entries_.empty() && (entries_.size() >= slots_ || used_bytes_ > limit_ - bytes)) {
            auto oldest = std::min_element(entries_.begin(), entries_.end(),
                [](const Entry& a, const Entry& b) { return a.used < b.used; });
            removeFiles(oldest->files);
            used_bytes_ -= oldest->bytes;
            entries_.erase(oldest);
        }
        entries_.push_back({name, tokens, bytes, ++clock_, std::move(copies)});
        used_bytes_ += bytes;
        return true;
    }

    void forget(const std::string& name) {
        // The engine may already be rebuilding these files as writable state.
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
            [&](const Entry& entry) {
                if (entry.name != name) return false;
                used_bytes_ -= entry.bytes;
                return true;
            }), entries_.end());
    }

    void removeFiles(const std::string& name) const {
        std::vector<std::filesystem::path> files;
        filesFor(name, files);
        removeFiles(files);
    }

    void clear() {
        for (const auto& entry : entries_) removeFiles(entry.files);
        entries_.clear();
        used_bytes_ = 0;
    }

private:
    struct Entry {
        std::string name;
        std::vector<int> tokens;
        size_t bytes;
        size_t used;
        std::vector<std::filesystem::path> files;
    };

    static void removeFiles(const std::vector<std::filesystem::path>& files) {
        for (const auto& file : files) {
            std::error_code error;
            std::filesystem::remove(file, error);
        }
    }

    bool filesFor(const std::string& name, std::vector<std::filesystem::path>& result) const {
        if (name.empty() || directory_.empty()) return false;
        std::error_code error;
        std::filesystem::directory_iterator it(directory_, error), end;
        while (!error && it != end) {
            if (it->path().filename().string().find(name + "_") == 0) result.push_back(it->path());
            it.increment(error);
        }
        return !error;
    }

    std::string directory_;
    std::vector<Entry> entries_;
    size_t slots_ = 0, limit_ = 0, clock_ = 0, used_bytes_ = 0;
};
}
