#include "mnncli_agent.hpp"
#include "mnncli_mcp.hpp"

#include <algorithm>
#include <climits>
#include <filesystem>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <vector>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace mnncli {
namespace {
using json = nlohmann::json;

std::string randomId() {
    unsigned char bytes[24];
    if (RAND_bytes(bytes, sizeof(bytes)) != 1) return "";
    const char* digits = "0123456789abcdef";
    std::string id;
    id.reserve(sizeof(bytes) * 2);
    for (auto byte : bytes) { id.push_back(digits[byte >> 4]); id.push_back(digits[byte & 15]); }
    return id;
}

std::string mcpToolId(const std::string& server, const std::string& name) {
    const auto value = server + "\n" + name;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(), digest);
    const char* digits = "0123456789abcdef";
    auto safe = [](const std::string& value, size_t limit) {
        std::string out;
        for (const unsigned char c : value) {
            if (out.size() >= limit) break;
            out.push_back((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ? static_cast<char>(c) : '_');
        }
        return out;
    };
    std::string id = "mcp_" + safe(server, 16) + "_" + safe(name, 24) + "_";
    for (size_t i = 0; i < 6; ++i) { id.push_back(digits[digest[i] >> 4]); id.push_back(digits[digest[i] & 15]); }
    return id;
}

bool relativePath(const std::string& path, bool allow_root) {
    if (path.empty() || path.size() > 4096 || path.find('\0') != std::string::npos || path[0] == '/') return false;
    if (path == ".") return allow_root;
    size_t start = 0;
    while (start < path.size()) {
        const auto end = path.find('/', start);
        const auto part = path.substr(start, end == std::string::npos ? end : end - start);
        if (part.empty() || part == "." || part == "..") return false;
        if (end == std::string::npos) return true;
        start = end + 1;
    }
    return false;
}

std::string encodeBase64(const std::string& bytes) {
    std::string encoded(4 * ((bytes.size() + 2) / 3), '\0');
    if (!bytes.empty()) EVP_EncodeBlock(reinterpret_cast<unsigned char*>(&encoded[0]),
                                        reinterpret_cast<const unsigned char*>(bytes.data()), static_cast<int>(bytes.size()));
    return encoded;
}

bool validUtf8(const std::string& text) {
    for (size_t i = 0; i < text.size();) {
        auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) { ++i; continue; }
        size_t count = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        if (!count || i + count > text.size()) return false;
        auto next = static_cast<unsigned char>(text[i + 1]);
        if ((next & 0xc0) != 0x80 || (c == 0xe0 && next < 0xa0) || (c == 0xed && next >= 0xa0) ||
            (c == 0xf0 && next < 0x90) || (c == 0xf4 && next >= 0x90)) return false;
        for (size_t j = 2; j < count; ++j)
            if ((static_cast<unsigned char>(text[i + j]) & 0xc0) != 0x80) return false;
        i += count;
    }
    return true;
}

#ifndef _WIN32
class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) : fd_(fd) {}
    ~FileDescriptor() { if (fd_ >= 0) close(fd_); }
    int get() const { return fd_; }
    int release() { int fd = fd_; fd_ = -1; return fd; }
    void reset(int fd) { if (fd_ >= 0) close(fd_); fd_ = fd; }
private:
    int fd_;
};

bool openParent(const std::string& root, const std::string& path, FileDescriptor& directory,
                std::string& name, std::string& error) {
    directory.reset(open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (directory.get() < 0) { error = "Cannot open workspace"; return false; }
    if (path == ".") { name = "."; return true; }
    size_t start = 0;
    while (true) {
        const auto end = path.find('/', start);
        const auto part = path.substr(start, end == std::string::npos ? end : end - start);
        if (end == std::string::npos) { name = part; return true; }
        const int next = openat(directory.get(), part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) { error = "Cannot open directory or path contains a symbolic link"; return false; }
        directory.reset(next);
        start = end + 1;
    }
}

bool fileAction(const json& action, json& result, std::string& error) {
    const std::string tool = action["tool"], root = action["workspace"], path = action["arguments"]["path"];
    FileDescriptor parent;
    std::string name;
    if (!openParent(root, path, parent, name, error)) return false;
    const auto& args = action["arguments"];
    if (tool == "list_directory") {
        FileDescriptor dir(path == "." ? dup(parent.get()) : openat(parent.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
        if (dir.get() < 0) { error = "Cannot open directory or path is a symbolic link"; return false; }
        DIR* entries = fdopendir(dir.release());
        if (!entries) { error = "Cannot list directory"; return false; }
        json items = json::array();
        bool truncated = false;
        while (auto entry = readdir(entries)) {
            const std::string item = entry->d_name;
            if (item == "." || item == "..") continue;
            if (items.size() >= 200) { truncated = true; break; }
            struct stat info;
            if (fstatat(dirfd(entries), item.c_str(), &info, AT_SYMLINK_NOFOLLOW) != 0) continue;
            const char* type = S_ISDIR(info.st_mode) ? "directory" : S_ISREG(info.st_mode) ? "file" : S_ISLNK(info.st_mode) ? "symlink" : "other";
            items.push_back({{"name", validUtf8(item) ? json(item) : json("[non-UTF-8 name]")}, {"type", type}, {"size", info.st_size}});
        }
        closedir(entries);
        result = {{"path", path}, {"entries", items}, {"truncated", truncated}};
        return true;
    }
    if (tool == "read_file") {
        FileDescriptor file(openat(parent.get(), name.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
        struct stat info;
        if (file.get() < 0 || fstat(file.get(), &info) != 0 || !S_ISREG(info.st_mode)) { error = "Cannot read a regular file at this path"; return false; }
        const auto offset = args.value("offset", 0);
        const auto limit = args.value("max_bytes", 32768);
        if (lseek(file.get(), offset, SEEK_SET) < 0) { error = "Cannot seek file"; return false; }
        std::string data(static_cast<size_t>(limit), '\0');
        ssize_t length = read(file.get(), &data[0], data.size());
        if (length < 0) { error = "Cannot read file"; return false; }
        data.resize(static_cast<size_t>(length));
        const bool utf8 = validUtf8(data);
        const auto next = static_cast<int64_t>(offset) + length;
        result = {{"path", path}, {"encoding", utf8 ? "utf-8" : "base64"}, {"content", utf8 ? data : encodeBase64(data)},
                  {"offset", offset}, {"next_offset", next < info.st_size ? json(next) : json(nullptr)}, {"total_bytes", info.st_size}};
        return true;
    }
    const bool overwrite = args.value("overwrite", false);
    struct stat existing;
    const bool exists = fstatat(parent.get(), name.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0;
    if (exists && (!overwrite || !S_ISREG(existing.st_mode))) { error = "File exists or is not a regular file"; return false; }
    const auto temporary = ".mnncli-" + randomId();
    if (temporary == ".mnncli-") { error = "Cannot generate temporary file name"; return false; }
    FileDescriptor file(openat(parent.get(), temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    if (file.get() < 0) { error = "Cannot create temporary file"; return false; }
    const std::string content = args["content"];
    size_t written = 0;
    while (written < content.size()) {
        const ssize_t n = write(file.get(), content.data() + written, content.size() - written);
        if (n <= 0) { unlinkat(parent.get(), temporary.c_str(), 0); error = "Cannot write file"; return false; }
        written += static_cast<size_t>(n);
    }
    if (exists) fchmod(file.get(), existing.st_mode & 0777);
    const int rc = overwrite ? renameat(parent.get(), temporary.c_str(), parent.get(), name.c_str())
                             : linkat(parent.get(), temporary.c_str(), parent.get(), name.c_str(), 0);
    if (!overwrite || rc != 0) unlinkat(parent.get(), temporary.c_str(), 0);
    if (rc != 0) { error = "Cannot replace file or destination already exists"; return false; }
    result = {{"path", path}, {"bytes_written", written}, {"replaced", exists}};
    return true;
}

bool commandAction(const json& action, json& result, std::string& error) {
    const auto& args = action["arguments"];
    FileDescriptor parent;
    std::string name;
    if (!openParent(action["workspace"], args.value("cwd", "."), parent, name, error)) return false;
    FileDescriptor cwd(name == "." ? dup(parent.get()) : openat(parent.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (cwd.get() < 0) { error = "Cannot open command working directory"; return false; }
    int pipefd[2];
    if (pipe(pipefd) != 0) { error = "Cannot create command pipe"; return false; }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);
#if defined(__APPLE__) || defined(__GLIBC__)
    if (posix_spawn_file_actions_addfchdir_np(&actions, cwd.get()) != 0) {
        posix_spawn_file_actions_destroy(&actions);
        close(pipefd[0]); close(pipefd[1]);
        error = "Cannot set command working directory"; return false;
    }
#else
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[0]); close(pipefd[1]);
    error = "Command working directory is unsupported on this platform"; return false;
#endif
    const std::string command = args["command"];
    const char* argv[] = {"/bin/sh", "-c", command.c_str(), nullptr};
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    pid_t pid;
    const int spawn_error = posix_spawn(&pid, "/bin/sh", &actions, &attr, const_cast<char* const*>(argv), environ);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);
    if (spawn_error != 0) { close(pipefd[0]); error = "Cannot start command"; return false; }
    FileDescriptor output(pipefd[0]);
    fcntl(output.get(), F_SETFL, fcntl(output.get(), F_GETFL, 0) | O_NONBLOCK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(args.value("timeout_seconds", 30));
    std::string captured;
    bool eof = false, exited = false, timed_out = false, truncated = false;
    int status = 0;
    while (!eof || !exited) {
        char buffer[8192];
        ssize_t n = -1;
        for (int batch = 0; batch < 16 && (n = read(output.get(), buffer, sizeof(buffer))) > 0; ++batch) {
            const auto space = static_cast<size_t>(131072) - captured.size();
            const auto keep = std::min(space, static_cast<size_t>(n));
            captured.append(buffer, keep);
            if (keep < static_cast<size_t>(n)) truncated = true;
        }
        if (n == 0) eof = true;
        if (!exited && waitpid(pid, &status, WNOHANG) == pid) exited = true;
        if (std::chrono::steady_clock::now() >= deadline) {
            timed_out = true;
            kill(-pid, SIGKILL);
            if (!exited) waitpid(pid, &status, 0);
            break;
        }
        if (!eof) { struct pollfd p = {output.get(), POLLIN, 0}; poll(&p, 1, 100); }
        else if (!exited) { struct pollfd p = {-1, 0, 0}; poll(&p, 0, 100); }
    }
    const bool utf8 = validUtf8(captured);
    result = {{"output", utf8 ? captured : encodeBase64(captured)}, {"encoding", utf8 ? "utf-8" : "base64"},
              {"truncated", truncated}, {"timed_out", timed_out},
              {"exit_code", !timed_out && WIFEXITED(status) ? json(WEXITSTATUS(status)) : json(nullptr)},
              {"signal", !timed_out && WIFSIGNALED(status) ? json(WTERMSIG(status)) : json(nullptr)}};
    return true;
}
#endif
} // namespace

bool AgentBroker::openSession(const json& request, json& response, std::string& error) {
#ifdef _WIN32
    error = "Agent tools are not available on Windows yet";
    return false;
#else
    if (!request.is_object() || !request.contains("workspace") || !request["workspace"].is_string() ||
        !request.contains("permissions") || !request["permissions"].is_object()) {
        error = "workspace and permissions are required";
        return false;
    }
    const std::string workspace = request["workspace"];
    std::error_code ec;
    if (workspace.empty() || workspace.size() > 4096 || workspace.find('\0') != std::string::npos || workspace[0] != '/') {
        error = "Workspace must be an absolute directory";
        return false;
    }
    const auto root = std::filesystem::canonical(workspace, ec);
    if (ec || !std::filesystem::is_directory(root, ec) || ec) { error = "Workspace directory is unavailable"; return false; }
    json permissions = json::object();
    for (const auto* name : {"command", "read", "write", "mcp"}) {
        if (request["permissions"].contains(name) && !request["permissions"][name].is_boolean()) {
            error = "Permissions must be boolean"; return false;
        }
        permissions[name] = request["permissions"].value(name, false);
    }
    json servers = json::object(), summaries = json::array();
    if (permissions["mcp"] == true) {
        if (!request.contains("mcp_config") || !request["mcp_config"].is_string() || request["mcp_config"].get<std::string>().empty()) {
            error = "MCP config path is required when MCP is enabled"; return false;
        }
        if (!readMcpConfig(request["mcp_config"].get<std::string>(), servers, summaries, error)) return false;
    }
    const auto id = randomId();
    if (id.empty()) { error = "Cannot create agent session"; return false; }
    std::vector<Session> expired_sessions;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            if (now - it->second.created > std::chrono::hours(8)) {
                expired_sessions.push_back(std::move(it->second));
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
        if (sessions_.size() >= 16) { error = "Too many agent sessions"; return false; }
        Session session;
        session.workspace = root.string(); session.permissions = permissions; session.created = now; session.mcp_servers = servers;
        sessions_[id] = std::move(session);
    }
    response = {{"session_id", id}, {"workspace", root.string()}, {"permissions", permissions}, {"mcp_servers", summaries}};
    return true;
#endif
}

void AgentBroker::closeSession(const std::string& session) {
    Session removed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = sessions_.find(session);
        if (found != sessions_.end()) {
            removed = std::move(found->second);
            sessions_.erase(found);
        }
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second.session == session) it = pending_.erase(it); else ++it;
        }
    }
}

bool AgentBroker::prepare(const std::string& session, const json& request, json& response, std::string& error) {
#ifdef _WIN32
    error = "Agent tools are not available on Windows yet";
    return false;
#else
    if (!request.is_object() || !request.contains("tool") || !request["tool"].is_string() ||
        !request.contains("arguments") || !request["arguments"].is_object()) {
        error = "tool and arguments are required"; return false;
    }
    const std::string tool = request["tool"];
    std::string workspace;
    json permissions, mcp_servers, mcp_tool;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session);
        if (it == sessions_.end() || std::chrono::steady_clock::now() - it->second.created > std::chrono::hours(8)) {
            error = "Agent session is unavailable or expired"; return false;
        }
        workspace = it->second.workspace;
        permissions = it->second.permissions;
        if (tool == "mcp_connect") mcp_servers = it->second.mcp_servers;
        const auto found = it->second.mcp_tools.find(tool);
        if (found != it->second.mcp_tools.end()) mcp_tool = found->second;
    }
    const char* permission = tool == "run_command" ? "command" : tool == "write_file" ? "write" :
                             (tool == "mcp_connect" || tool.compare(0, 4, "mcp_") == 0) ? "mcp" : "read";
    if (!permissions.value(permission, false)) { error = "This tool category is disabled"; return false; }
    auto args = request["arguments"];
    if (tool == "run_command") {
        if (!args.contains("command") || !args["command"].is_string() || args["command"].get<std::string>().empty() ||
            args["command"].get<std::string>().size() > 8192 || args["command"].get<std::string>().find('\0') != std::string::npos ||
            (args.contains("cwd") && (!args["cwd"].is_string() || !relativePath(args["cwd"].get<std::string>(), true))) ||
            (args.contains("timeout_seconds") && (!args["timeout_seconds"].is_number_integer() || args["timeout_seconds"] < 1 || args["timeout_seconds"] > 60))) {
            error = "Invalid command, working directory or timeout"; return false;
        }
        if (!args.contains("cwd")) args["cwd"] = ".";
        if (!args.contains("timeout_seconds")) args["timeout_seconds"] = 30;
    } else if (tool == "list_directory" || tool == "read_file" || tool == "write_file") {
        if (!args.contains("path") || !args["path"].is_string() || !relativePath(args["path"].get<std::string>(), tool == "list_directory")) {
            error = "File path must be relative to the workspace and cannot contain . or .."; return false;
        }
        if (tool == "read_file" && ((args.contains("offset") && (!args["offset"].is_number_integer() || args["offset"] < 0 || args["offset"] > INT_MAX)) ||
            (args.contains("max_bytes") && (!args["max_bytes"].is_number_integer() || args["max_bytes"] < 1 || args["max_bytes"] > 65536)))) {
            error = "Invalid file read offset or size"; return false;
        }
        if (tool == "write_file" && (!args.contains("content") || !args["content"].is_string() || args["content"].get<std::string>().size() > 65536 ||
            (args.contains("overwrite") && !args["overwrite"].is_boolean()))) { error = "Invalid file content or overwrite option"; return false; }
        if (tool == "write_file" && !args.contains("overwrite")) args["overwrite"] = false;
    } else if (tool == "mcp_connect") {
        if (!args.contains("server") || !args["server"].is_string() || !mcp_servers.contains(args["server"].get<std::string>())) {
            error = "Unknown MCP server"; return false;
        }
        const std::string name = args["server"];
        const auto& config = mcp_servers[name];
        if (config.contains("url")) {
            args = {{"server", name}, {"config", {{"url", config["url"]}, {"headers", config["headers"]}}}};
        } else {
            std::filesystem::path cwd = config.value("cwd", "").empty() ? std::filesystem::path(workspace) : std::filesystem::path(config["cwd"].get<std::string>());
            if (cwd.is_relative()) cwd = std::filesystem::path(workspace) / cwd;
            std::error_code ec;
            cwd = std::filesystem::canonical(cwd, ec);
            if (ec || !std::filesystem::is_directory(cwd, ec) || ec) { error = "MCP working directory is unavailable"; return false; }
            args = {{"server", name}, {"config", {{"command", config["command"]}, {"args", config["args"]},
                                                    {"env", config["env"]}, {"cwd", cwd.string()}}}};
        }
    } else if (tool.compare(0, 4, "mcp_") == 0 && mcp_tool.is_object()) {
        if (args.dump().size() > 65536) { error = "MCP arguments exceed 64 KiB"; return false; }
        args = {{"server", mcp_tool["server"]}, {"name", mcp_tool["name"]}, {"input", args}};
    } else { error = "Unknown agent tool"; return false; }
    const auto id = randomId();
    if (id.empty()) { error = "Cannot create approval request"; return false; }
    json action = {{"tool", tool == "mcp_connect" || tool.compare(0, 4, "mcp_") != 0 ? tool : "mcp_call"},
                   {"workspace", workspace}, {"arguments", args}};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (now - it->second.created > std::chrono::minutes(5)) it = pending_.erase(it); else ++it;
        }
        if (pending_.size() >= 64) { error = "Too many pending approvals"; return false; }
        if (sessions_.find(session) == sessions_.end()) { error = "Agent session was closed"; return false; }
        pending_[id] = {action, session, now};
    }
    json display = action;
    if (tool == "mcp_connect") {
        auto& config = display["arguments"]["config"];
        for (const auto* key : {"env", "headers"}) if (config.contains(key)) {
            auto& values = config[key];
            for (auto it = values.begin(); it != values.end(); ++it) it.value() = "[hidden]";
        }
    }
    response = {{"request_id", id}, {"action", display}, {"expires_in_seconds", 300}};
    return true;
#endif
}

bool AgentBroker::execute(const std::string& session, const std::string& id, json& response, std::string& error) {
    json action;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pending_.find(id);
        auto session_it = sessions_.find(session);
        if (it == pending_.end() || it->second.session != session || session_it == sessions_.end()) {
            error = "Approval request not found or already used"; return false;
        }
        action = it->second.action;
        const auto now = std::chrono::steady_clock::now();
        const bool expired = now - it->second.created > std::chrono::minutes(5) ||
                             now - session_it->second.created > std::chrono::hours(8);
        pending_.erase(it);
        if (expired) { error = "Approval request expired"; return false; }
    }
#ifndef _WIN32
    if (action["tool"] == "mcp_connect") {
        json discovered;
        const std::string server = action["arguments"]["server"];
        auto connection = McpConnection::connect(action["arguments"]["config"], action["workspace"], discovered, error);
        if (!connection) return false;
        std::map<std::string, json> registered;
        json definitions = json::array();
        for (const auto& tool : discovered) {
            const std::string name = tool["name"], key = mcpToolId(server, name);
            if (registered.count(key)) { error = "MCP tool identifier collision"; return false; }
            const std::string description = "[MCP " + server + "/" + name + "] " + tool.value("description", "");
            json definition = {{"type", "function"}, {"function", {{"name", key}, {"description", description},
                                                                      {"parameters", tool["inputSchema"]}}}};
            registered[key] = {{"server", server}, {"name", name}};
            definitions.push_back(definition);
        }
        std::shared_ptr<McpConnection> previous;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = sessions_.find(session);
            if (it == sessions_.end()) { error = "Agent session was closed"; return false; }
            for (const auto& item : registered) {
                const auto existing = it->second.mcp_tools.find(item.first);
                if (existing != it->second.mcp_tools.end() && existing->second["server"] != server) {
                    error = "MCP tool identifier collision"; return false;
                }
            }
            auto old = it->second.mcp_connections.find(server);
            if (old != it->second.mcp_connections.end()) previous = std::move(old->second);
            for (auto tool = it->second.mcp_tools.begin(); tool != it->second.mcp_tools.end();) {
                if (tool->second["server"] == server) tool = it->second.mcp_tools.erase(tool); else ++tool;
            }
            it->second.mcp_connections[server] = connection;
            it->second.mcp_tools.insert(registered.begin(), registered.end());
        }
        response = {{"server", server}, {"tools", definitions}};
        return true;
    }
    if (action["tool"] == "mcp_call") {
        std::shared_ptr<McpConnection> connection;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = sessions_.find(session);
            if (it == sessions_.end()) { error = "Agent session was closed"; return false; }
            const auto server = it->second.mcp_connections.find(action["arguments"]["server"].get<std::string>());
            if (server == it->second.mcp_connections.end()) { error = "MCP server is disconnected"; return false; }
            connection = server->second;
        }
        return connection->call(action["arguments"]["name"].get<std::string>(), action["arguments"]["input"], response, error);
    }
    return action["tool"] == "run_command" ? commandAction(action, response, error) : fileAction(action, response, error);
#else
    error = "Agent tools are not available on Windows yet";
    return false;
#endif
}

void AgentBroker::cancel(const std::string& session, const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = pending_.find(id);
    if (it != pending_.end() && it->second.session == session) pending_.erase(it);
}

bool AgentBroker::disconnectMcp(const std::string& session, const std::string& server, std::string& error) {
    std::shared_ptr<McpConnection> connection;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(session);
        if (it == sessions_.end()) { error = "Agent session was closed"; return false; }
        auto found = it->second.mcp_connections.find(server);
        if (found == it->second.mcp_connections.end()) { error = "MCP server is not connected"; return false; }
        connection = std::move(found->second);
        it->second.mcp_connections.erase(found);
        for (auto tool = it->second.mcp_tools.begin(); tool != it->second.mcp_tools.end();) {
            if (tool->second["server"] == server) tool = it->second.mcp_tools.erase(tool); else ++tool;
        }
    }
    return true;
}
} // namespace mnncli
