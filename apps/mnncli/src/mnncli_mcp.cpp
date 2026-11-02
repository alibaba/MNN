#include "mnncli_mcp.hpp"
#include "httplib.h"

#include <chrono>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace mnncli {
namespace {
using json = nlohmann::json;

bool safeString(const json& value, size_t limit) {
    return value.is_string() && value.get<std::string>().size() <= limit &&
           value.get<std::string>().find('\0') == std::string::npos;
}

bool parseHttpUrl(const std::string& url, std::string& origin, std::string& path) {
    static const std::regex pattern("^(https?)://([^/?#]+)(/[^#]*)?$");
    std::smatch match;
    if (url.size() > 4096 || !std::regex_match(url, match, pattern) || match[2].str().find('@') != std::string::npos ||
        url.find_first_of(" \t\r\n") != std::string::npos) return false;
    const auto authority = match[2].str();
    if (match[1] == "http" && authority != "localhost" && authority.compare(0, 10, "localhost:") != 0 &&
        authority != "127.0.0.1" && authority.compare(0, 10, "127.0.0.1:") != 0 &&
        authority != "[::1]" && authority.compare(0, 6, "[::1]:") != 0) return false;
    origin = match[1].str() + "://" + authority;
    path = match[3].matched ? match[3].str() : "/";
    return true;
}

std::string resolveExecutable(const std::string& command, const std::string& path) {
#ifndef _WIN32
    if (command.find('/') != std::string::npos) return access(command.c_str(), X_OK) == 0 ? command : "";
    const auto paths = path + ":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin";
    size_t start = 0;
    while (start <= paths.size()) {
        const auto end = paths.find(':', start);
        const auto part = paths.substr(start, end == std::string::npos ? end : end - start);
        const auto candidate = part + "/" + command;
        if (!part.empty() && access(candidate.c_str(), X_OK) == 0) return candidate;
        if (end == std::string::npos) break;
        start = end + 1;
    }
#endif
    return "";
}
} // namespace

bool readMcpConfig(const std::string& path, json& servers, json& summaries, std::string& error) {
    std::error_code ec;
    if (path.empty() || path[0] != '/' || path.find('\0') != std::string::npos || path.size() > 4096 ||
        !std::filesystem::is_regular_file(path, ec) || ec || std::filesystem::file_size(path, ec) > 262144 || ec) {
        error = "MCP config must be an absolute path to a JSON file under 256 KiB"; return false;
    }
    std::ifstream input(path);
    if (!input) { error = "Cannot read MCP config"; return false; }
    std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto config = json::parse(content, nullptr, false);
    if (!config.is_object() || !config.contains("mcpServers") || !config["mcpServers"].is_object() || config["mcpServers"].size() > 32) {
        error = "MCP config requires an mcpServers object with at most 32 servers"; return false;
    }
    static const std::regex identifier("^[A-Za-z0-9_-]{1,48}$"), env_name("^[A-Za-z_][A-Za-z0-9_]*$");
    servers = json::object(); summaries = json::array();
    for (auto it = config["mcpServers"].begin(); it != config["mcpServers"].end(); ++it) {
        const auto name = it.key();
        auto server = it.value();
        if (!std::regex_match(name, identifier) || !server.is_object()) { error = "Invalid MCP server name or configuration"; return false; }
        if (server.contains("disabled") && !server["disabled"].is_boolean()) { error = "Invalid MCP disabled flag"; return false; }
        if (server.value("disabled", false)) continue;
        if (server.contains("url")) {
            std::string origin, endpoint;
            if (!safeString(server["url"], 4096) || server.contains("command") ||
                !parseHttpUrl(server["url"].get<std::string>(), origin, endpoint)) {
                error = "MCP HTTP requires an HTTPS URL or a loopback HTTP URL"; return false;
            }
            if (!server.contains("headers")) server["headers"] = json::object();
            if (!server["headers"].is_object() || server["headers"].size() > 32) { error = "Invalid MCP HTTP headers"; return false; }
            static const std::regex header_name("^[A-Za-z0-9-]{1,128}$");
            json keys = json::array();
            for (auto header = server["headers"].begin(); header != server["headers"].end(); ++header) {
                auto lower = header.key();
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                if (!std::regex_match(header.key(), header_name) || !safeString(header.value(), 8192) ||
                    header.value().get<std::string>().find_first_of("\r\n") != std::string::npos ||
                    lower == "host" || lower == "content-type" || lower == "accept" || lower == "origin" ||
                    lower == "content-length" || lower == "transfer-encoding" || lower == "connection" ||
                    lower == "mcp-session-id" || lower == "mcp-protocol-version") {
                    error = "Invalid or reserved MCP HTTP header"; return false;
                }
                keys.push_back(header.key());
            }
            servers[name] = server;
            summaries.push_back({{"name", name}, {"transport", "streamable-http"}, {"url", server["url"]}, {"header_keys", keys}});
            continue;
        }
        if (!server.contains("command") || !safeString(server["command"], 4096) || server["command"].get<std::string>().empty()) {
            error = "MCP server requires a stdio command or Streamable HTTP URL"; return false;
        }
        if (!server.contains("args")) server["args"] = json::array();
        if (!server["args"].is_array() || server["args"].size() > 64) { error = "Invalid MCP server arguments"; return false; }
        for (const auto& arg : server["args"]) if (!safeString(arg, 8192)) { error = "Invalid MCP server argument"; return false; }
        if (!server.contains("env")) server["env"] = json::object();
        if (!server["env"].is_object() || server["env"].size() > 64) { error = "Invalid MCP server environment"; return false; }
        json keys = json::array();
        for (auto env = server["env"].begin(); env != server["env"].end(); ++env) {
            if (!std::regex_match(env.key(), env_name) || !safeString(env.value(), 8192)) { error = "Invalid MCP environment variable"; return false; }
            keys.push_back(env.key());
        }
        if (server.contains("cwd") && !safeString(server["cwd"], 4096)) { error = "Invalid MCP working directory"; return false; }
        servers[name] = server;
        summaries.push_back({{"name", name}, {"transport", "stdio"}, {"command", server["command"]},
                             {"args", server["args"]}, {"env_keys", keys}, {"cwd", server.value("cwd", "")}});
    }
    return true;
}

struct McpConnection::Impl {
#ifndef _WIN32
    pid_t pid = -1;
    int input = -1, output = -1;
    std::string buffered;
    int64_t next_id = 1;
    std::mutex mutex;
    json tools = json::array();
    bool http = false;
    std::string http_origin, http_endpoint, http_session, protocol_version;
    json http_headers = json::object();

    httplib::Headers headers() const {
        httplib::Headers result = {{"Accept", "application/json, text/event-stream"}};
        for (auto it = http_headers.begin(); it != http_headers.end(); ++it) result.emplace(it.key(), it.value().get<std::string>());
        if (!http_session.empty()) result.emplace("Mcp-Session-Id", http_session);
        if (!protocol_version.empty()) result.emplace("MCP-Protocol-Version", protocol_version);
        return result;
    }

    void replyHttp(const json& message) {
        httplib::Client client(http_origin);
        client.set_connection_timeout(5); client.set_read_timeout(5); client.set_write_timeout(5);
        client.set_follow_location(false);
        client.Post(http_endpoint, headers(), message.dump(), "application/json");
    }

    bool httpExchange(const json& message, bool expect_response, const json& expected_id, json& result, std::string& error) {
        httplib::Client client(http_origin);
        if (!client.is_valid()) { error = "Invalid MCP HTTP endpoint"; return false; }
        client.set_connection_timeout(10); client.set_read_timeout(30); client.set_write_timeout(10);
        client.set_follow_location(false);
        httplib::Request request;
        request.method = "POST"; request.path = http_endpoint; request.body = message.dump(); request.headers = headers();
        request.set_header("Content-Type", "application/json");
        std::string content_type, body, pending, event_data, session_header;
        int status = 0; bool found = false, oversized = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        request.response_handler = [&](const httplib::Response& response) {
            status = response.status; content_type = response.get_header_value("Content-Type");
            session_header = response.get_header_value("Mcp-Session-Id");
            for (const unsigned char c : session_header) if (c < 0x21 || c > 0x7e) { error = "Invalid MCP HTTP session ID"; return false; }
            if (session_header.size() > 256 || (!http_session.empty() && !session_header.empty() && session_header != http_session)) {
                error = "Invalid MCP HTTP session ID"; return false;
            }
            if (http_session.empty() && !session_header.empty()) http_session = session_header;
            return true;
        };
        request.content_receiver = [&](const char* data, size_t length, size_t, size_t) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            if (content_type.find("text/event-stream") == std::string::npos) {
                body.append(data, length); if (body.size() > 1024 * 1024) { oversized = true; return false; } return true;
            }
            pending.append(data, length);
            if (pending.size() + event_data.size() > 1024 * 1024) { oversized = true; return false; }
            size_t newline;
            while ((newline = pending.find('\n')) != std::string::npos) {
                auto line = pending.substr(0, newline); pending.erase(0, newline + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) {
                    if (event_data.empty()) continue;
                    const auto event = json::parse(event_data, nullptr, false); event_data.clear();
                    if (!event.is_object()) { error = "Invalid MCP SSE message"; return false; }
                    if (event.contains("method") && event.contains("id")) {
                        json reply = {{"jsonrpc", "2.0"}, {"id", event["id"]}};
                        if (event["method"] == "ping") reply["result"] = json::object();
                        else reply["error"] = {{"code", -32601}, {"message", "Client method not supported"}};
                        replyHttp(reply);
                    } else if (expect_response && event.contains("id") && event["id"] == expected_id) {
                        result = event; found = true; return false;
                    }
                } else if (line.compare(0, 5, "data:") == 0) {
                    if (!event_data.empty()) event_data += '\n';
                    event_data += line.size() > 5 && line[5] == ' ' ? line.substr(6) : line.substr(5);
                }
            }
            return true;
        };
        httplib::Response response;
        httplib::Error transport_error;
        const bool ok = client.send(request, response, transport_error);
        if (oversized) { error = "MCP HTTP response exceeds 1 MiB"; return false; }
        if (status < 200 || status >= 300) { error = "MCP HTTP request failed with status " + std::to_string(status); return false; }
        if (!error.empty()) return false;
        if (!expect_response) { if (!ok || status != 202) { error = "MCP notification was not accepted"; return false; } return true; }
        if (!found && content_type.find("application/json") != std::string::npos) {
            if (!ok) { error = "MCP HTTP response failed"; return false; }
            result = json::parse(body, nullptr, false); found = result.is_object();
        }
        if (!found || !result.contains("jsonrpc") || result["jsonrpc"] != "2.0" || !result.contains("id") || result["id"] != expected_id) {
            if (error.empty()) error = "Invalid or incomplete MCP HTTP response"; return false;
        }
        if (result.contains("error")) { error = "MCP error: " + result["error"].dump(); return false; }
        if (!result.contains("result") || !result["result"].is_object()) { error = "Invalid MCP result"; return false; }
        json payload = result["result"];
        result = std::move(payload);
        return true;
    }

    ~Impl() {
        if (http && !http_session.empty()) {
            httplib::Client client(http_origin);
            client.set_connection_timeout(2); client.set_read_timeout(2); client.set_write_timeout(2);
            client.set_follow_location(false); client.Delete(http_endpoint, headers());
        }
        if (input >= 0) close(input);
        if (output >= 0) close(output);
        if (pid > 0) {
            int status = 0;
            if (waitpid(pid, &status, WNOHANG) == 0) {
                kill(-pid, SIGTERM);
                for (int i = 0; i < 10; ++i) {
                    if (waitpid(pid, &status, WNOHANG) == pid) { pid = -1; break; }
                    struct pollfd p = {-1, 0, 0}; poll(&p, 0, 50);
                }
                if (pid > 0) { kill(-pid, SIGKILL); waitpid(pid, &status, 0); }
            }
        }
    }

    bool send(const json& message, std::chrono::steady_clock::time_point deadline, std::string& error) {
        const std::string data = message.dump() + "\n";
        size_t offset = 0;
        while (offset < data.size()) {
            if (std::chrono::steady_clock::now() >= deadline) { error = "MCP write timed out"; return false; }
            const auto n = write(input, data.data() + offset, data.size() - offset);
            if (n > 0) { offset += static_cast<size_t>(n); continue; }
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) { struct pollfd p = {input, POLLOUT, 0}; poll(&p, 1, 100); continue; }
            error = "MCP server input closed"; return false;
        }
        return true;
    }

    bool request(const std::string& method, const json& params, json& result, std::string& error) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        const auto id = next_id++;
        if (http) return httpExchange({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}}, true, id, result, error);
        if (!send({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}}, deadline, error)) return false;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto newline = buffered.find('\n');
            if (newline == std::string::npos) {
                struct pollfd p = {output, POLLIN, 0};
                const int ready = poll(&p, 1, 100);
                if (ready < 0 && errno != EINTR) { error = "MCP read failed"; return false; }
                if (ready <= 0) continue;
                char bytes[8192]; const auto n = read(output, bytes, sizeof(bytes));
                if (n > 0) { buffered.append(bytes, static_cast<size_t>(n)); if (buffered.size() > 1024 * 1024) { error = "MCP response exceeds 1 MiB"; return false; } continue; }
                if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
                error = "MCP server output closed"; return false;
            }
            const auto line = buffered.substr(0, newline); buffered.erase(0, newline + 1);
            auto message = json::parse(line, nullptr, false);
            if (!message.is_object() || !message.contains("jsonrpc") || !message["jsonrpc"].is_string() || message["jsonrpc"] != "2.0") { error = "Invalid MCP JSON-RPC message"; return false; }
            if (message.contains("method") && message["method"].is_string()) {
                if (message.contains("id")) {
                    json reply = {{"jsonrpc", "2.0"}, {"id", message["id"]}};
                    if (message["method"] == "ping") reply["result"] = json::object();
                    else reply["error"] = {{"code", -32601}, {"message", "Client method not supported"}};
                    if (!send(reply, deadline, error)) return false;
                }
                continue;
            }
            if (!message.contains("id") || message["id"] != id) { error = "Unexpected MCP response ID"; return false; }
            if (message.contains("error")) { error = "MCP error: " + message["error"].dump(); return false; }
            if (!message.contains("result") || !message["result"].is_object()) { error = "Invalid MCP result"; return false; }
            result = message["result"];
            return true;
        }
        error = "MCP request timed out";
        return false;
    }

    bool initialized(std::string& error) {
        const json message = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
        if (http) { json ignored; return httpExchange(message, false, nullptr, ignored, error); }
        return send(message, std::chrono::steady_clock::now() + std::chrono::seconds(5), error);
    }

    bool list(json& list, std::string& error) {
        list = json::array(); std::set<std::string> names, cursors; std::string cursor; size_t total_size = 0;
        for (int page = 0; page < 8; ++page) {
            json params = json::object(), data;
            if (!cursor.empty()) params["cursor"] = cursor;
            if (!request("tools/list", params, data, error)) return false;
            if (!data.contains("tools") || !data["tools"].is_array()) { error = "Invalid MCP tool list"; return false; }
            for (const auto& tool : data["tools"]) {
                if (!tool.is_object() || !tool.contains("name") || !safeString(tool["name"], 128) || tool["name"].get<std::string>().empty() ||
                    !tool.contains("inputSchema") || !tool["inputSchema"].is_object() || !tool["inputSchema"].contains("type") ||
                    !tool["inputSchema"]["type"].is_string() || tool["inputSchema"]["type"] != "object" ||
                    (tool.contains("description") && !safeString(tool["description"], 4096)) || tool.dump().size() > 65536 ||
                    !names.insert(tool["name"].get<std::string>()).second || list.size() >= 100) {
                    error = "Invalid, duplicate or excessive MCP tools"; return false;
                }
                total_size += tool.dump().size();
                if (total_size > 262144) { error = "MCP tool list exceeds 256 KiB"; return false; }
                list.push_back(tool);
            }
            if (!data.contains("nextCursor") || data["nextCursor"].is_null()) { tools = list; return true; }
            if (!safeString(data["nextCursor"], 1024) || data["nextCursor"].get<std::string>().empty()) { error = "Invalid MCP cursor"; return false; }
            cursor = data["nextCursor"].get<std::string>();
            if (!cursors.insert(cursor).second) { error = "Repeated MCP cursor"; return false; }
        }
        error = "MCP tool list exceeds the page limit";
        return false;
    }
#endif
};

McpConnection::McpConnection(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
McpConnection::~McpConnection() = default;

std::shared_ptr<McpConnection> McpConnection::connect(const json& config, const std::string& workspace,
                                                       json& tools, std::string& error) {
#ifdef _WIN32
    error = "MCP stdio is not available on Windows yet"; return nullptr;
#else
    auto impl = std::unique_ptr<Impl>(new Impl);
    if (config.contains("url")) {
        impl->http = true;
        if (!parseHttpUrl(config["url"].get<std::string>(), impl->http_origin, impl->http_endpoint)) {
            error = "Invalid MCP HTTP URL"; return nullptr;
        }
        impl->http_headers = config["headers"];
    } else {
    std::map<std::string, std::string> env;
    for (char** entry = environ; *entry; ++entry) { std::string item(*entry); const auto equal = item.find('='); if (equal != std::string::npos) env[item.substr(0, equal)] = item.substr(equal + 1); }
    for (auto it = config["env"].begin(); it != config["env"].end(); ++it) env[it.key()] = it.value().get<std::string>();
    std::error_code ec;
    const auto cwd = config.value("cwd", "").empty() ? workspace : config["cwd"].get<std::string>();
    const auto directory = std::filesystem::canonical(cwd, ec);
    if (ec || !std::filesystem::is_directory(directory, ec) || ec) { error = "MCP working directory is unavailable"; return nullptr; }
    const std::string command = config["command"];
    const auto executable = resolveExecutable(command.find('/') != std::string::npos && command[0] != '/' ?
                                               (directory / command).string() : command, env["PATH"]);
    if (executable.empty()) { error = "MCP server executable not found"; return nullptr; }
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0) { error = "Cannot create MCP input pipe"; return nullptr; }
    if (pipe(out_pipe) != 0) { close(in_pipe[0]); close(in_pipe[1]); error = "Cannot create MCP output pipe"; return nullptr; }
    posix_spawn_file_actions_t actions; posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    for (const int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]}) posix_spawn_file_actions_addclose(&actions, fd);
    std::vector<std::string> args = {"/bin/sh", "-c", "cd \"$1\" && shift && exec \"$@\"", "mnncli-mcp", directory.string(), executable};
    for (const auto& arg : config["args"]) args.push_back(arg.get<std::string>());
    std::vector<char*> argv; for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str())); argv.push_back(nullptr);
    std::vector<std::string> env_values; for (const auto& item : env) env_values.push_back(item.first + "=" + item.second);
    std::vector<char*> envp; for (const auto& item : env_values) envp.push_back(const_cast<char*>(item.c_str())); envp.push_back(nullptr);
    posix_spawnattr_t attr; posix_spawnattr_init(&attr); posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP); posix_spawnattr_setpgroup(&attr, 0);
    const int rc = posix_spawn(&impl->pid, "/bin/sh", &actions, &attr, argv.data(), envp.data());
    posix_spawnattr_destroy(&attr); posix_spawn_file_actions_destroy(&actions);
    close(in_pipe[0]); close(out_pipe[1]);
    if (rc != 0) { close(in_pipe[1]); close(out_pipe[0]); impl->pid = -1; error = "Cannot start MCP server"; return nullptr; }
    impl->input = in_pipe[1]; impl->output = out_pipe[0];
    fcntl(impl->input, F_SETFL, fcntl(impl->input, F_GETFL, 0) | O_NONBLOCK);
    fcntl(impl->output, F_SETFL, fcntl(impl->output, F_GETFL, 0) | O_NONBLOCK);
    fcntl(impl->input, F_SETFD, FD_CLOEXEC); fcntl(impl->output, F_SETFD, FD_CLOEXEC);
    }
    json initialized;
    if (!impl->request("initialize", {{"protocolVersion", "2025-06-18"}, {"capabilities", json::object()},
                                      {"clientInfo", {{"name", "mnncli"}, {"version", "1.0.0"}}}}, initialized, error)) return nullptr;
    if (!initialized.contains("protocolVersion") || !initialized["protocolVersion"].is_string()) { error = "Invalid MCP protocol version"; return nullptr; }
    const auto version = initialized["protocolVersion"].get<std::string>();
    if (version != "2025-06-18" && version != "2025-03-26" && version != "2024-11-05") { error = "Unsupported MCP protocol version"; return nullptr; }
    if (impl->http && version == "2024-11-05") { error = "Legacy HTTP+SSE MCP is not supported"; return nullptr; }
    impl->protocol_version = version;
    if (!initialized.contains("capabilities") || !initialized["capabilities"].is_object() ||
        !initialized["capabilities"].contains("tools")) { error = "MCP server does not advertise tools"; return nullptr; }
    if (!impl->initialized(error) || !impl->list(tools, error)) return nullptr;
    return std::shared_ptr<McpConnection>(new McpConnection(std::move(impl)));
#endif
}

bool McpConnection::call(const std::string& name, const json& arguments, json& result, std::string& error) {
#ifdef _WIN32
    error = "MCP is not available"; return false;
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    bool found = false; for (const auto& tool : impl_->tools) found |= tool["name"] == name;
    if (!found || !arguments.is_object() || arguments.dump().size() > 65536) { error = "Unknown MCP tool or invalid arguments"; return false; }
    return impl_->request("tools/call", {{"name", name}, {"arguments", arguments}}, result, error);
#endif
}
} // namespace mnncli
