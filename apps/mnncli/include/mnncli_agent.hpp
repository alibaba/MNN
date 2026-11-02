#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "json.hpp"

namespace mnncli {
class McpConnection;

class AgentBroker {
public:
    bool openSession(const nlohmann::json& request, nlohmann::json& response, std::string& error);
    void closeSession(const std::string& session);
    bool prepare(const std::string& session, const nlohmann::json& request, nlohmann::json& response, std::string& error);
    bool execute(const std::string& session, const std::string& id, nlohmann::json& response, std::string& error);
    void cancel(const std::string& session, const std::string& id);
    bool disconnectMcp(const std::string& session, const std::string& server, std::string& error);

private:
    struct Pending {
        nlohmann::json action;
        std::string session;
        std::chrono::steady_clock::time_point created;
    };
    struct Session {
        std::string workspace;
        nlohmann::json permissions;
        std::chrono::steady_clock::time_point created;
        nlohmann::json mcp_servers;
        std::map<std::string, std::shared_ptr<McpConnection>> mcp_connections;
        std::map<std::string, nlohmann::json> mcp_tools;
    };
    std::mutex mutex_;
    std::map<std::string, Pending> pending_;
    std::map<std::string, Session> sessions_;
};

} // namespace mnncli
