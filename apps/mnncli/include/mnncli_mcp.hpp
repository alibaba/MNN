#pragma once

#include <memory>
#include <string>
#include "json.hpp"

namespace mnncli {

class McpConnection {
public:
    static std::shared_ptr<McpConnection> connect(const nlohmann::json& config, const std::string& workspace,
                                                   nlohmann::json& tools, std::string& error);
    ~McpConnection();
    bool call(const std::string& name, const nlohmann::json& arguments, nlohmann::json& result, std::string& error);

private:
    struct Impl;
    explicit McpConnection(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

bool readMcpConfig(const std::string& path, nlohmann::json& servers, nlohmann::json& summaries, std::string& error);

} // namespace mnncli
