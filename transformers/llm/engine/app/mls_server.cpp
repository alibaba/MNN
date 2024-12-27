//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mls_server.hpp"
#include <iostream>
#include "httplib.h"

namespace mls {

void MlsServer::Start() {
    // Create a server instance
    httplib::Server svr;

    // Define a route for the GET request on "/"
    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("Hello, World!", "text/plain");
    });

    // Define a route for the GET request on "/json"
    svr.Get("/json", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content(R"({"message": "Hello, JSON!"})", "application/json");
    });

    // Start the server on port 8080
    std::cout << "Starting server on http://localhost:8080\n";
    if (!svr.listen("0.0.0.0", 8080)) {
        std::cerr << "Error: Could not start server.\n";
    }
}
}