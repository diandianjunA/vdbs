#pragma once

#include "httplib.h"
#include "rapidjson/document.h"
#include <string>
#include "index_engine.h"

class IndexHttpServer {
public:
    enum class CheckType {
        SEARCH
    };

    IndexHttpServer(const std::string& host, int port);
    void start();

private:
    void searchHandler(const httplib::Request& req, httplib::Response& res);

    httplib::Server server;
    std::string host;
    int port;
    IndexEngine indexEngine;
};