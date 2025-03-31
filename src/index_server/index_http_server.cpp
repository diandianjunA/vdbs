#include "index_http_server.h"
#include "config.h"
#include "logger.h"
#include "rdma_common.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

IndexHttpServer::IndexHttpServer(const std::string &host, int port)
    : host(host), port(port), indexEngine(config->dim) {
    indexEngine.init_gpu();
    server.Post("/search",
                [this](const httplib::Request &req, httplib::Response &res) {
                    searchHandler(req, res);
                });
}

void IndexHttpServer::start() { server.listen(host.c_str(), port); }

void setJsonResponse(const rapidjson::Document &json_response,
                     httplib::Response &res) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json_response.Accept(writer);
    res.set_content(buffer.GetString(), "application/json");
}

void setErrorJsonResponse(httplib::Response &res, int error_code,
                          const std::string &errorMsg) {
    rapidjson::Document json_response;
    json_response.SetObject();
    rapidjson::Document::AllocatorType &allocator =
        json_response.GetAllocator();
    json_response.AddMember("code", error_code, allocator);
    json_response.AddMember("error_msg", rapidjson::StringRef(errorMsg.c_str()),
                            allocator);
    setJsonResponse(json_response, res);
}

void IndexHttpServer::searchHandler(const httplib::Request &req,
                                    httplib::Response &res) {
    // 解析JSON请求
    rapidjson::Document json_request;
    json_request.Parse(req.body.c_str());

    if (!json_request.IsObject()) {
        GlobalLogger->error("invalid JSON request");
        res.status = 400;
        setErrorJsonResponse(res, 400, "invalid JSON request");
        return;
    }

    // 获取查询参数
    std::vector<float> query;

    const rapidjson::Value &objects = json_request["objects"];
    if (!objects.IsArray()) {
        throw std::runtime_error("objects type not match");
    }
    for (auto &obj : objects.GetArray()) {
        if (obj.HasMember("vector") && obj["vector"].IsArray()) {
            const rapidjson::Value &row = obj["vector"];
            std::vector<float> vector;
            for (rapidjson::SizeType j = 0; j < row.Size(); j++) {
                vector.push_back(row[j].GetFloat());
            }
            query.insert(query.end(), vector.begin(), vector.end());
        } else {
            throw std::runtime_error(
                "Missing vectors or id parameter in the request");
        }
    }

    int k = json_request["k"].GetInt();

    std::pair<std::vector<long>, std::vector<float>> results =
        indexEngine.search_vectors(query, k);

    // 将结果转换为JSON
    rapidjson::Document json_response;
    json_response.SetObject();
    rapidjson::Document::AllocatorType &allocator =
        json_response.GetAllocator();

    // 检查是否有有效的搜索结果
    bool valid_results = false;
    rapidjson::Value vectors(rapidjson::kArrayType);
    rapidjson::Value distances(rapidjson::kArrayType);
    for (size_t i = 0; i < results.first.size(); i++) {
        if (results.first[i] != -1) {
            valid_results = true;
            vectors.PushBack(results.first[i], allocator);
            distances.PushBack(results.second[i], allocator);
        }
    }

    if (valid_results) {
        json_response.AddMember("vectors", vectors, allocator);
        json_response.AddMember("distances", distances, allocator);
    }

    // 设置响应
    json_response.AddMember("code", "0", allocator);
    setJsonResponse(json_response, res);
}