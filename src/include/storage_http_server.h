#pragma once

#include "httplib.h"
#include "rapidjson/document.h"
#include <string>
#include "rdma_pool.h"
#include "hnswlib/hnswlib.h"

class StorageHttpServer {
  public:
    enum class CheckType {
        CONNECT_RDMA,
    };

    StorageHttpServer(const std::string &host, int port);
    void start();
    void* get_buffer() { return index->data_level0_memory_; }
    size_t get_size() { return index->max_elements_ * index->size_data_per_element_; }

  private:
    void connectRdma(const httplib::Request &req, httplib::Response &res);
    void insertHandler(const httplib::Request& req, httplib::Response& res);

    httplib::Server server;
    std::string host;
    int port;
    hnswlib::SpaceInterface<float>* space;
    hnswlib::HierarchicalNSW<float>* index;
};