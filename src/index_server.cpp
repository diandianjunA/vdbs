#include "index_http_server.h"
#include "index_http_client.h"
#include "rdma_pool.h"
#include "config.h"
#include "logger.h"

int main() {

    init_global_logger();
    set_log_level(spdlog::level::debug);
    GlobalLogger->info("Global logger initialized");

    device = std::make_shared<RdmaDevice>();
    int num_data = 10000;
    int dim = 128;
    int max_m = 32;
    int ef_construction = 200;
    config->dim = dim;
    config->num_data = num_data;
    config->max_m = max_m;
    config->ef_construction = ef_construction;
    config->shard_num = 1;
    config->memory_region_size = 100 * 1024 * 1024;
    config->calculate();
    std::string etcdEndpoints = "http://127.0.0.1:2379";
    
    etcdClient_ = std::make_unique<etcd::Client>(etcdEndpoints);
    IndexHttpServer indexHttpServer("0.0.0.0", 4000);
    GlobalLogger->info("Index server started");
    indexHttpServer.start();

    return 0;
}