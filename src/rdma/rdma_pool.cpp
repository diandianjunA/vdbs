#include "rdma_pool.h"
#include "logger.h"

std::shared_ptr<RdmaPool> pool;

RdmaPool::RdmaPool(void *buf, size_t memory_region_size): buffer(buf, memory_region_size) {
    GlobalLogger->info("QP pool initialized");
}

QPEntry* RdmaPool::add_qp() {
    QPEntry entry(buffer.get_pd());
    GlobalLogger->debug("QP added");
    qps.push_back(entry);
    return &qps.back();
}