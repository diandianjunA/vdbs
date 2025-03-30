#pragma once
#include "rdma_common.h"

class RdmaPool {
  public:
    RdmaPool(void *buf, size_t memory_region_size);
    ~RdmaPool() {}
    RdmaBuffer *get_buffer() { return &buffer; }
    int get_size() { return size; }
    QPEntry* get_qp(int qp_id) { return &qps[qp_id]; }
    QPEntry* add_qp();

  private:
    int size;                 // QP条目数量
    std::vector<QPEntry> qps; // QP实例列表
    RdmaBuffer buffer;        // RDMA缓冲区
};

extern std::shared_ptr<RdmaPool> pool;