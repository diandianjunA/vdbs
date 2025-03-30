#pragma once
#include "rdma_common.h"

class RdmaPools {
  public:
    RdmaPools(int size, void *buf, size_t memory_region_size);
    ~RdmaPools() {}
    RdmaBuffer *get_buffer() { return &buffer; }
    int get_size() { return size; }
    int rdma_read(int qp_id, char *local_addr, size_t size, size_t offset);
    QPEntry *get_qp(int qp_id) { return &qps[qp_id]; }
    QPEntry *add_qp();
    int get_entry_node(int qp_id);
    void connect_qp(int qp_id);

  private:
    int size;                 // QP条目数量
    std::vector<QPEntry> qps; // QP实例列表
    RdmaBuffer buffer;        // RDMA缓冲区
};

extern std::shared_ptr<RdmaPools> pool;