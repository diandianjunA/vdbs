#pragma once
#include <byteswap.h>
#include <infiniband/verbs.h>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <vector>

/* poll CQ timeout in millisec (2 seconds) */
#define MAX_POLL_CQ_TIMEOUT 2000
#define CQ_SIZE 100

#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#elif __BYTE_ORDER == __BIG_ENDIAN
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t {
    uint64_t addr;   /* Buffer address */
    uint32_t rkey;   /* Remote key */
    uint32_t qp_num; /* QP number */
    uint16_t lid;    /* LID of the IB port */
};

class RdmaDevice {
  public:
    RdmaDevice();
    ~RdmaDevice() {
        if (ctx) {
            ibv_close_device(ctx);
        }
    }
    int get_ib_port() { return ib_port; }
    struct ibv_context *get_ctx() { return ctx; }
    uint32_t get_lid() { return port_attr.lid; }

  private:
    struct ibv_port_attr port_attr; // IB端口属性
    struct ibv_context *ctx;        // IB设备上下文
    int ib_port;                    // IB端口号
};

class QPEntry {
  public:
    QPEntry(struct ibv_pd *pd);
    QPEntry() {}
    QPEntry(const QPEntry &entry) {
        qp = entry.qp;
        cq = entry.cq;
        remote_props = entry.remote_props;
        connected = entry.connected;
    }
    QPEntry &operator=(const QPEntry &entry) {
        qp = entry.qp;
        cq = entry.cq;
        remote_props = entry.remote_props;
        connected = entry.connected;
        return *this;
    }
    QPEntry(QPEntry &&entry) {
        qp = entry.qp;
        cq = entry.cq;
        remote_props = entry.remote_props;
        connected = entry.connected;
    }
    QPEntry &operator=(QPEntry &&entry) {
        qp = entry.qp;
        cq = entry.cq;
        remote_props = entry.remote_props;
        connected = entry.connected;
        return *this;
    }
    ~QPEntry() {
        // if (qp) {
        //     ibv_destroy_qp(qp);
        // }
        // if (cq) {
        //     ibv_destroy_cq(cq);
        // }
    }
    struct ibv_qp *get_qp() { return qp; }
    struct ibv_cq *get_cq() { return cq; }
    struct cm_con_data_t *get_remote_props() { return &remote_props; }
    bool is_connected() { return connected; }
    int connect_qp(struct cm_con_data_t remote_props);
    int modify_qp_to_init();
    int modify_qp_to_rtr();
    int modify_qp_to_rts();
    int poll_completion();
    void set_entry_node(unsigned node) { entry_node = node; }
    unsigned get_entry_node() { return entry_node; }
    int post_receive();
    std::mutex &get_mtx() { return mtx; }

  private:
    struct ibv_qp *qp;                 // QP实例
    struct ibv_cq *cq;                 // 绑定的CQ
    struct cm_con_data_t remote_props; // 远端QP属性
    bool connected;                    // 是否已连接
    unsigned entry_node;
    std::mutex mtx;
};

class RdmaBuffer {
  public:
    RdmaBuffer(void *buf, size_t size);
    ~RdmaBuffer() {
        ibv_dereg_mr(mr);
        ibv_dealloc_pd(pd);
    }
    void *get_buf() { return buf; }
    size_t get_size() { return size; }
    struct ibv_mr *get_mr() { return mr; }
    struct ibv_pd *get_pd() { return pd; }

  private:
    void *buf;         // 缓冲区指针
    size_t size;       // 缓冲区大小
    struct ibv_mr *mr; // 内存区域
    struct ibv_pd *pd; // 保护域
};

extern std::shared_ptr<RdmaDevice> device;