#include "logger.h"
#include "rdma_common.h"

std::shared_ptr<RdmaDevice> device;

RdmaDevice::RdmaDevice() {
    struct ibv_device **dev_list = NULL;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_device *ib_dev = NULL;
    ib_port = 1;

    int num_devices;
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        GlobalLogger->error("failed to get IB devices list");
    }
    if (!num_devices) {
        GlobalLogger->error("found {} device(s)", num_devices);
    }
    ib_dev = dev_list[0];
    if (!ib_dev) {
        GlobalLogger->error("failed to get IB device");
    }
    ctx = ibv_open_device(ib_dev);
    if (!ctx) {
        GlobalLogger->error("failed to open device");
    }
    const char* dev_name = ibv_get_device_name(ib_dev);
    if (!dev_name) {
        GlobalLogger->error("failed to get device name");
    }
    ibv_free_device_list(dev_list);
    if (ibv_query_port(ctx, ib_port, &port_attr)) {
        GlobalLogger->error("ibv_query_port on port {} failed", ib_port);
    }
    GlobalLogger->info("HCA {} found", dev_name);
}

RdmaBuffer::RdmaBuffer(void *buf, size_t size) : buf(buf), size(size) {
    pd = ibv_alloc_pd(device->get_ctx());
    GlobalLogger->info("Allocated protection domain");
    if (!pd) {
        GlobalLogger->error("failed to allocate protection domain");
        exit(1);
    }
    mr = ibv_reg_mr(pd, buf, size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE);
    GlobalLogger->info("Registered memory region");
    if (!mr) {
        GlobalLogger->error("failed to register memory region");
        exit(1);
    }
}

QPEntry::QPEntry(struct ibv_pd *pd) {
    cq = ibv_create_cq(device->get_ctx(), CQ_SIZE, NULL, NULL, 0);
    if (!cq) {
        GlobalLogger->error("failed to create CQ");
    }
    struct ibv_qp_init_attr qp_init_attr = {
        .send_cq = cq, // 发送完成队列（CQ）指针
        .recv_cq = cq, // 接收完成队列（CQ）指针
        .cap =
            {
                // QP容量限制
                .max_send_wr = 1024, // 发送队列最大未完成工作请求（WR）
                .max_recv_wr = 1024, // 接收队列最大未完成WR
                .max_send_sge = 32,  // 每个发送WR的分散/聚合元素（SGE）数
                .max_recv_sge = 32   // 每个接收WR的SGE数
            },
        .qp_type = IBV_QPT_RC, // QP类型（可靠连接）
        .sq_sig_all = 0,       // 信号模式（0=仅显式请求触发CQ事件）
    };
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        GlobalLogger->error("failed to create QP");
    }
    // int rc = 0;
    // rc = modify_qp_to_init();
    // GlobalLogger->info("Modified QP state to INIT");
    // if (rc) {
    //     GlobalLogger->error("failed to modify QP state to INIT");
    // }
}

int QPEntry::modify_qp_to_init() {
    struct ibv_qp_attr attr;
    int flags;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = device->get_ib_port();
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE;
    flags =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc) {
        GlobalLogger->error("failed to modify QP state to INIT");
    }
    return rc;
}

int QPEntry::modify_qp_to_rtr() {
    struct ibv_qp_attr attr;
    int flags;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_256;
    attr.dest_qp_num = remote_props.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote_props.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = device->get_ib_port();
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc) {
        GlobalLogger->error("failed to modify QP state to RTR");
    }
    return rc;
}

int QPEntry::modify_qp_to_rts() {
    struct ibv_qp_attr attr;
    int flags;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 0x12;
    attr.retry_cnt = 6;
    attr.rnr_retry = 0;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc) {
        GlobalLogger->error("failed to modify QP state to RTS");
    }
    return rc;
}

int QPEntry::connect_qp(struct cm_con_data_t remote_props) {
    this->remote_props = remote_props;
    GlobalLogger->info("Connecting QP to remote QP number {}", remote_props.qp_num);
    int rc = 0;
    rc = modify_qp_to_init();
    GlobalLogger->info("Modified QP state to INIT");
    if (rc) {
        GlobalLogger->error("failed to modify QP state to INIT");
        return rc;
    }
    rc = modify_qp_to_rtr();
    GlobalLogger->info("Modified QP state to RTR");
    if (rc) {
        GlobalLogger->error("failed to modify QP state to RTR");
        return rc;
    }
    rc = modify_qp_to_rts();
    GlobalLogger->info("Modified QP state to RTS");
    if (rc) {
        GlobalLogger->error("failed to modify QP state to RTS");
        return rc;
    }
    connected = true;
    return rc;
}

int QPEntry::poll_completion() {
    struct ibv_wc wc;
    int rc;
    do {
        rc = ibv_poll_cq(cq, 1, &wc);
        if (rc < 0) {
            GlobalLogger->error("failed to poll CQ");
            return rc;
        }
    } while (rc == 0);
    if (wc.status != IBV_WC_SUCCESS) {
        GlobalLogger->error("failed status {}", ibv_wc_status_str(wc.status));
        return -1;
    }
    return 0;
}

// int QPEntry::post_receive() {
//     struct ibv_recv_wr rr;
//     struct ibv_sge sge;
//     struct ibv_recv_wr *bad_wr;
//     int rc;
//     /* prepare the scatter/gather entry */
//     memset(&sge, 0, sizeof(sge));
//     sge.addr = (uintptr_t)buf;
//     sge.length = memory_size;
//     sge.lkey = res->mr->lkey;
//     /* prepare the receive work request */
//     memset(&rr, 0, sizeof(rr));
//     rr.next = NULL;
//     rr.wr_id = 0;
//     rr.sg_list = &sge;
//     rr.num_sge = 1;
//     /* post the Receive Request to the RQ */
//     rc = ibv_post_recv(res->qp, &rr, &bad_wr);
//     if (rc)
//         fprintf(stderr, "failed to post RR\n");
//     else
//         fprintf(stdout, "Receive Request was posted\n");
//     return rc;
// }