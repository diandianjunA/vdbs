#include "index_http_client.h"
#include "logger.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

std::unique_ptr<etcd::Client> etcdClient_;

std::pair<struct cm_con_data_t, unsigned int>
IndexHttpClient::connectRdma(uint64_t addr, uint32_t rkey, uint32_t qp_num,
                             uint16_t lid) {
    client.set_read_timeout(30, 0);
    client.set_connection_timeout(10, 0);
    rapidjson::Document json_request;
    json_request.SetObject();
    rapidjson::Document::AllocatorType &allocator = json_request.GetAllocator();
    json_request.AddMember("addr", addr, allocator);
    json_request.AddMember("rkey", rkey, allocator);
    json_request.AddMember("qp_num", qp_num, allocator);
    json_request.AddMember("lid", lid, allocator);

    GlobalLogger->debug("Sending RDMA connection request: addr = {:#x}, rkey = "
                        "{:#x}, qp_num = {:#x}, lid = {:#x}",
                        addr, rkey, qp_num, lid);

    httplib::Headers headers = {
        {"Content-Type", "application/json"},
    };
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json_request.Accept(writer);
    auto res = client.Post("/connectRdma", headers, buffer.GetString(),
                           "application/json");
    if (!res) {
        throw std::runtime_error("Failed to connect to server");
    }
    if (res->status != 200) {
        throw std::runtime_error("Failed to connect to server");
    }

    rapidjson::Document json_response;
    json_response.Parse(res->body.c_str());
    if (!json_response.IsObject()) {
        throw std::runtime_error("Invalid JSON response");
    }

    cm_con_data_t remote_props;
    remote_props.addr = json_response["addr"].GetUint64();
    remote_props.rkey = json_response["rkey"].GetUint();
    remote_props.qp_num = json_response["qp_num"].GetUint();
    remote_props.lid = json_response["lid"].GetUint();
    unsigned int entry_id = json_response["entry_node"].GetUint();

    GlobalLogger->debug("Received RDMA connection response: addr = {:#x}, rkey "
                        "= {:#x}, qp_num = {:#x}, lid = {:#x}",
                        remote_props.addr, remote_props.rkey,
                        remote_props.qp_num, remote_props.lid);
    GlobalLogger->debug("Entry node: {}", entry_id);

    return std::make_pair(remote_props, entry_id);
}
