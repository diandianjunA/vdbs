#include "index_engine.h"
#include "config.h"
#include "gpu_memory_manager.h"
#include "logger.h"
#include "rdma_pools.h"
#include "search_kernel.cuh"
#include <functional>
#include <queue>
#include <thread>

IndexEngine::IndexEngine(int dim) : dim(dim) {}

void IndexEngine::init_gpu() {
    gpu_memory_manager = std::make_shared<GPUMemoryManager>(
        config->shard_num, config->size_data_per_element,
        config->size_per_shard);
    gpu_memory_manager.get()->allocate_memory(config->memory_region_size);
    pool = std::make_shared<RdmaPools>(config->shard_num,
                                       gpu_memory_manager.get()->get_buffer(),
                                       config->memory_region_size);

    cuda_init(config->dim, config->size_data_per_element, config->offset_data,
              config->max_m, config->num_data / config->shard_num,
              config->data_size);
    GlobalLogger->debug("dim: {}", config->dim);
    GlobalLogger->debug("size_data_per_element: {}",
                        config->size_data_per_element);
    GlobalLogger->debug("offset_data: {}", config->offset_data);
    GlobalLogger->debug("max_m: {}", config->max_m);
    GlobalLogger->debug("num_data: {}", config->num_data);
    GlobalLogger->debug("data_size: {}", config->data_size);
    GlobalLogger->info("GPU initialized");
}

// 查询向量
std::pair<std::vector<long>, std::vector<float>>
IndexEngine::search_vectors(const std::vector<float> &query, int k,
                            int ef_search) {
    int query_num = query.size() / dim;
    std::vector<int> ids(query_num * k);
    std::vector<float> distances(query_num * k, std::numeric_limits<float>::max());
    std::vector<int> found(query_num);

    int shard_num = config->shard_num;
    std::vector<std::thread> threads;
    for (int i = 0; i < shard_num; i++) {
        threads.push_back(std::thread([&, i]() {
            char *data = gpu_memory_manager.get()->get_shard_buffer(i);
            int num_data = config->num_data;
            int num_per_shard = num_data / shard_num;
            int size_per_shard = gpu_memory_manager.get()->get_size_per_shard();
            int memory_per_query = size_per_shard / query_num;
            int num_per_query =
                memory_per_query / config->size_data_per_element;
            int size_per_element =
                gpu_memory_manager.get()->get_size_per_element();
            std::vector<bool> visited_table(num_per_shard * query_num, false);
            std::vector<int> fetch_nodes(ef_search * query_num);
            std::vector<int> entries(query_num);
            std::vector<int> fetch_size(query_num);
            unsigned int entry_node = pool.get()->get_entry_node(i);
            for (int j = 0; j < query_num; j++) {
                fetch_nodes[j * ef_search] = entry_node;
                fetch_size[j] = 1;
            }
            std::function<bool(std::vector<int> &)> check =
                [&](std::vector<int> &fetch_size) {
                    for (int i = 0; i < query_num; i++) {
                        if (fetch_size[i] > 0) {
                            return true;
                        }
                    }
                    return false;
                };
            while (check(fetch_size)) {
                std::vector<std::thread> query_threads;
                for (int j = 0; j < query_num; j++) {
                    query_threads.push_back(std::thread([&, j]() {
                        int entry = 0;
                        for (int z = 0; z < fetch_size[j]; z++) {
                            entry = fetch_nodes[j * ef_search + z];
                            if (visited_table[j * num_per_shard + entry]) {
                                continue;
                            }
                            int query_size = 0;
                            if (entry + num_per_query > num_per_shard) {
                                query_size =
                                    (num_per_shard - entry) * size_per_element;
                            } else {
                                query_size = memory_per_query;
                            }
                            pool.get()->rdma_read(
                                i, data + memory_per_query * j, query_size,
                                entry * size_per_element);
                            // 将fetch_nodes中后面的节点挪到前面
                            for (int k = 0; k < fetch_size[j] - z; k++) {
                                fetch_nodes[j * ef_search + k] =
                                    fetch_nodes[j * ef_search + z + 1 + k];
                            }
                            fetch_size[j] -= (z + 1);
                            break;
                        }
                        entries[j] = entry;
                        // GlobalLogger->debug("thread {} entry: {}", j, entry);
                    }));
                }
                for (auto &t : query_threads) {
                    t.join();
                }
                // GlobalLogger->debug("entries: {}", entries[0]);
                // cuda_check((char *)data, entries[0], size_per_query /
                // size_per_element);
                cuda_search(data, query, query_num, ef_search, k,
                            memory_per_query, visited_table, fetch_nodes,
                            fetch_size, entries, distances, ids,
                            found.data());
            }
        }));
    }
    for (auto &t : threads) {
        t.join();
    }
    std::vector<long> ids_long;
    for (int i = 0; i < query_num; i++) {
        for (int j = 0; j < k; j++) {
            ids_long.push_back(ids[i * k + j]);
        }
    }
    return std::make_pair(ids_long, distances);

    // auto [data, mutex] = gpu_memory_manager.get()->get_shard_buffer(0);
    // unsigned int entry_node = 0;
    // int num = 20;
    // int size = num * config->size_data_per_element;
    // int size_per_element = config->size_data_per_element;
    // pool.get()->rdma_read(0, (char *)data, size, entry_node *
    // size_per_element); cuda_check((char *) data, entry_node, num); return
    // std::make_pair(std::vector<long>(), std::vector<float>());
}
