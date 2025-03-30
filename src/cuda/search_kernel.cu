#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include "priority_queue.cuh"
#include "search_kernel.cuh"
#include <time.h>

#define CHECK(res)                                                             \
    {                                                                          \
        if (res != cudaSuccess) {                                              \
            printf("Error ：%s:%d , ", __FILE__, __LINE__);                    \
            printf("code : %d , reason : %s \n", res,                          \
                   cudaGetErrorString(res));                                   \
            exit(-1);                                                          \
        }                                                                      \
    }

__inline__ __device__ unsigned int *
get_linklist0(char *data, unsigned int internal_id, int content_id) {
    return (unsigned int *)(data +
                            (internal_id - content_id) * size_data_per_element);
}

__inline__ __device__ unsigned short int getListCount(unsigned int *ptr) {
    return *((unsigned short int *)ptr);
}

__global__ void search_kernel(char *data_, const float *query_data,
                              int num_query, int k, int size_per_query,
                              const int *entry_node, Node *device_pq,
                              int *global_fetch_nodes, int *fetch_size,
                              int ef_search_, bool *visited_table,
                              int *global_candidate_nodes,
                              float *global_candidate_distances, int *found_cnt,
                              int *nns, float *distances) {

    static __shared__ int size;

    // int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    Node *ef_search_pq = device_pq + ef_search * blockIdx.x;
    int *candidate_nodes = global_candidate_nodes + ef_search * blockIdx.x;
    float *candidate_distances =
        global_candidate_distances + ef_search * blockIdx.x;
    int *fech_nodes = global_fetch_nodes + ef_search * blockIdx.x;

    bool *_visited_table = visited_table + num_data * blockIdx.x;
    int& fetch_node_size = fetch_size[blockIdx.x];
    int content_id = entry_node[blockIdx.x];
    char *data = data_ + size_per_query * blockIdx.x;
    int num_per_query = size_per_query / size_data_per_element;

    for (int i = blockIdx.x; i < num_query; i += gridDim.x) {
        if (threadIdx.x == 0) {
            size = 0;
        }
        __syncthreads();

        const float *src_vec = query_data + i * dims;
        PushNodeToSearchPq(data, ef_search_pq, &size, query_data, entry_node[i],
                           content_id, ef_search_);
        // printf("entry_node[%d] = %d\n", i, entry_node[i]);

        if (CheckVisited(_visited_table, entry_node[i])) {
            continue;
        }
        __syncthreads();

        int idx = GetCand(ef_search_pq, size);
        // printf("idx = %d, size: %d\n", idx, size);
        while (idx >= 0) {
            __syncthreads();
            if (threadIdx.x == 0)
                ef_search_pq[idx].checked = true;
            int entry = ef_search_pq[idx].nodeid;
            __syncthreads();

            unsigned int *entry_neighbor_ptr =
                get_linklist0(data, entry, content_id);
            int deg = getListCount(entry_neighbor_ptr);
            // printf("deg[%d] = %d\n", entry, deg);

            for (int j = 1; j <= deg; ++j) {
                int dstid = *(entry_neighbor_ptr + j);
                // printf("dstid = %d\n", dstid);

                if (CheckVisited(_visited_table, dstid)) {
                    continue;
                }
                __syncthreads();

                if (dstid < content_id || dstid >= content_id + num_per_query) {
                    if (fetch_node_size >= ef_search) {
                        PqPop(fech_nodes, &fetch_node_size);
                        PqPush(fech_nodes, &fetch_node_size, dstid);
                    } else {
                        PqPush(fech_nodes, &fetch_node_size, dstid);
                    }
                    continue;
                }
                __syncthreads();

                PushNodeToSearchPq(data, ef_search_pq, &size, src_vec, dstid,
                                   content_id, ef_search_);
            }
            __syncthreads();
            idx = GetCand(ef_search_pq, size);
        }
        __syncthreads();

        // get sorted neighbors
        if (threadIdx.x == 0) {
            int size2 = size;
            while (size > 0) {
                candidate_nodes[size - 1] = ef_search_pq[0].nodeid;
                candidate_distances[size - 1] = ef_search_pq[0].distance;
                PqPop(ef_search_pq, &size);
            }
            found_cnt[i] = size2 < k ? size2 : k;
            for (int j = 0; j < found_cnt[i]; ++j) {
                nns[j + i * k] = candidate_nodes[j];
                distances[j + i * k] = out_scalar(candidate_distances[j]);
            }
        }
        __syncthreads();
    }
}

__global__ void kernel_check(char *data_, int content_id, int size) {
    printf("Hello from kernel\n");

    for (int i = content_id; i < content_id + size; i++) {
        float *data = getDataByInternalId(data_, i, content_id);
        printf("data[%d] = [", i);
        for (int j = 0; j < dims; j++) {
            printf("%f, ", data[j]);
        }
        printf("]\n");
    }

    for (int i = content_id; i < content_id + size; i++) {
        unsigned int *linklist = get_linklist0(data_, i, content_id);
        int deg = getListCount(linklist);
        printf("deg[%d] = %d\n", i, deg);
        printf("linklist[%d] = [", i);
        for (int j = 1; j <= deg; j++) {
          printf("%d, ", *(linklist + j));
        }
        printf("]\n");
    }
}

void cuda_search(char *data_, const std::vector<float> &query, int num_query,
                 int ef_search_, int k, int size_per_query,
                 std::vector<bool> &visited_table,
                 std::vector<int> &fetch_nodes, std::vector<int> fetch_size,
                 std::vector<int> &entries, float *distances, int *indices,
                 int *found) {
    int block_cnt_ = num_query;
    int dim = query.size() / num_query;
    int num_data_ = 0;
    cudaMemcpyFromSymbol(&num_data_, num_data, sizeof(int));
    thrust::device_vector<float> device_query(num_query * dim);
    thrust::device_vector<Node> device_pq(ef_search_ * block_cnt_);
    thrust::device_vector<int> global_candidate_nodes(ef_search_ * block_cnt_);
    thrust::device_vector<float> global_candidate_distances(ef_search_ *
                                                            block_cnt_);
    thrust::device_vector<int> device_found_cnt(num_query);
    thrust::device_vector<int> device_nns(k * num_query);
    thrust::device_vector<float> device_distances(k * num_query);
    thrust::device_vector<int> device_entries(num_query);
    thrust::device_vector<int> device_fetch_nodes(num_query * ef_search_);
    thrust::device_vector<int> device_fetch_size(num_query);
    thrust::device_vector<bool> device_visited_table(num_data_ * num_query,
                                                     false);
    thrust::copy(visited_table.begin(), visited_table.end(),
                 device_visited_table.begin());
    thrust::copy(query.begin(), query.end(), device_query.begin());
    thrust::copy(entries.begin(), entries.end(), device_entries.begin());
    thrust::copy(fetch_nodes.begin(), fetch_nodes.end(),
                 device_fetch_nodes.begin());
    thrust::copy(fetch_size.begin(), fetch_size.end(),
                 device_fetch_size.begin());

    search_kernel<<<block_cnt_, 256>>>(
        data_, thrust::raw_pointer_cast(device_query.data()), num_query, k,
        size_per_query, thrust::raw_pointer_cast(device_entries.data()),
        thrust::raw_pointer_cast(device_pq.data()),
        thrust::raw_pointer_cast(device_fetch_nodes.data()),
        thrust::raw_pointer_cast(device_fetch_size.data()), ef_search_,
        thrust::raw_pointer_cast(device_visited_table.data()),
        thrust::raw_pointer_cast(global_candidate_nodes.data()),
        thrust::raw_pointer_cast(global_candidate_distances.data()),
        thrust::raw_pointer_cast(device_found_cnt.data()),
        thrust::raw_pointer_cast(device_nns.data()),
        thrust::raw_pointer_cast(device_distances.data()));
    CHECK(cudaDeviceSynchronize());
    thrust::copy(device_nns.begin(), device_nns.end(), indices);
    thrust::copy(device_distances.begin(), device_distances.end(), distances);
    thrust::copy(device_found_cnt.begin(), device_found_cnt.end(), found);
    thrust::copy(device_fetch_nodes.begin(), device_fetch_nodes.end(),
                 fetch_nodes.begin());
    thrust::copy(device_fetch_size.begin(), device_fetch_size.end(),
                 fetch_size.begin());
    thrust::copy(device_visited_table.begin(), device_visited_table.end(),
                 visited_table.begin());
}

void cuda_init(int dims_, size_t size_data_per_element_, size_t offsetData_,
               int max_m_, int num_data_, size_t data_size_) {
    cudaMemcpyToSymbol(dims, &dims_, sizeof(int));
    cudaMemcpyToSymbol(size_data_per_element, &size_data_per_element_,
                       sizeof(size_t));
    cudaMemcpyToSymbol(offsetData, &offsetData_, sizeof(size_t));
    cudaMemcpyToSymbol(num_data, &num_data_, sizeof(int));
    cudaMemcpyToSymbol(data_size, &data_size_, sizeof(size_t));

    cudaMemcpyToSymbol(max_m, &max_m_, sizeof(int));
    CHECK(cudaDeviceSynchronize());

    // kernel_check<<<1, 1>>>();
    // CHECK(cudaDeviceSynchronize());
    // kernel_check2<<<1, 1>>>();
    // CHECK(cudaDeviceSynchronize());
}

void cuda_check(char *data_, int content_id, int size) {
    kernel_check<<<1, 1>>>(data_, content_id, size);
    CHECK(cudaDeviceSynchronize());
}