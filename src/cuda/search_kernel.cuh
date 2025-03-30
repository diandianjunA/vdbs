#pragma once

void cuda_init(int dims_, size_t size_data_per_element_, size_t offsetData_,
               int max_m_, int num_data_, size_t data_size_);

void cuda_search(char *data_, const std::vector<float> &query, int num_query,
                 int ef_search_, int k, int size_per_query,
                 std::vector<bool> &visited_table,
                 std::vector<int> &fetch_nodes, std::vector<int> fetch_size,
                 std::vector<int> &entries, float *distances, int *indices,
                 int *found);

void cuda_check(char *data_, int content_id, int size);