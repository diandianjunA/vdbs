#include "dist_calculate.cuh"
#include "utils.cuh"

__inline__ __device__ float *
getDataByInternalId(char *data, unsigned int internal_id, int content_id) {
    return (float *)(data + (internal_id - content_id) * size_data_per_element +
                     offsetData);
}

template <typename T> __inline__ __device__ void PqPop(T *pq, int *size) {
    if (threadIdx.x != 0)
        return;
    if (*size == 0)
        return;
    (*size)--;
    if (*size == 0)
        return;
    T tail_dist = pq[*size];
    int p = 0, r = 1;
    while (r < *size) {
        if (r < (*size) - 1 and gt(pq[r + 1], pq[r]))
            r++;
        if (ge(tail_dist, pq[r]))
            break;
        pq[p] = pq[r];
        p = r;
        r = 2 * p + 1;
    }
    pq[p] = pq[*size];
}

template <typename T>
__inline__ __device__ void PqPush(T *pq, int *size, T node) {
    if (threadIdx.x != 0)
        return;
    int idx = *size;
    while (idx > 0) {
        int nidx = (idx + 1) / 2 - 1;
        if (ge(pq[nidx], node))
            break;
        pq[idx] = pq[nidx];
        idx = nidx;
    }
    pq[idx] = node;
    (*size)++;
}

__inline__ __device__ bool CheckAlreadyExists(const Node *pq, const int size,
                                              const int nodeid) {
    __syncthreads();
    // figure out the warp/ position inside the warp
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    static __shared__ bool shared[WARP_SIZE];
    bool exists = false;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (pq[i].nodeid == nodeid) {
            exists = true;
        }
    }

#if __CUDACC_VER_MAJOR__ >= 9
    unsigned int active = __activemask();
    exists = __any_sync(active, exists);
#else
    exists = __any(exists);
#endif
    // write out the partial reduction to shared memory if appropiate
    if (lane == 0) {
        shared[warp] = exists;
    }

    __syncthreads();

    // if we we don't have multiple warps, we're done
    if (blockDim.x <= WARP_SIZE) {
        return shared[0];
    }

    // otherwise reduce again in the first warp
    exists = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : false;
    if (warp == 0) {
#if __CUDACC_VER_MAJOR__ >= 9
        active = __activemask();
        exists = __any_sync(active, exists);
#else
        exists = __any(exists);
#endif
        // broadcast back to shared memory
        if (threadIdx.x == 0) {
            shared[0] = exists;
        }
    }
    __syncthreads();
    return shared[0];
}

__inline__ __device__ void PushNodeToSearchPq(char *data, Node *pq, int *size,
                                              const float *src_vec,
                                              const int dstid, int content_id, int ef_search) {
    if (CheckAlreadyExists(pq, *size, dstid)) {
        return;
    }
    const float *dst_vec = getDataByInternalId(data, dstid, content_id);
    float dist = GetDistanceByVec(src_vec, dst_vec, dims);
    __syncthreads();
    Node node = {dist, dstid, false};
    // printf("size: %d, ef_search: %d\n", *size, ef_search);
    if (*size < ef_search) {
        // printf("before push thread_id: %d, PushNodeToSearchPq: %f %f, size: %d\n",
               threadIdx.x, pq[0].distance, dist, *size);
        PqPush(pq, size, node);
        // printf("after push thread_id: %d, PushNodeToSearchPq: %f %f, size: %d\n",
               threadIdx.x, pq[0].distance, dist, *size);
    } else if (gt(pq[0].distance, dist)) {
        // printf("thread_id: %d, PushNodeToSearchPq: %f %f, size: %d\n",
        // threadIdx.x, pq[0].distance, dist, *size);
        PqPop(pq, size);
        PqPush(pq, size, node);
    }
    __syncthreads();
}

__inline__ __device__ bool CheckVisited(bool *visited_table, int target) {
    __syncthreads();
    bool ret = false;
    if (visited_table[target] != true) {
        __syncthreads();
        if (threadIdx.x == 0) {
            visited_table[target] = false;
        }
    } else {
        ret = true;
    }
    __syncthreads();
    return ret;
}

__inline__ __device__ int warp_reduce_cand(const Node *pq, int cand) {
#if __CUDACC_VER_MAJOR__ >= 9
    unsigned int active = __activemask();
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        int _cand = __shfl_down_sync(active, cand, offset);
        if (_cand >= 0) {
            if (cand == -1) {
                cand = _cand;
            } else {
                bool update = gt(pq[cand].distance, pq[_cand].distance);
                if (update)
                    cand = _cand;
            }
        }
    }
#else
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        int _cand = __shfl_down(cand, offset);
        if (_cand >= 0) {
            if (cand == -1) {
                cand = _cand;
            } else {
                bool update = gt(pq[cand].distance, pq[_cand.distance]);
                if (update)
                    cand = _cand;
            }
        }
    }
#endif
    return cand;
}

__inline__ __device__ int GetCand(const Node *pq, const int size) {
    __syncthreads();

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    static __shared__ int shared[WARP_SIZE];
    float dist = INFINITY;
    int cand = -1;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (not pq[i].checked) {
            bool update = gt(dist, pq[i].distance);
            if (update) {
                cand = i;
                dist = pq[i].distance;
            }
        }
    }
    cand = warp_reduce_cand(pq, cand);

    // write out the partial reduction to shared memory if appropiate
    if (lane == 0) {
        shared[warp] = cand;
    }
    __syncthreads();

    // if we we don't have multiple warps, we're done
    if (blockDim.x <= WARP_SIZE) {
        return shared[0];
    }

    // otherwise reduce again in the first warp
    cand = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1;
    if (warp == 0) {
        cand = warp_reduce_cand(pq, cand);
        // broadcast back to shared memory
        if (threadIdx.x == 0) {
            shared[0] = cand;
        }
    }
    __syncthreads();
    return shared[0];
}