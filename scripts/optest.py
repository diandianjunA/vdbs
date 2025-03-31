import threading
import random
import time
import requests
import numpy as np

# 查询请求函数
def search_vectors(vector, search_k, url="http://localhost:4000/search"):
    payload = {
        "operation": "search",
        "objects": [{
            "vector": vector,
        }],
        "k": search_k
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Search result: {response.json()}")
        else:
            print(f"Failed to search vectors. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error searching vectors: {e}")

# 插入请求函数
def insert(vector, vector_id, url="http://localhost:4000/insert"):
    payload = {
        "operation": "insert",
        "objects": [{
            "id": vector_id,
            "vector": vector,
        }]
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Inserted vectors successfully.")
        else:
            print(f"Failed to insert vectors. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error inserting vectors: {e}")

# 测试线程函数
def run_test(thread_id, num_operations, query_ratio, vector_dim, search_k):
    for _ in range(num_operations):
        vector = np.random.random(vector_dim).tolist()
        if random.random() < query_ratio:
            # 查询操作
            search_vectors(vector, search_k)
        else:
            # 插入操作
            vector_id = thread_id * 10000 + random.randint(1, 10000)
            insert(vector, vector_id)

# 主函数
def main():
    # 参数设置
    num_threads = 4          # 并发线程数
    num_operations = 100      # 每个线程执行的总操作数
    query_ratio = 1         # 查询操作的比例（0~1）
    vector_dim = 128          # 向量维度
    search_k = 5             # 查询返回的K个最近邻
    start_time = time.time()

    # 创建线程
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=run_test, args=(i, num_operations, query_ratio, vector_dim, search_k))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Throughput: {num_threads * num_operations / (end_time - start_time):.2f} operations/second")

if __name__ == "__main__":
    main()
