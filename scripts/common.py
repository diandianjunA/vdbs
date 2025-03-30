import requests
import os
import numpy as np

def generate_random_float_vector(size=128, num_vectors=1):
    """
    生成一个包含指定数量随机浮点数的向量。
 
    参数:
    size (int): 向量中浮点数的数量，默认为128。
 
    返回:
    numpy.ndarray: 包含随机浮点数的向量。
    """
    return np.random.rand(num_vectors, size).astype(np.float32)

def insert(vectors, url="http://192.168.6.201:3000/insert"):
    """
    将向量数据发送到服务器

    :param vectors: 要插入的向量（numpy array）
    :param url: 插入向量的URL
    :param index_type: 索引类型
    """
    payload = {
        "operation": "insert",
        "objects": []
    }
    for i in range(len(vectors)):
        payload["objects"].append({
            "id": i,
            "vector": vectors[i].tolist(),
        })
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Inserted vectors successfully.")
        else:
            print(f"Failed to insert vectors. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error inserting vectors: {e}")

def search(vectors, top_k=10, url="http://localhost:4000/search"):
    """
    从服务器搜索最相似的向量

    :param vector: 要搜索的向量（numpy array）
    :param top_k: 返回的最相似向量数量
    :param url: 搜索向量的URL
    :param index_type: 索引类型
    """
    payload = {
        "operation": "search",
        "objects": [],
        "k": top_k
    }
    for i in range(len(vectors)):
        payload["objects"].append({
            "vector": vectors[i].tolist(),
        })
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Search result: {response.json()}")
        else:
            print(f"Failed to search. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error searching: {e}")

if __name__ == "__main__":
    vector = generate_random_float_vector(num_vectors=1000)
    insert(vector)
    input("Press Enter to search vectors...")
    vector = generate_random_float_vector(num_vectors=1)
    search(vector)

    