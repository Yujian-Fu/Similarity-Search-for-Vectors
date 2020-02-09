import numpy as np 
import os
import faiss
import time

dataset_list = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy', 
]

entropy_list = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/entropy.npy'
]

save_path = '/home/y/yujianfu/similarity_search/datasets/'
k_list = [3, 5, 10, 50]

for dataset_index in range(len(dataset_list)):
    dataset_path = dataset_list[dataset_index]
    dataset_name = dataset_path.split('/')[-1].split('_')[0]
    entropy_path = entropy_list[dataset_index]

    dataset = np.load(dataset_path)
    instances, dimension = dataset.shape
    dataset = np.transpose(dataset)

    entropy = np.load(entropy_path)
    performance = np.zeros(dimension, 7)
    index_brute = faiss.IndexFlatL2(instances)
    index_brute.add(dataset)
    quantilizer = faiss.IndexFlatL2(instances)
    index = faiss.IndexIVFFlat(quantilizer, instances, 5)
    index.probe = 2
    index.train(dataset[0:50,:])

    for i in range(dimension):
        performance[i, 0] = entropy[i, 0]
        dis_truth, id_truth = index_brute.search(dataset[i], k_list[-1])
        for j in range(len(k_list)):
            k = k_list[j]
            time_start = time.time()
            dis_search, ID_search = index.search(dataset[i], k)
            time_end = time.time()
            qps = 1 / (time_end - time_start)
            recall = len(set(id_truth[:, 0:k]) & set(ID_search))/len(set(id_truth[:, 0:k]))
            performance[i, 2*j+1] = recall
            performance[i, 2*j+2] = qps
    
    np.save(op.path.join(save_path, dataset_name, 'entropy_performance.npy'),performance)



         



