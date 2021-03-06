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
    dataset_name = 'ANN_SIFT10K'
    #dataset_path.split('/')[-1].split('_')[0]
    entropy_path = entropy_list[dataset_index]

    dataset = np.ascontiguousarray(np.load(dataset_path).astype('float32'))
    instances = dataset.shape[0]
    dimension = dataset.shape[1]
    dataset = np.ascontiguousarray(np.transpose(dataset))

    entropy = np.load(entropy_path)
    performance = np.zeros((dimension, 9))
    index_brute = faiss.IndexFlatL2(instances)
    index_brute.add(dataset)
    quantilizer = faiss.IndexFlatL2(instances)
    index = faiss.IndexHNSWFlat(instances, 12)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 40
    index.add(dataset)

    for i in range(dimension):
        performance[i, 0] = entropy[i, 0]
        dis_truth, id_truth = index_brute.search(dataset[i, :].reshape(1, -1), k_list[-1])
        for j in range(len(k_list)):
            k = k_list[j]
            time_start = time.time()
            dis_search, ID_search = index.search(dataset[i, :].reshape(1, -1), k)
            time_end = time.time()
            qps = 1 / (time_end - time_start)
            dis_truth, id_truth = index_brute.search(dataset[i, :].reshape(1, -1), k)
            recall = len(set(id_truth[0,:]) & set(ID_search[0,:]))/len(set(id_truth[0, :]))
            #print(id_truth, ID_search)
            performance[i, 2*j+1] = recall
            performance[i, 2*j+2] = qps
            print('the result now is: ', i, k, recall, qps)
    
    np.save(os.path.join(save_path, dataset_name, 'entropy_performance.npy'),performance)



         



