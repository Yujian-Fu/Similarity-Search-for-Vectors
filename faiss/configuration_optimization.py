import numpy as np
import faiss
import os
from faiss_configuration import CONFIG
from fvecs_read import fvecs_read
import time

def read_dataset(file_name):
    if file_name.split('.')[-1] == 'npy':
        file = np.load(file_name)
    elif file_name.split('.')[-1] == 'fvecs':
        file = fvecs_read(file_name)
    else:
        print ('the file name', file_name, 'is wrong!')

    return file


# this is used to find the optimal setting for various algorithm implementations
# the evaluate dataset include ANN_SIFT10K, ANN_SIFT1M and SIFT10M, the dimension is 128 for all

search_set_list = [
    #'/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT10K/SIFT10K_base.npy', 
    '/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/SIFT1M_base.npy',
    '/home/yujian/Downloads/similarity_search_datasets/SIFT10M/SIFT10M_feature.npy'
]


query_set_list = [
    #'/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT10K/SIFT10K_query.npy',
    '/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/SIFT1M_query_sub.npy',
    '/home/yujian/Downloads/similarity_search_datasets/SIFT10M/SIFT10M_feature_query.npy'
]


learn_set_list = [
    #'/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT10K/SIFT10K_train.npy',
    '/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/SIFT1M_train.npy',
    '/home/yujian/Downloads/similarity_search_datasets/SIFT10M/SIFT10M_feature_learn.npy'
]

#the path to save your recall and qps
save_path = '/home/yujian/Downloads/similarity_search_datasets/exp_record/'
k = 100

# the tested algorithms include: IVFFlat, IVFPQ,  PQ, HNSWFlat, LSH
time_weight = 0.5
recall_weight = 1 - time_weight 
# how to set a proper price function 
# price_function = 

for i in range(len(search_set_list)):
    dataset_name = search_set_list[i].split('/')[-1].split('_')[0]
    search_dataset = read_dataset(search_set_list[i])
    query_dataset = read_dataset(query_set_list[i])
    learn_dataset = read_dataset(learn_set_list[i])

    #dataset description 
    dimension = search_dataset.shape[1]
    query_length = query_dataset.shape[0]

    if not os.path.exists(os.path.join(save_path, dataset_name)):
        os.makedirs(os.path.join(save_path, dataset_name))

    #get groundtruth by brute force
    time_start = time.time()
    quantilizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexFlatL2(dimension)
    index.add(search_dataset)
    dis_truth, ID_truth = index.search(query_dataset, k)
    time_end = time.time()
    time_brute = time_end - time_start
    qps_brute = query_length / time_brute 
    file = open(os.path.join(save_path, dataset_name, 'qps_brute.txt'), 'w')
    file.write(' qps: ' + str(qps_brute) + ' query_length: ' +  str(query_length))
    file.close()
    np.save(os.path.join(save_path, dataset_name, 'truth_ID.npy'), ID_truth)
    np.save(os.path.join(save_path, dataset_name, 'truth_dis.npy'), dis_truth)
    

    # parameter for IVFFlat: 
    # the number of centroids in IVFFlat
    nlist_list = [5, 10, 20 ,50, 100, 200, 400, 800]
    # the number of centroids that will be visited in IVFFlat
    nprobe_list = [1, 3, 5, 8, 10, 20, 50, 80, 150, 200, 300]

    if not os.path.exists(os.path.join(save_path, dataset_name, 'IVFFlat')):
        os.makedirs(os.path.join(save_path, dataset_name, 'IVFFlat'))

    file = open(os.path.join(save_path, dataset_name, 'IVFFlat', 'qps_recall_IVFFlat.txt'), 'w')
    total_number = 0 
    for j in range(len(nlist_list)): 
        total_number += np.sum(list(map(lambda x:x<=nlist_list[j], nprobe_list)))

    for nlist in nlist_list:
        for nprobe in nprobe_list[0 : np.sum(list(map(lambda x:x<nlist, nprobe_list)))]:
            recall_record = np.zeros((query_length, 1))
            time_start = time.time()
            quantilizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantilizer, dimension, nlist) 
            index.nprobe = nprobe 
            assert not index.is_trained 
            index.train(learn_dataset) 
            assert index.is_trained 
            index.add(search_dataset) 
            dis_IVF, ID_IVF = index.search(query_dataset, k) 
            time_end = time.time() 
            time_IVF = time_end - time_start 
            for j in range(query_length): 
                ground_truth = ID_truth[j, :] 
                search_result = ID_IVF[j, :] 
                recall_record[j, 0] = len(set(ground_truth) & set(search_result))/len(set(ground_truth)) 
            
            recall = 0 
            for j in range(query_length):
                recall += recall_record[j, 0]
            recall = recall/query_length
            print('the IVF recall is with parameter: ', recall, nlist, nprobe)
            qps_IVF = query_length / time_IVF
            file.write('nlist: ' + str(nlist) + ' nprobe: ' + str(nprobe) + ' recall: ' + str(recall) + ' qps: ' + str(qps_IVF) + '\n')
            np.save(os.path.join(save_path, dataset_name, 'IVFFlat', ' nlist'+' '+ str(nlist)+' '+ 'nprobe' + str(nprobe) + '_ID.npy'), recall_record)
            np.save(os.path.join(save_path, dataset_name, 'IVFFlat', ' nlist'+' '+ str(nlist)+' '+ 'nprobe' + str(nprobe) + '_dis.npy'), dis_IVF)

    file.close()
    # parameters for HNSWFlat
    #
    num_of_neighbors_list = [4, 8, 12, 24, 36, 48, 64, 96]
    #
    efConstruction_list = [100, 200,  300, 400, 500, 600, 700, 800, 900]
    #
    efSearch_list = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    if not os.path.exists(os.path.join(save_path, dataset_name, 'HNSW')):
        os.makedirs(os.path.join(save_path, dataset_name, 'HNSW'))

    file = open(os.path.join(save_path, dataset_name, 'HNSW', 'qps_recall_HNSW.txt'), 'w')
    for num_of_neighbors in num_of_neighbors_list:
        for efConstruction in efConstruction_list:
            for efSearch in efSearch_list:
                recall_record = np.zeros((query_length, 1))
                time_start = time.time()
                quantilizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexHNSWFlat(dimension, num_of_neighbors)
                index.efConstruction = efConstruction
                index.efSearch = efSearch
                index.add(search_dataset)
                dis_hnsw, ID_hnsw = index.search(query_dataset, k)
                time_end = time.time()
                time_hnsw = time_end - time_start
                for j in range(query_length):
                    ground_truth = ID_truth[j, :]
                    search_result = ID_hnsw[j, :]
                    recall_record[j, 0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in range(query_length):
                    recall += recall_record[j, 0]
                recall = recall / query_length
                print('the hnsw recall with parameter is', recall, num_of_neighbors, efConstruction, efSearch)
                qps_hnsw = query_length / time_hnsw
                file.write('num_neighbor: '+ str(num_of_neighbors) + ' efCon: ' + str(efConstruction) + ' efS: ' + str(efSearch) + ' recall: ' +  str(recall) + ' qps: ' + str(qps_hnsw) + '\n')
                np.save(os.path.join(save_path, dataset_name, 'HNSW', ' num_neigh '+ str(num_of_neighbors)+' efCon ' + str(efConstruction) + ' efS ' + str(efSearch) + '_ID.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name, 'HNSW', ' num_neigh '+ str(num_of_neighbors)+' efCon ' + str(efConstruction) + ' efS ' + str(efSearch) + '_dis.npy'), dis_hnsw)
    file.close()




'''
    # parameters for IVFPQ:
    #the number of 
    nlist = 
    # the number of 
    code_size = 
    #the number of 
    nbits = 
    #
    nprobe = 


    # parameters for PQ
    #
    M = 
    #
    nbits = 
    #
    nprobe = 




    # parameters for LSH
'''

