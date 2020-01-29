import numpy as np
from faiss-exps import read_dataset
import faiss
import os

# this is used to find the optimal setting for various algorithm implementations
# the evaluate dataset include ANN_SIFT10K, ANN_SIFT1M and SIFT10M, the dimension is 128 for all

search_set_list = CONFIG.DATASET_PATH_LIST
search_set_list = [search_set_list[2], search_set_list[1], search_set_list[7]]
query_set_list = CONFIG.QUERY_PATH_LIST
query_set_list = [query_set_list[2], query_set_list[1], query_set_list[7]]
learn_set_list = CONFIG.TRAIN_PATH_LIST
learn_set_list = [learn_set_list[2], learn_set_list[1], learn_set_list[7]]

#the path to save your recall and qps
save_path = ''
k = 100

# the tested algorithms include: IVFFlat, IVFPQ,  PQ, HNSWFlat, LSH
time_weight = 
recall_weight = 1 - time_weight 
# how to set a proper price function 
price_function = 

for i in range(len(search_set_list)):
    dataset_name = 
    search_dataset = read_dataset(search_set_list[i])
    query_dataset = read_dataset(query_set_list[i])
    learn_dataset = read_dataset(learn_set_list[i])

    #dataset description 
    dimension = dataset.shape[1]
    query_length = query.shape[0]
    quantilizer = faiss.IndexFlatL2(dimension)

    #get groundtruth by brute force
    time_start = time.time()
    index = faiss.IndexFlatL2(dimension)
    index.add(search_dataset)
    dis_truth, ID_truth = index.search(query, k)
    time_end = time.time()
    time_brute = time_end - time_start
    qps_brute = query_length / time_brute 
    file = open(os.path.join(save_path, 'qps_IVFFlat.txt'), 'w')
    file.write(' qps: ', qps_IVF, ' query_length: ', query_length)
    file.close()
    np.save(os.path.join(save_path, dataset_name+'truth_ID.npy'), ID_truth)
    np.save(os.path.join(save_path, dataset_name+'truth_dis.npy'), dis_truth)
    

    # parameter for IVFFlat: 
    # the number of centroids in IVFFlat
    nlist_list = [5, 10, 20 ,50, 100, 200, 400, 800, 1000, 1200]
    # the number of centroids that will be visited in IVFFlat
    nprobe_list = [1, 3, 5, 8, 10, 20, 50, 80, 150, 200, 300]

    file = open(os.path.join(save_path, dataset_name+'qps_recall_IVFFlat.txt'), 'w')
    total_number = 0 
    for j in len(nlist): 
        totoal_numer += np.sum(list(map(lambda x:x<=nlist_list[j], nprobe_list)))

    for nlist in nlist_list:
        for nprobe in nprobe_list[0:]np.sum(list(map(lambda x:x<=nlist, nprobe_list)))
            recall_record = np.zeros((query_length, 1))
            time_start = time.time()
            index = faiss.IndexIVFFlat(quantilizer, dimension, nlist) 
            index.probe = nprobe 
            assert not index.is_trained 
            index.train(learn_dataset) 
            assert index.is_trained 
            index.add(search_dataset) 
            dis_IVF, ID_IVF = index.search(query_dataset, k) 
            time_end = time.time() 
            time_IVF = time_end - time_start 
            for j in query_length: 
                ground_truth = ID_truth[j, :] 
                search_result = ID_IVF[j, :] 
                recall_record[j, 0] = len(set(ground_truth) & set(search_result))/len(set(ground_truth)) 
            
            recall = 0 
            for j in query_length:
                recall += recall_record[j, 0]
            recall = recall/query_length
            print('the IVF recall is', recall)
            qps_IVF = query_length / time_IVF
            file.write('nlist: ', nlist, ' nprobe: ', nprobe, ' recall: ', recall, ' qps: ', qps_IVF)
            np.save(os.path.join(save_path, dataset_name+' nlist'+' '+ str(nlist)+' '+ 'nprobe' + str(nprobe) + '_ID.npy'), recall_record)
            np.save(os.path.join(save_path, dataset_name+' nlist'+' '+ str(nlist)+' '+ 'nprobe' + str(nprobe) + '_dis.npy'), dis_IVF)

    file.close()
    file = open(os.path.join(save_path, 'qps_recall_hnsw.txt'), 'w')
    # parameters for HNSWFlat
    #
    num_of_neighbors_list = [4, 8, 12, 24, 36, 48, 64, 96]
    #
    efConstruction_list = [100, 200,  300, 400, 500, 600, 700, 800, 900]
    #
    efSearch_list = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    file = open(os.path.join(save_path, dataset_name + 'qps_recall_HNSW.txt'), 'w')
    for num_of_neighbors in num_of_neighbors_list:
        for efConstruction in efConstruction_list:
            for efSearch in efSearch_list:
                recall_record = np.zeros((query_length, 1))
                time_start = time.time()
                index = faiss.IndexHNSWFlat(dimension, num_of_neighbors)
                index.efConstruction = efConstruction
                index.efSearch = efSearch
                index.add(search_dataset)
                dis_hnsw, ID_hnsw = index.search(query_dataset, k)
                time_end = time.time()
                time_hnsw = time_end - time_start
                for j in query_length:
                    ground_truth = ID_truth[j, :]
                    search_result = ID_hnsw[j, :]
                    recall_record[j, 0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in query_length:
                    recall += recall_record[j, 0]
                recall = recall / time_hnsw
                print('the hnsw recall is', recall)
                qps_hnsw = query_length / time_hnsw
                file.write('num_neighbor: ', num_of_neighbors, ' efCon: ', efConstruction, ' efS: ', efSearch, ' reacall: ', reacall, ' qps: ', qps_hnsw)
                np.save(os.path.join(save_path, dataset_name+' num_neigh '+ str(num_of_neighbors)+' efCon ' + str(efConstruction) + ' efS ' + str(efSearch) + '_ID.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name+' num_neigh '+ str(num_of_neighbors)+' efCon ' + str(efConstruction) + ' efS ' + str(efSearch) + '_dis.npy'), dis_hnsw)
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

