# This is the code for testing the construction time, memory consumption and recall, qps, distance ratio
from faiss.faiss_configuration import CONFIG
import faiss
import nmslib
import numpy as np 
from memory_profiler import profile
import os
from annoy import AnnoyIndex
import time

dataset_list_ = ['SIFT10K', 'SIFT1M', 'GIST1M', 'SIFT10M', 'Deep10M']
dataset_list = [
    [
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_train.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_query.npy',
    ],

    [
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_train.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_query_sub.npy',
    ],

    [
        '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_learn.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_query.npy',

    ],

    [
        '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature_learn.npy',
        '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature.npy',
        '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature_query.npy'
    ],

    [
        '/home/y/yujianfu/similarity_search/datasets/Deep1B/Deep10M_train.npy',
        '/home/y/yujianfu/similarity_search/datasets/Deep1B/Deep10M.npy',
        '/home/y/yujianfu/similarity_search/datasets/Deep1B/Deep10M_query.npy'
    ]
    ]

algorithm_list = ['HNSW', 'LSH', 'IVFPQ', 'VP-tree']
K_list = [1, 5, 10, 20, 50, 100, 200, 500]
save_dir = '/home/y/yujianfu/similarity_search/datasets/exp_record/'
param_list = {'HNSW': [64, 600, 300], 'LSH': [1024], 'IVFPQ': [400, 480, 200], 'Annoy': [100]}

#dataset is a list contains [train_dataset, search_dataset, query_dataset]
@profile(precision=4,stream=open('./memory_profiler.log','a'))
def faiss_build(algorithm, dataset):
    dimension = dataset[0].shape[1]
    assert dataset[0].shape[1] == dataset[1].shape[1] == dataset[2].shape[1]
    time_start = time.time()
    if algorithm == 'HNSW':
        param = param_list['HNSW']
        assert len(param) == 3
        index = faiss.IndexHNSWFlat(dimension, param[0])
        index.hnsw.efConstruction = params[1]
        index.hnsw.edSearch = params[2]

    elif algorithm == 'LSH':
        param = param_list['LSH']
        assert len(param) == 1
        index = faiss.IndexLSH(dimension, param[0])
    
    elif algorithm == 'IVFPQ':
        quantilizer = faiss.IndexFlatL2(dimension)
        param = param_list['IVFPQ']
        assert len(param) == 3
        index = faiss.IndexIVFPQ(quantilizer, dimension, param[0], param[1], 8)
        index.nprobe = param[2]
    assert not index.is_trained 
    index.train(dataset[0])
    assert index.is_trained
    index.add(dataset[1])
    time_end = time.time()
    return index, time_end-time_start
        

@profile(precision=4,stream=open('./memory_profiler.log','a'))
def annoy_build(dataset):
    dimension = dataset[0].shape[1]
    instances = dataset[1].shape[0]
    assert dataset[0].shape[1] == dataset[1].shape[1] == dataset[2].shape[1]
    time_start = time.time()
    param = param_list['Annoy']
    assert len(param) == 1
    index = AnnoyIndex(dimension, 'euclidean')
    for i in range(instances):
        index.add(i, dataset[1][i, :])
    time_end = time.time()
    return index, time_end - time_start


def faiss_search(index, dataset, truth_ID, truth_dis, k):
    query_length = dataset[2].shape[0]

    time_start = time.time()
    dis, ID = index.search(dataset[2], k)
    time_end = time.time()

    recall = 0
    recall_record = np.zeros((query_length, 1))
    for i in range(query_length):
        ground_truth = truth_ID[i, :]
        search_result = ID[i, :]
        recall_record[i,0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))

    recall = np.mean(recall_record)
    print('the recall for ', str(k), ' is ', str(recall))

    dis_matrix = dis / truth_dis
    dis_record = np.mean(dis_matrix, axis = 0)
    dis_ratio = np.mean(dis_matrix)
    return recall, dis_ratio, recall_record, dis_record, query_length/(time_end - time_start)


def annoy_search(index, dataset, truth_ID, truth_dis, k):
    query_length = dataset[2].shape[0]
    dis, ID = np.zeros(())
    time_start = time.time()
    search_time = 0
    recall_record = np.zeros((query_length, 1))
    dis_record = np.array((1, k))

    for i in range(query_length):
        time_start = time.time()
        [ID, dis] = index.get_nns_by_vector(dataset[3][i, :], k, include_distances = True)
        search_time += time.time() - time_start
        ground_truth = truth_ID[i, :]
        recall_record[i,0] = len(set(ground_truth) & set(ID)) / len(set(ground_truth))
        dis_ratio += np.array(dis) / truth_dis[i, :]

    dis_record /= query_length
    dis_ratio = np.mean(dis)
    recall = np.mean(recall_record)

    return recall, dis_ratio, recall_record, dis_record, query_length/(search_time)


def record(save_path, cons_time, recall, dis_ratio, recall_record, dis_record, qps, k):
    record_file = open(os.path.join(save_path, 'record.txt'), 'a')
    record_file.write('the constime: '+str(cons_time)+' the recall, dis_ratio, qps with k = ', str(k) + ' is '+ str(recall)+' '+str(dis_ratio)+' '+str(qps))
    np.save(os.path.join(save_path, 'recall_record.npy'), recall_record)
    np.save(os.path.join(save_path, 'dis_record.npy'), dis_record)


def faiss_test(algorithm, dataset_path):
    dataset_name = dataset_path[0].split('/')[-2]
    dataset = [np.load(dataset_path[i]) for i in range(3)]
    save_path = os.path.join(save_dir, dataset_name, algorithm)
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(save_path)
    index, cons_time = faiss_build(algorithm, dataset)
    for k in K_list:
        search_dataset = np.load(dataset_list[0])
        query_dataset = np.load(dataset_list[1])
        index_brute = faiss.IndexFlatL2(search_dataset.shape[0])
        truth_ID, truth_dis = index_brute.search(query_dataset, K_list[-1])
        recall, dis_ratio, recall_record, dis_record, qps = faiss_search(index, dataset, truth_ID, truth_dis, k)
        print('faiss with algorithm '+str(algorithm))+ ' k: ' + str(k) + ' recall: '+str(recall)
        record(save_path, cons_time, recall, dis_ratio, recall_record, dis_record, qps, k)

def annoy_test(dataset_path):
    dataset_name = dataset_path[0].split('/')[-2]
    dataset = [np.load(dataset_path[i]) for i in range(3)]
    save_path = os.path.join(save_dir, dataset_name, 'annoy')
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(save_path)

    index, cons_time = annoy_build(dataset)
    for k in K_list:
        search_dataset = np.load(dataset_list[0])
        query_dataset = np.load(dataset_list[1])
        index_brute = faiss.IndexFlatL2(search_dataset.shape[0])
        truth_ID, truth_dis = index_brute.search(query_dataset, K_list[-1])
        recall, dis_ratio, recall_record, dis_record, qps = annoy_search(index, dataset, truth_ID, truth_dis, k)
        record(save_path, cons_time, recall, dis_ratio, recall_record, dis_record, qps, k)


def exps():
    for dataset_path in dataset_list:
        annoy_test(dataset_path)
        for algorithm in algorithm_list:
            faiss_test (algorithm, dataset_path)

exps()







'''
def nmslib_test(algorithm, dataset, truth_ID, truth_dis):
    dataset_name = dataset.split('/')
    if not os.path.exists(os.path.join(save_path, dataset, )):
        os.makedirs()
    record_file = open('', 'w')

    index = nmslib_build(algorithm, dataset)
    recall, dis_ratio = nmslib_search(index, dataset)
    nmslib_record(recall, dis)ratio
'''