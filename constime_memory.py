# This is the code for testing the construction time, memory consumption and recall, qps, distance ratio
import faiss
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
        '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_learn.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_query.npy',

    ],



    [
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_train.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
        '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_query_sub.npy',
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

algorithm_list = ['LSH','HNSW', 'IVFPQ']
K_list = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
save_dir = '/home/y/yujianfu/similarity_search/datasets/exp_record/'
param_list = {'HNSW': [64, 600, 300], 'LSH': [2048], 'IVFPQ': [400, 64, 200], 'Annoy': [100]}

#dataset is a list contains [train_dataset, search_dataset, query_dataset]

def get_distance(dataset, ID):
    search_dataset = dataset[1]
    query_dataset = dataset[2]
    dis_result = np.zeros(ID.shape)
    k = ID.shape[1]
    query_length = query_dataset.shape[0]
    for i in range(query_length):
        query_vector = query_dataset[i, :]
        for j in range(k):
            result_vector = search_dataset[int(ID[i, j]),:]
            dis_result[i, j] = np.sum(np.square(query_vector - result_vector))
    return dis_result


def faiss_build(algorithm, dataset,):
    dimension = dataset[0].shape[1]
    assert dataset[0].shape[1] == dataset[1].shape[1] == dataset[2].shape[1]
    time_start = time.time()
    if algorithm == 'HNSW':
        param = param_list['HNSW']
        assert len(param) == 3
        index = faiss.IndexHNSWFlat(dimension, param[0])
        index.hnsw.efConstruction = param[1]
        index.hnsw.efSearch = param[2]
        index.add(dataset[1])
        time_end = time.time()
        return index, time_end-time_start

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
    else:
        print('the algorithm input is wrong! ', algorithm)

    
    index.train(dataset[0])
    assert index.is_trained
    index.add(dataset[1])
    time_end = time.time()
    return index, time_end-time_start


def annoy_build(dataset, dataset_name):
    dimension = dataset[0].shape[1]
    instances = dataset[1].shape[0]
    assert dataset[0].shape[1] == dataset[1].shape[1] == dataset[2].shape[1]
    time_start = time.time()
    param = param_list['Annoy']
    assert len(param) == 1
    index = AnnoyIndex(dimension, 'euclidean')
    for i in range(instances):
        index.add_item(i, dataset[1][i, :])
        if i % 10000 == 0:
            print('annoy now finished ', i, ' instances ')
    index.build(param[0])
    time_end = time.time()
    #index.save('./'+dataset_name+'.ann')
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
        if (-1 in recall_record):
            print(recall_record)

    recall = np.mean(recall_record)

    dis = get_distance(dataset, ID)
    truth_dis[truth_dis< 0.001] = 0.001
    dis[dis< 0.001] = 0.001

    dis_matrix = dis / truth_dis

    dis_matrix[np.isnan(dis_matrix)] = 1.0
    dis_record = np.mean(dis_matrix, axis = 0)
    dis_ratio = np.mean(dis_matrix)
    #if dis_ratio > 10:
       #print('there seems to be a distance error, ', dis[dis_matrix > 10], truth_dis[dis_matrix>10], dis_matrix[dis_matrix>10], ID[dis_matrix>10]) 

    return recall, dis_ratio, recall_record, dis_record, query_length/(time_end - time_start)


def annoy_search(index, dataset, truth_ID, truth_dis, k):
    query_length = dataset[2].shape[0]
    time_start = time.time()
    search_time = 0
    recall_record = np.zeros((query_length, 1))
    dis_record = np.zeros((1, k)).astype('float')
    truth_dis[truth_dis < 0.001] = 0.001

    for i in range(query_length):
        time_start = time.time()
        [ID, dis] = index.get_nns_by_vector(dataset[2][i, :], k, include_distances = True)
        dis = np.square(dis)
        dis[dis<0.001] = 0.001
        search_time += time.time() - time_start
        ground_truth = truth_ID[i, :]
        recall_record[i,0] = len(set(ground_truth) & set(ID)) / len(set(ground_truth))
        #print(np.array(dis), truth_dis[i, :], np.square(np.array(dis)) / truth_dis[i, :], ground_truth, ID)
        #print(dataset[2][i,:], dataset[1][truth_ID[0,0],:], dataset[1][ID[0],:])
        for j in range(k):
            dis_record[0, j] += dis[j] / truth_dis[i, j]

    dis_record[np.isnan(dis_record)] = 1.0
    dis_record /= query_length
    dis_ratio = np.mean(dis_record)
    recall = np.mean(recall_record)


    return recall, dis_ratio, recall_record, dis_record, query_length/(search_time)


def record(save_path, record_file, cons_time, recall, dis_ratio, recall_record, dis_record, qps, k):
    record_file.write('the constime: '+str(cons_time)+' the recall, dis_ratio, qps with k = '+ str(k) + ' is '+ str(recall)+' '+str(dis_ratio)+' '+str(qps)+'\n')
    np.save(os.path.join(save_path, 'recall_record.npy'), recall_record)
    np.save(os.path.join(save_path, 'dis_record.npy'), dis_record)


#@profile(precision=4,stream=open('./memory_profiler.log','a'))
@profile(precision=4)
def faiss_test(algorithm, dataset_path):
    dataset_name = dataset_path[0].split('/')[-2]
    dataset = [np.ascontiguousarray(np.load(dataset_path[i]).astype('float32')) for i in range(3)]
    save_path = os.path.join(save_dir, dataset_name, algorithm)
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(save_path)
    
    record_file = open(os.path.join(save_path, 'record.txt'), 'w')
    print('start building faiss index with dataset ', dataset_name, ' and algorithm', algorithm)
    index, cons_time = faiss_build(algorithm, dataset)
    print('finish build faiss index')
    search_dataset = dataset[1]
    query_dataset = dataset[2]
    index_brute = faiss.IndexFlatL2(search_dataset.shape[1])
    index_brute.add(search_dataset)
    for k in K_list:
        truth_dis, truth_ID = index_brute.search(query_dataset, k)
        recall, dis_ratio, recall_record, dis_record, qps = faiss_search(index, dataset, truth_ID, truth_dis, k)
        print('faiss with algorithm '+str(algorithm)+ ' k: ' + str(k) + ' recall: '+str(recall) + ' dis_ratio ' + str(dis_ratio))
        record(save_path, record_file, cons_time, recall, dis_ratio, recall_record, dis_record, qps, k)

@profile(precision=4)
def annoy_test(dataset_path):
    dataset_name = dataset_path[0].split('/')[-2]
    dataset = [np.ascontiguousarray(np.load(dataset_path[i]).astype('float32')) for i in range(3)]
    save_path = os.path.join(save_dir, dataset_name, 'annoy')
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(save_path)

    record_file = open(os.path.join(save_path, 'record.txt'), 'w')
    print('start building annoy index with dataset ', dataset_name)
    index, cons_time = annoy_build(dataset, dataset_name)
    print('finish building annoy index')
    search_dataset = dataset[1]
    query_dataset = dataset[2]
    index_brute = faiss.IndexFlatL2(search_dataset.shape[1])
    index_brute.add(search_dataset)
    
    for k in K_list: 
        truth_dis, truth_ID = index_brute.search(query_dataset, k)
        recall, dis_ratio, recall_record, dis_record, qps = annoy_search(index, dataset, truth_ID, truth_dis, k)
        print('Annoy with k: ' + str(k) + ' recall: '+str(recall) + ' dis_ratio: ' + str(dis_ratio))
        record(save_path, record_file, cons_time, recall, dis_ratio, recall_record, dis_record, qps, k)


def exps():
    #file = open('./memory_profiler.log', 'a')
    for dataset_path in dataset_list:
        annoy_test(dataset_path)
        #file.write('now processing annoy with dataset'+dataset_path[0].split('/')[-2])
        for algorithm in algorithm_list:
            faiss_test (algorithm, dataset_path)
            #file.write('now processing faiss '+algorithm+' with dataset'+dataset_path[0].split('/')[-2])

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