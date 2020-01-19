import numpy as np 
from nmslib_configuration import CONFIG
import nmslib
import time 
import os 
from sklearn.neighbors import NearestNeighbors


function_list = ['brute force', 'bnsw', 'sw-graph', 'vp-tree', 'napp', 'simple_invindx']

def print_result(distance, ID, time_recorder, dataset_name, k):
    path = os.path.join(CONFIG.RECORDING_FILE, datset_name)
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(CONFIG.RECORDING_FILE, dataset_name, str(k))
    if not os.path.exists(path):
        os.makedirs(path)
    
    file = open(os.path.join(path, dataset_name+'_function_time.txt'), 'w')
    assert len(function_list) == CONFIG.NUMBER_OF_EXPERIMENTS
    for i in range(CONFIG.NUMBER_OF_EXPERIMENTS):
        file.write(function_list[i] + ' ' + str(time_recorder[i, 1]-time_recorder[i, 0]) + ' ' + str(time_recorder[i, 2]-time_recorder[i, 1])+'\n')

    np.save(os.path.join(path, dataset_name+'_dis.npy'), distance)
    np.save(os.path.join(path, dataset_name+'_ID.npy'), ID)
    file.close()


def test_and_record(dataset, query, train_dataset, dataset_name, k):
    time_recorder = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS, 3))
    dimension = dataset.shape[1]
    query_length = query.shape[0]
    distance = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS, query_length, k))
    ID = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS, query_length, k))
    counter = 0


    #search by vp-tree
    index_time_params = {'indexThreadQty': CONFIG.num_threads}

    time_recorder[counter, 0] = time.time()
    index = nmslib.init(method = 'vp-tree', space = 'l2')
    index.addDataPointBatch(dataset.astype('int'))
    index.createIndex(print_progress = True)

    time_recorder[counter, 1] = time.time()
    query_time_params = {'efSearch': CONFIG.efS}
    index.setQueryTimeParams(query_time_params)
    neighbors = index.knnQueryBatch(query, k, CONFIG.num_threads)

    time_recorder[counter, 2] = time.time()
    for i in range(len(neighbors)):
        neighbor = np.array(neighbors[i])
        ID[counter, i, :] = neighbor[0, :]
        distance[counter, i, :] = neighbor[1, :]
    counter += 1
    print('vp-tree completed!')


    #search by brute force

    time_recorder[counter, 0] = time.time()
    index = nmslib.init(method = 'brute_force', space = 'l2')
    index.addDataPointBatch(dataset)
    index.createIndex(print_progress = True)

    time_recorder[counter, 1] = time.time()
    query_time_params = {'efSearch': CONFIG.efS}
    index.setQueryTimeParams(query_time_params)
    neighbors = index.knnQueryBatch(query, k, CONFIG.num_threads)

    time_recorder[counter, 2] = time.time()
    for i in range(len(neighbors)):
        neighbor = np.array(neighbors[i])
        ID[counter, i, :] = neighbor[0, :]
        distance[counter, i, :] = neighbor[1, :]
    counter += 1
    print('brute force completed!')


    #search by hnsw
    index_time_params = {'indexThreadQty': CONFIG.num_threads}

    time_recorder[counter, 0] = time.time()
    index = nmslib.init(method = 'hnsw', space = 'l2')
    index.addDataPointBatch(dataset)
    index.createIndex(index_time_params, print_progress = True)

    time_recorder[counter, 1] = time.time()
    query_time_params = {'efSearch': CONFIG.efS}
    index.setQueryTimeParams(query_time_params)
    neighbors = index.knnQueryBatch(query, k, CONFIG.num_threads)

    time_recorder[counter, 2] = time.time()
    for i in range(len(neighbors)):
        neighbor = np.array(neighbors[i])
        ID[counter, i, :] = neighbor[0, :]
        distance[counter, i, :] = neighbor[1, :]
    counter += 1
    print('hnsw completed!')

    #search by sw-graph
    index_time_params = {'indexThreadQty': CONFIG.num_threads}
    
    time_recorder[counter, 0] = time.time()
    index = nmslib.init(method = 'sw-graph', space = 'l2')
    index.addDataPointBatch(dataset)
    index.createIndex(index_time_params, print_progress = True)

    time_recorder[counter, 1] = time.time()
    query_time_params = {'efSearch': CONFIG.efS}
    index.setQueryTimeParams(query_time_params)
    neighbors = index.knnQueryBatch(query, k, CONFIG.num_threads)

    time_recorder[counter, 2] = time.time()
    for i in range(len(neighbors)):
        neighbor = np.array(neighbors[i])
        ID[counter, i, :] = neighbor[0, :]
        distance[counter, i, :] = neighbor[1, :]
    counter += 1
    print('sw-graph completed!')

    #search by napp
    index_time_params = {'indexThreadQty': CONFIG.num_threads}

    time_recorder[counter, 0] = time.time()
    index = nmslib.init(method = 'napp', space = 'l2')
    index.addDataPointBatch(dataset)
    index.createIndex(index_time_params, print_progress = True)

    time_recorder[counter, 1] = time.time()
    query_time_params = {'efSearch': CONFIG.efS}
    index.setQueryTimeParams(query_time_params)
    neighbors = index.knnQueryBatch(query, k, CONFIG.num_threads)

    time_recorder[counter, 2] = time.time()
    for i in range(len(neighbors)):
        neighbor = np.array(neighbors[i])
        ID[counter, i, :] = neighbor[0, :]
        distance[counter, i, :] = neighbor[1, :]
    counter += 1
    print('napp completed!')

    #search by simple_invindx
    index_time_params = {'indexThreadQty': CONFIG.num_threads}

    time_recorder[counter, 0] = time.time()
    index = nmslib.init(method = 'simple_invindx', space = 'l2')
    index.addDataPointBatch(dataset)
    index.createIndex(index_time_params, print_progress = True)

    time_recorder[counter, 1] = time.time()
    query_time_params = {'efSearch': CONFIG.efS}
    index.setQueryTimeParams(query_time_params)
    neighbors = index.knnQueryBatch(query, k, CONFIG.num_threads)

    time_recorder[counter, 2] = time.time()
    for i in range(len(neighbors)):
        neighbor = np.array(neighbors[i])
        ID[counter, i, :] = neighbor[0, :]
        distance[counter, i, :] = neighbor[1, :]
    counter += 1
    print('simple_invindx completed!')

    assert counter == CONFIG.NUMBER_OF_EXPERIMENTS

    print_result(distance, ID, time_recorder, dataset_name, k)
    




    























