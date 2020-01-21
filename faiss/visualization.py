import numpy as np
from faiss_configuration import CONFIG
import os
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

dataset_name = 'deep1M_base'
ID_name = 'deep1M_base_ID.npy'
time_name = 'deep1M_base_function_time.txt'
recording_path = 'E:\Code_for_Similarity_Search\FAISS\searching_record'
#recording_path =  '/home/y/yujianfu/similarity_search/Similarity-Search-for-Vectors/faiss/searching_record/'

def compute_recall():


    recall_matrix = np.zeros((len(CONFIG.K),6))
    recall = 0
    for i in range(len(CONFIG.K)):
        path = os.path.join(recording_path, dataset_name, str(CONFIG.K[i]))
        ID = np.load(os.path.join(path, ID_name))
        exps, num_query, k = ID.shape
        assert exps == 7

        for search_methods in range(1, exps):
            for query in range(num_query):
                ground_truth = ID[0, query, :]
                search_result = ID[search_methods, query, :]
                recall += len(set(ground_truth) & set(search_result))/len(set(ground_truth))
                #print(ground_truth,search_result, recall)
            
            recall = recall / num_query
            print(recall)
            recall_matrix[i, search_methods-1] = recall
            recall = 0

    np.save(os.path.join(recording_path, dataset_name, 'recall_matrix.npy'), recall_matrix)


def draw_figure():
    matrix_path = ''
    recall_matrix = np.load(matrix_path)
    x = CONFIG.k
    plt.plot(x, recall_matrix[:, 0], x, recall_matrix[:, 1], x, recall_matrix[:, 2], x, recall_matrix[:, 3], x, recall_matrix[:, 4], x, recall_matrix[:, 5], x, recall_matrix[:, 6],)

def compute_acceleration():
    time_matrix = np.zeros((len(CONFIG.K), 7))
    for k in range(len(CONFIG.K)):
        path = os.path.join(recording_path, dataset_name, str(CONFIG.K[k]), time_name)
        file = open(path)

        time_all = file.read()
        time_list = time_all.split('\n')

        for i in range(7):
            time = float(time_list[i].split(' ')[-1])
            print(time)
            time_matrix[k, i] = time


    x = CONFIG.K
    plt.plot(x, time_matrix[:, 0], x, time_matrix[:, 1], x, time_matrix[:, 2], x, time_matrix[:, 3], x, time_matrix[:, 4], x, time_matrix[:, 5], x, time_matrix[:, 6])
    plt.show()
    print(time_matrix)


#compute_recall()
compute_acceleration()


