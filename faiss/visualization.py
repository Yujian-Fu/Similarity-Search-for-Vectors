import numpy as np
from faiss_configuration import CONFIG
import os
from sklearn.metrics import recall_score


dataset_name = 'deep1M_base'
ID_name = 'deep1M_base_ID.npy'
recording_path = 'E:\Code_for_Similarity_Search\FAISS\searching_record'

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
            recall += recall_score(ground_truth, search_result, average = 'micro')
        
        recall = recall / num_query
        recall_matrix[i, search_methods-1] = recall
        recall = 0

print(recall_matrix)

    



