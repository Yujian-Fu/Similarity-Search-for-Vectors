import numpy as np
import sys 
import nmslib 
import time 
import math 
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
#search in euclidean space within the optimized index

all_data_matrix = np.random.random((10000,100))
(data_matrix, query_matrix) = train_test_split(all_data_matrix, test_size = 0.1)

K = 100
num_threads = 4
# initiate and then add dat points
space_name = 'l2'
index = nmslib.init(method = 'vptree', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
index.addDataPointBatch(data_matrix)

# create index
start_time = time.time()
#index.createIndex({'M':10, 'indexThreadQty': 4, 'efConstruction': 100})
index.createIndex({'chunkBucket':1, 'bucketSize':5, 'selectPivotAttempts': 10})
end_time = time.time()

#index.setQueryTimeParams({'efSearch': 100})

query_qty = query_matrix.shape[0]
start_time = time.time()
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end_time = time.time()
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
      (end_time-start_time, float(end_time-start_time)/query_qty, num_threads*float(end_time-start_time)/query_qty))

# Computing gold-standard data 
print('Computing gold-standard data')

start_time = time.time()
sindx = NearestNeighbors(n_neighbors=K, metric='l2', algorithm='brute').fit(data_matrix)
end_time = time.time()

print('Brute-force preparation time %f' % (end_time - start_time))

start_time = time.time() 
gs = sindx.kneighbors(query_matrix)
end_time = time.time()

print('brute-force kNN time total=%f (sec), per query=%f (sec)' % 
      (end_time-start_time, float(end_time-start_time)/query_qty) )

# Finally computing recall
recall=0.0
for i in range(0, query_qty):
  correct_set = set(gs[1][i])
  ret_set = set(nbrs[i][0])
  recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
  print(correct_set, '\n this is to divide \n', ret_set)
recall = recall / query_qty
print('kNN recall %f' % recall)

# Save a meta index, but no data!
#index.saveIndex('dense_index_optim.bin', save_data=False)



























