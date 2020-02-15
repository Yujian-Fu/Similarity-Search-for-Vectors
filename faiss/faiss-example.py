import numpy as np
import faiss
import time

dimension = 64
k = 4
nlist = 100
#this is the total number of cells
m = 8 
#number of subquantilizers
num_target = 100000
num_query = 10000
np.random.seed(1234)
xb = np.random.random((num_target, dimension)).astype('float32')
#generate the target vectors and the query vector, dimension is 64

xb[:, 0] += np.arange(num_target) / 1000
xq = np.random.random((num_query, dimension)).astype('float32')
xq[:, 0] += np.arange(num_query) / 1000
#add the bias to the target and the query
#xb is the target value in the database

time1 = time.time()
index = faiss.IndexFlatL2(dimension)
print(index.is_trained)
#add means add data to the index for further search
#if the index doesn't need to be trained, then is_trained = true
index.add(xb)
print(index.ntotal)
#build the index

#the search function returns two matrix, the first is the distance between the query and the neighbors
#the second is the ID of all neighbors, sorted by increasing distance
D, I = index.search(xq, k)
print(I[:5])

time2 = time.time()

##########################################################################################
#there are two parameters for this index 
nlist = 100
time3 = time.time()
quantilizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantilizer, dimension, nlist)
#use the product quantilizer for faster searching
assert not index.is_trained
index.train(xb);
assert index.is_trained

index.add(xb)
#add xb in the index object (should be the most time consuming part)

index.nprobe = 10
#the nprobe is the number of cells that are visited to perform a search
#if nprobe = nlist then it is a brute force search

D, I = index.search(xq, k)
print(I[:5])
time4 = time.time()
###########################################################################################

#this is the number of sub quantilizers
time5 = time.time()
quantilizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantilizer, dimension, nlist, m, 8)
#

index.train(xb)
index.add(xb)
print('good')

D, I = index.search(xq, k)
print(I[:5])
time6 = time.time()

########################################################################################
time7 = time.time()
#this is the gpu faiss for similarity searching
res = faiss.StandardGpuResources()
#the statement of using a GPU
index_flat = faiss.IndexFlatL2(dimension)
#firstly build an index in CPU
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
#make it into a gpu index
print(gpu_index_flat.ntotal)
gpu_index_flat.add(xb)
print(gpu_index_flat.ntotal)
D, I = gpu_index_flat.search(xq, k)
print(I[:5])
time8 = time.time()

#########################################################################################
time9 = time.time()
#this is the multiple gpu usage for faiss searching
ngpus = faiss.get_num_gpus()
print('number of gpus: ', ngpus)
cpu_index = faiss.IndexFlatL2(dimension)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
#there is no use of ngpus?
gpu_index.add(xb)
print(gpu_index.ntotal)

D, I = gpu_index.search(xq, k)
print(I[:5])
time10 = time.time()

#########################################################################################
time11 = time.time()
quantilizer = faiss.IndexFlatL2(dimension)

#is M the number of levels of HNSW?
index = faiss.IndexHNSWFlat(dimension, m)
index.add(xb)
D, I = index.search(xq, k)
print(I[:5])
time12 = time.time()

############################################################################################


print('the brute force index: ' , time2-time1)
print('the IVF index ', time4-time3)
print('the product quantilizer index', time6-time5)
print('the brute force index on gpu', time8-time7)
print('the brute force index on multiple gpus', time10-time9)
print('the HNSW index ', time12-time11)













