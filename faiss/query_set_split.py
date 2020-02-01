import numpy as np 
import os
from fvecs_read import fvecs_read
import sys
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

def distance_computing(query_point, search_dataset):
    (instances, dimension) = search_dataset.shape
    a = (query_point ** 2).dot(np.ones((dimension, instances)))
    b = np.ones(query_point.shape).dot((np.transpose(search_dataset)**2))
    c = query_point.dot(np.transpose(search_dataset))
    return np.sqrt(a + b - 2*c)


def read_dataset(file_name):
    if file_name.split('.')[-1] == 'npy':
        file = np.load(file_name)
    elif file_name.split('.')[-1] == 'fvecs':
        file = fvecs_read(file_name)
    else:
        print ('the file name', file_name, 'is wrong!')

    return np.ascontiguousarray(file.astype('float32'))


start_num = sys.argv[1:]
start_num = int(start_num[0])

dataset_list = [
    #'/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy', 
    #'/home/y/yujianfu/similarity_search/datasets/Cifar/images_train.npy',
    
    #'/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
    
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d.npy',
    
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature.npy'
    
    
    #'/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT10K/SIFT10K_base.npy'
]


#LID computation
for dataset_path in dataset_list[start_num:start_num+1]:
    K = 1000
    print('the dataset path is ', dataset_path)
    search_dataset = read_dataset(dataset_path)
    record_path = '/'
    for split_part in dataset_path.split('/')[0:-1]:
        record_path = os.path.join(record_path, split_part)
    
    (instances, dimension) = search_dataset.shape
    assert instances > dimension

    RC = np.zeros((instances, 1))
    LID_MLE_1000 = np.zeros((instances, 1))
    LID_MLE_500 = np.zeros((instances, 1))
    LID_RV_1000 = np.zeros((instances, 1))
    LID_RV_500 = np.zeros((instances, 1))

    quantilizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexFlatL2(dimension)
    index.add(search_dataset)
    dis_matrix, ID = index.search(search_dataset, K+100)
    print('finish computing distance')


    for i in range(instances):
        K = 1000
        if i % 1000 == 0:
            print ('now computing ', i, ' in ', instances)
        distance = dis_matrix[i , :].reshape([1, K+100])
        zero_sum = np.sum(list(map(lambda x:x == 0, distance)))
        distance = distance[ 0, zero_sum:zero_sum+K ].reshape([1, K])

        d_mean = np.mean(distance)
        d_min = np.min(distance)
        RC[i, 0] = d_mean / d_min
        # LID_MLE_1000
        IDx = 0
        for m in range(K):
            IDx = IDx + (1/K)*np.log(distance[0 , m]/distance[0, K-1])
        IDx = -1 / IDx
        LID_MLE_1000[i, 0] = IDx
        #LID_RV_1000
        numerator = np.log(K) - np.log(int(K/2))
        demoninator = np.log(distance[0, K - 1]) - np.log(distance[0 , int(K/2)])
        LID_RV_1000[i, 0] = numerator / demoninator

        K = 500
        # LID_MLE_500
        IDx = 0
        for m in range(K):
            IDx = IDx + (1/K)*np.log(distance[0 , m]/distance[ 0, K - 1])
        IDx = -1 / IDx
        LID_MLE_500[i, 0] = IDx
        #LID_RV_500
        numerator = np.log(K) - np.log(int(K/2))
        demoninator = np.log(distance[0 , K - 1]) - np.log(distance[0, int(K/2)])
        LID_RV_500[i, 0] = numerator / demoninator

    if not os.path.exists(os.path.join(record_path,'LID_and_RC')):
        os.makedirs(os.path.join(record_path,'LID_and_RC'))
    
    np.save(os.path.join(record_path, 'LID_and_RC', 'RC.npy'), RC)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_MLE_1000.npy'), LID_MLE_1000)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_MLE_500.npy'), LID_MLE_500)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_RV_1000.npy'), LID_RV_1000)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_RV_500.npy'), LID_RV_500)

LID_path = [

]

RC_path = [

]

for LID_file in LID_path:
    LID_record = np.load(LID_file)
    sns.kdeplot(LID_record, shade = 'True', color = 'black')
    sns.rugplot(LID_record, color = 'blue')
    plt.vline(x, ymin, ymax, color = 'c', linestyles = 'dashed')
    index_ID = np.argsort(LID_record)
    small_set_ID = index_ID[0:1000]
    largest_set_ID = index_ID[-1001:-1]
    mean_LID = index_ID[int(len(index_ID)/2)-500:int(len(index_ID)/2)+500, :]
    multiple_LID = index_ID[arange( int(len(index_ID)/2) - 500*int(len(index_ID)/1000), int(len(index_ID)/2) + 500*int(len(index_ID)/1000), int(len(index_ID)/1000), ]
    origin_file = np.load(dataset_path)
    np.save(os.path.join(save_path, 'small_LID.npy'), origin_file[small_set_ID, :])
    np.save(os.path.join(save_path, 'small_LID.npy'), origin_file[small_set_ID, :])





