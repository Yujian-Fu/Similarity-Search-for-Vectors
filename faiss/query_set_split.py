import numpy as np 
import os
from fvecs_read import fvecs_read
import sys
#import faiss
import matplotlib.pyplot as plt
import seaborn as sns
'''
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


    res = faiss.StandardGpuResources()
    
    index = faiss.IndexFlatL2(dimension)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(search_dataset)
    dis_matrix, ID = gpu_index_flat.search(search_dataset, K+20)
    print('finish computing distance')


    for i in range(instances):
        K = 1000
        if i % 1000 == 0:
            print ('now computing ', i, ' in ', instances)
        distance = dis_matrix[i , :].reshape([1, K+20])
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
'''

dataset_list = [
    'Cifar',
    'deep1M',
    'ANN_GIST1M',
    'Glove',
    'MNIST',
    'ANN_SIFT1M',
    'ANN_SIFT10K'
]

'''
dataset_path_list = [
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/Cifar/cifar-10-batches-py/images_train.npy',
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/Deep1M(with PQ from Deep1B)/deep1M/deep1M_base.npy',
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/ANN_GIST1M/gist/GIST1M_base.npy',
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/Glove/glove_840_300d.npy',
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/MNIST/MNIST_train_data.npy',
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/ANN_SIFT1M/sift/SIFT1M_base.npy',
    '/media/yujian/Seagate Backup Plus Drive/Datasets_for_Similarity_Search/ANN_SIFT10K/siftsmall/SIFT10K_train.npy'
]
'''

dataset_path_list = [
    '/home/y/yujianfu/similarity_search/datasets/Cifar/images_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy'
]


Metrics_list = [
    'LID_MLE_500',
    'LID_MLE_1000',
    'LID_RV_500',
    'LID_RV_1000',
    'RC'
    
]

#record_path = '/home/yujian/Desktop/LID_and_RC/'
record_path = '/home/y/yujianfu/similarity_search/datasets/'

for metric in Metrics_list:
    plt.figure()
    print('now processing ', metric)

    for i in range(len(dataset_list)):
        plt.subplot(len(dataset_list), 1, i+1)
        dataset = dataset_list[i]
        print('now processing', dataset)
        LID_file = os.path.join(record_path, dataset, 'LID_and_RC', metric+'.npy')
        LID_record = np.load(LID_file)
        LID_record = LID_record.reshape(LID_record.shape[0],)
        origin_file = np.load(dataset_path_list[i])
        dimension = origin_file.shape[1]
        sns.kdeplot(LID_record, shade = 'True', color = 'black', label = dataset+' '+ str(dimension))
        sns.rugplot(LID_record, color = 'black')
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        plt.vlines(np.median(LID_record), y_min, y_max, color = 'black', linestyles = 'dashed')
        #plt.show()
        '''
        index_ID = np.argsort(LID_record)
        small_set_ID = index_ID[0:1000]
        largest_set_ID = index_ID[-1001:-1]
        mean_set_ID = index_ID[int(len(index_ID)/2)-500:int(len(index_ID)/2)+500]
        multiple_set_ID = index_ID[np.arange( int(len(index_ID)/2) - 500*int(len(index_ID)/1000), int(len(index_ID)/2) + 500*int(len(index_ID)/1000), int(len(index_ID)/1000))]

        print('the index size:', len(small_set_ID), len(largest_set_ID), len(mean_set_ID), len(multiple_set_ID))
        dataset_path = dataset_path_list[i]
        origin_file = np.load(dataset_path)
        save_path = os.path.join('/home/y/yujianfu/similarity_search/datasets/Selected_Dataset', dataset, metric)
        #save_path = os.path.join('/home/yujian/Desktop/Selected Dataset', dataset, metric)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'small_LID.npy'), origin_file[small_set_ID, :])
        np.save(os.path.join(save_path, 'large_LID.npy'), origin_file[largest_set_ID, :])
        np.save(os.path.join(save_path, 'mean_LID.npy'), origin_file[mean_set_ID, :])
        np.save(os.path.join(save_path, 'multiple_LID.npy'), origin_file[multiple_set_ID, :])
        '''
    save_path = os.path.join('/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/', metric)
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    plt.legend()
    plt.suptitle(metric)
    plt.savefig(os.path.join(save_path, metric+'.png'))