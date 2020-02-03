import numpy as np 
import os

def compute_entropy(accumulate_column):
    instances = accumulate_column.shape[0]
    each_probability = 1 / (instances)
    entropy = 0

    for i in range(instances):
        if accumulate_column[i, ] != 0:
            probability_i = accumulate_column[i, ] * each_probability
            entropy += probability_i*np.log(probability_i)
    
    return -entropy

dataset_path_list = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/images_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy'
    #'E:\Datasets_for_Similarity_Search\ANN_SIFT10K\siftsmall\SIFT10K_base.npy'
]

for dataset_path in dataset_path_list:
    print('now processing ', dataset_path)
    dataset = np.load(dataset_path)
    instances, dimension = dataset.shape
    entropy = np.zeros((dimension, 1))
    for i in range(dimension):
        if i % 50 == 0:
            print('now processing ', i, 'dimension in ', dimension)
        accumulate_column = np.zeros((instances, 1))
        feature_column = dataset[:, i]
        max_value = max(feature_column)
        min_value = min(feature_column)

        for j in range(instances):
            index = int((feature_column[j, ] - min_value) / ((max_value - min_value)/instances))
            if index == instances:
                index -= 1
            #print('the index is ', index)
            accumulate_column[int(index), 0] += 1
            
        entropy[i, 0] = compute_entropy(accumulate_column)
    save_path_list = dataset_path.split('/')[0:-1]
    save_path = '/'
    for i in range(len(save_path_list)):
        save_path = os.path.join(save_path, save_path_list[i])
    np.save(os.path.join(save_path, 'entropy.npy'), entropy)

