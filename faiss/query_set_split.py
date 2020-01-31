import numpy as np 
import os

def distance_computing(query_point, search_dataset):
    (instances, dimension) = search_dataset.shape
    a = (query_point ** 2).dot(np.ones((dimension, instances)))
    b = np.ones(query_point.shape).dot((np.transpose(search_dataset)**2))
    c = query_point.dot(np.transpose(search_dataset))
    return np.sqrt(a + b - 2*c)





dataset_list = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy', 
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
]


K = 1000
for dataset_path in dataset_list:
    search_dataset = np.load(dataset_path)
    record_path = '/'
    for split_part in dataset_path.split('/')[0:-1]:
        record_path = os.path.join(record_path, split_part)
    (instances, dimension) = search_dataset.shape

    RC = np.zeros((instances, 1))
    LID_MLE_1000 = np.zeros((instances, 1))
    LID_MLE_500 = np.zeros((instances, 1))
    LID_RV_1000 = np.zeros((instances, 1))
    LID_RV_500 = np.zeros((instances, 1))

    for i in range(instances):
        query_point = search_dataset[i, :]
        distance = distance_computing(query_point, search_dataset)
        distance = np.sort(distance)
        assert distance.shape == (instances, )
        zero_sum = 0
        for j in range(instances):
            if distance[j, ] == 0:
                zero_sum += 1
        if zero_sum == 1:
            print('computing ', i, 'in', instances)
        elif zero_sum > 1:
            print('something about the distance computing wrong: sum_zero = ', zero_sum)
        distance = distance[zero_sum: , ]

        dis_RC = distance[0:K, ]
        d_mean = np.mean(dis_RC)
        d_min = np.min(dis_RC)
        RC[i, 0] = d_mean / d_min
        # LID_MLE_1000
        IDx = 0
        for m in range(K):
            IDx = IDx + (1/K)*np.log(distance[m, ]/distance[K, ])
        IDx = -1 / IDx
        LID_MLE_1000[i, 0] = IDx
        #LID_RV_1000
        numerator = np.log(K) - np.log(int(K/2))
        demoninator = np.log(distance[K, ]) - np.log(distance[int(K/2), ])
        LID_RV_1000[i, 0] = numerator / demoninator

        K = 500
        # LID_MLE
        IDx = 0
        for m in range(K):
            IDx = IDx + (1/K)*np.log(distance[m, ]/distance[K, ])
        IDx = -1 / IDx
        LID_MLE_500[i, 0] = IDx
        #LID_RV
        numerator = np.log(K) - np.log(int(K/2))
        demoninator = np.log(distance[K, ]) - np.log(distance[int(K/2), ])
        LID_RV_500[i, 0] = numerator / demoninator

    if not os.path.exists(os.path.join(record_path,'LID_and_RC')):
        os.makedirs(os.path.join(record_path,'LID_and_RC'))
    
    np.save(os.path.join(record_path, 'LID_and_RC', 'RC.npy'), RC)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_MLE_1000.npy'), LID_MLE_1000)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_MLE_500.npy'), LID_MLE_500)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_RV_1000.npy'), LID_RV_1000)
    np.save(os.path.join(record_path, 'LID_and_RC', 'LID_RV_500.npy'), LID_RV_500)











