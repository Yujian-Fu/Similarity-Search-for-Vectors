import numpy as np 
import seaborn as sns
import os
import matplotlib.pyplot as plt
import glob
'''
k_list = [5, 10, 20, 40, 80, 120, 200, 400, 600, 800, 1000]

model_list = [
    'IVFFlat',
    'IVFPQ',
    'LSH',
    'HNSW',
    'PQ'
]


dataset_list = [
    'deep1M',
    'GIST1M',
    'SIFT1M',
    'SIFT10K'
]

metric_list = [
    'large_LID_',
    'small_LID_',
    'large_RC_',
    'small_RC_'
]

record_path = '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset_performance'

for dataset in dataset_list:
    for model in model_list:
        file_path = os.path.join(record_path, dataset, model, model+'.txt')
        file = open(file_path)
        content = file.read()
        content = content.split('\n')[1:-1]
        instances = len(content)
        assert instances % 4 == 0
        performance = np.zeros((instances, 2))
        plt.figure()
        performance = np.zeros((int(instances/4), 2))

        for i in range(int(instances/4)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i, 0] = recall
            performance[i, 1] = qps
        plt.plot(performance[:,0], performance[:, 1], label = 'large_LID')
        
        for i in range(int(instances/4), int(instances/2)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i%(int(instances/4)), 0] = recall
            performance[i%(int(instances/4)), 1] = qps
        plt.plot(performance[:,0], performance[:, 1], label = 'small_LID')

        for i in range(int(instances/2), int(instances*3/4)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i%(int(instances/4)), 0] = recall
            performance[i%(int(instances/4)), 1] = qps
        plt.plot(performance[:,0], performance[:, 1], label = 'large_RC')

        for i in range(int(instances*3/4), int(instances)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i%(int(instances/4)), 0] = recall
            performance[i%(int(instances/4)), 1] = qps
        plt.plot(performance[:,0], performance[:, 1], label = 'small_RC')
        
        plt.title('recall-qps '+dataset+' '+model)
        plt.legend()
        plt.savefig(os.path.join(record_path, dataset, model, 'recall-qps'+dataset+'-'+model+'.png'))

        for k in k_list:
            for metric in metric_list:
                recall_dis = np.load(os.path.join(record_path, dataset, model,metric+str(k)+'_recall.npy'))

                recall_dis = recall_dis.reshape(recall_dis.shape[0],)
                print(recall_dis.shape, recall_dis)                
                sns.distplot(recall_dis)
                sns.rugplot(recall_dis, color = 'black')
                axes = plt.gca()
                y_min, y_max = axes.get_ylim()
                plt.vlines(np.median(recall_dis), y_min, y_max, color = 'black', linestyles = 'dashed')
                print('the median is ', np.median(recall_dis))
                plt.savefig(os.path.join(record_path, dataset, model, metric+'_'+str(k)+'_.png'))
'''

dataset_list = [
    'deep1M',
    'GIST1M',
    'SIFT1M',
    'SIFT10K',
    'SIFT10M'
]

algo_list = [
    'LSH',
    'HNSW',
    'IVFFlat',
    'IVFPQ',
    'PQ'
]

record_path = '/home/yujian/Downloads/similarity_search_datasets/exp_record'

for algo in algo_list:
    for dataset in dataset_list:
        count_performance = np.zeros((1, 10))
        std_performance = np.zeros((1, 10))
        file_path = glob.glob(os.path.join(record_path, dataset, algo, '*.npy'))
        for i in range(len(file_path)):
            if file_path[i].split('/')[-1].split('_')[-1] == 'recall.npy':
                recall = np.load(file_path[i])
                mean_recall = np.mean(recall)
                std_recall = np.std(recall)
                if not (np.isnan(mean_recall) and np.isnan(std_recall)):
                    if mean_recall == 1:
                        mean_recall = 0.9
                    count_performance[0, int(mean_recall*10)] += 1
                    std_performance[0, int(mean_recall*10)] += std_recall
        for i in range(10):
            std_performance[0, i] /= count_performance[0, i]
        np.save(os.path.join(record_path, dataset, algo, 'std_recall.npy'), std_performance)
        print('the std_recall is', dataset, algo, std_performance)



     

