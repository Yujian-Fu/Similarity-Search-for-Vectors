import numpy as np 
import seaborn as sns
import os

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
    'large_LID',
    'small_LIC',
    'large_RC',
    'small_RC'
]

record_path = '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset_performance'

for dataset in dataset_list:
    for model in model_list:
        file_path = os.path.join(record_path, dataset, model, model+'.txt')
        file = open(file_path)
        content = file.read()
        content = content.split('\n')[1:-1]
        print(content)
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
            performance[i, 0] = recall
            performance[i, 1] = qps
            plt.plot(performance[:,0], performance[:, 1], label = 'small_LID')

        for i in range(int(instances/2), int(instances*3/4)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i, 0] = recall
            performance[i, 1] = qps
            plt.plot(performance[:,0], performance[:, 1], label = 'large_RC')

        for i in range(int(instances*3/4), int(instances)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i, 0] = recall
            performance[i, 1] = qps
            plt.plot(performance[:,0], performance[:, 1], label = 'small_RC')
        
        plt.title('recall-qps '+dataset+' '+model)
        plt.legend()
        plt.savefig(os.path.join(record_path, dataset, model, 'recall-qps'+dataset+'-'+model+'.png'))

        for k in k_list:
            for metric in metric_list:
                recall_dis = np.load(os.path.join(record_path, dataset, model,metric+str(k)+'.npy'))
                recall_dis = recall_dis.reshape(recall_dis.shape[0],)
                sns.kdeplot(recall_dis, shade = 'True', color = 'black')
                sns.rugplot(recall_dis, color = 'black')
                axes = plt.gca()
                y_min, y_max = axes.get_ylim()
                plt.vlines(np.median(recall_dis), y_min, y_max, color = 'black', linestyles = 'dashed')
                print('the median is ', np.median(recall_dis))
                plt.savefig(os.path.join(record_path, dataset, model, metric+str(k)+'.png'))

                        

              

