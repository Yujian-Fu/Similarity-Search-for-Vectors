import numpy as np   
import matplotlib.pyplot as plt


def get_optimal(performance):
    assert performance.shape[1] == 2
    instances = performance.shape[0]

    optimal_setting = []
    for i in range(instances):
        recall = performance[i, 0]
        better_flag = 0
        for j in range(instances):
            if performance[j, 0] >= recall:
                if performance[j, 1] >= performance[j, 0]:
                    better_flag = 1
                    break
        if better_flag == 0:
            optimal_setting.append(i)
    
    return optimal_setting

def draw_line_figure(performance):


def draw_scatter_figure(performance):
    




dataset_list = [
    ''

]

algo_list = [
    ''
    
]

record_path = ''


for dataset in dataset_list:
    for algo in algo_list:
        file_path = '/home/yujian/Downloads/similarity_search_datasets/exp_record/SIFT10K/LSH/qps_recall_LSH.txt'

        file = open(file_path, 'r')

        content = file.read()
        content = content.split('\n')[0:-1]
        instances = len(content)
        performance = np.zeros((instances, 2))
        for i in range(instances):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i, 0] = recall
            performance[i, 1] = qps
        
        optimal_ID = get_optimal(performance)
        optimal_setting = performance [optimal_ID, :]
        draw_scatter_figure()
        draw_line_figure()

        fig = plt.figure()
        plt.scatter(performance[:, 0], performance[:, 1], alpha = 0.8)
        
        plt.savefig('/home/yujian/Desktop/result.png')













