import numpy as np   
import matplotlib.pyplot as plt
import heapq
from scipy import optimize
from scipy.interpolate import interp1d 
import glob
import os

def f_2(x, A, B, C):
    return A*x*x+B*x+C

def f_3(x, A, B, C, D):
    return A*x**3+B*x**2+C*x+D

def f_4(x, A, B, C, D, E):
    return A*x**4+B*x**3+C*x**2+D*x+E

def get_optimal(performance):
    assert performance.shape[1] == 2
    instances = performance.shape[0]

    index = []
    for i in range(instances):
        kick_out = 0
        for j in range(instances):
            if performance[j, 0] > performance[i, 0]:
                if performance[j, 1] > performance [i, 1]:
                    kick_out = 1
        if kick_out == 0:
            index.append(i)
        
    return index

def draw_line_figure(optimal_setting, dataset, algo):
    #qps_new = spline(optimal_setting[:, 0], optimal_setting[:, 1], recall_new)
    
    recall = optimal_setting[:, 0]
    qps = optimal_setting[:, 1]
    A, B, C, D = optimize.curve_fit(f_3, recall, qps)[0]
    recall_new = np.zeros((recall.shape[0]*3-2, ))

    for i in range(recall.shape[0]-1):
        recall_new[3*i, ] = recall[i,]
        recall_new[3*i + 1, ] = 2*recall[i, ]/3 + recall[i+1,]/3
        recall_new[3*i + 2, ] = recall[i, ]/3 + 2*recall[i+1,]/3
    
    recall_new[-1, ] = recall[-1, ]
    
    print(recall, recall_new)
    qps_new = np.zeros((recall.shape[0]*3-2, ))

    for j in range(recall_new.shape[0]):
        qps_new[j,] = f_3(recall_new[j, ], A, B, C, D)

    if algo != 'HNSW':
        plt.plot(recall_new, qps_new, label = dataset+' ' + algo)
    else:
        plt.plot(recall_new, qps_new, label = dataset+' ' + algo)

    plt.legend(prop={'size':12})
    plt.grid(alpha = 0.5, linestyle= '--')
    plt.tight_layout()
    plt.yscale('log') 
    #plt.show()
    return recall, qps

def draw_scatter_figure(performance, dataset, algo):
    plt.scatter(performance[:, 0], performance[:, 1], label = dataset+' ' + algo, marker= '*')
    plt.xlabel('recall')
    plt.ylabel('qps') 
    plt.yscale('log') 
    plt.legend(prop= {'size':12})
    plt.grid(alpha = 0.5, linestyle = '--')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(record_path, dataset, algo, 'configuration_scatter.png'))
    




dataset_list = [
    #'deep1M',
    #'GIST1M',
    #'SIFT1M',
    'SIFT10K',
    #'SIFT10M'
]

algo_list = [
    'LSH',
    'HNSW',
    'IVFFlat',
    'IVFPQ',
    'PQ'
]

record_path = '/home/yujian/Downloads/similarity_search_datasets/exp_record'


for dataset in dataset_list:
    plt.figure()

    for algo in algo_list:
        #file_path = os.path.join(record_path, algo, dataset)
        file_path = glob.glob(os.path.join(record_path, dataset, algo, '*.txt'))
        if file_path[0].split('/')[-1] == 'optimal_configuration.txt':
            file_path = file_path[1]
        else:
            file_path = file_path[0]
        
        #'/home/yujian/Downloads/similarity_search_datasets/exp_record/SIFT10K/LSH/qps_recall_LSH.txt'

        file = open(file_path, 'r')

        content = file.read()
        content = content.split('\n')[0:-1]
        instances = len(content)
        performance = np.zeros((int(instances/4), 2))
        cons_length = len(content[0].split(' '))
        for i in range(instances):
            line = content[i].split(' ')
            if len(line) > cons_length:
                recall = float(line[-3])
                qps = float(line[-1])
                k = int(line[-5])
                if k == 500:
                    performance[int(i/4), 0] = recall
                    performance[int(i/4), 1] = qps
        
        #draw_scatter_figure(performance, dataset, algo)

        index = get_optimal(performance)


        optimal_setting = performance[index, :]
        new_file = open(os.path.join(record_path, dataset, algo,'optimal_configuration.txt'), 'w')
        for i in range(len(index)):
            new_file.write(content[index[i]]+'\n')

        optimal_setting = optimal_setting[optimal_setting[:, 0].argsort()]

        recall_new, qps_new = draw_line_figure(optimal_setting, dataset, algo)
        #plt.plot(optimal_setting[:, 0], optimal_setting[:, 1], label = dataset+ ' ' + algo)

        #fig = plt.figure()
        #plt.scatter(performance[:, 0], performance[:, 1], alpha = 0.8)
    plt.plot(1.0, 7959, marker= '*', label = 'brute_force')
    plt.xlabel('recall')
    plt.ylabel('qps')
    plt.legend(prop={'size':12})
    plt.grid(alpha = 0.5, linestyle= '--')
    plt.tight_layout()
    plt.title('configuration skyline (k = 500)')
    plt.show()
    #plt.savefig(os.path.join(record_path, dataset, 'configuration_skyline.png'))














