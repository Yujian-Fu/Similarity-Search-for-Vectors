import numpy as np 

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

record_path = ''

for dataset in dataset_list:
    for model in model_list:
        file_path = op.path.join(record_path, dataset, model, model+'.txt')
        content = file.read()
        content = content.split('\n')[1:]
        instances = len(content)
        performance = np.zeros((instances, 2))
        plt.figure()
        for i in range(int(instances/4)):
            line = content[i].split(' ')
            recall = float(line[-3])
            qps = float(line[-1])
            performance[i, 0] = recall
            performance[i, 1] = qps
            plt.plot(performance[:,0], performance[:, 1])

        

