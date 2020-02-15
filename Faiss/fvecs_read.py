import numpy as np

def fvecs_read(filename):
    ffile = np.fromfile(filename, dtype = np.float32)
    if ffile.size == 0:
        return zeros((0, 0))
    dimension = ffile.view(np.int32)[0]
    assert dimension > 0
    ffile = ffile.reshape(-1, 1+dimension)
    if not all(ffile.view(np.int32)[:, 0] == dimension):
        raise IOError('Non-uniform vector sizes in ' + filename)
    ffile = ffile[:, 1:]
    ffile = ffile.copy()
    return ffile




