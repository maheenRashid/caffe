import numpy as np;
import scipy

def getIndexingArray(big_array,small_array):
    small_array=np.array(small_array);
    big_array=np.array(big_array);
    assert np.all(np.in1d(small_array,big_array))

    big_sort_idx= np.argsort(big_array)
    small_sort_idx= np.searchsorted(big_array[big_sort_idx],small_array)
    index_arr = big_sort_idx[small_sort_idx]
    return index_arr

def getIdxRange(num_files,batch_size):
    idx_range=range(0,num_files+1,batch_size);
    if idx_range[-1]!=num_files:
        idx_range.append(num_files);
    return idx_range;

def readLinesFromFile(file_name):
    with open(file_name,'rb') as f:
        lines=f.readlines();
    lines=[line.strip('\n') for line in lines];
    return lines

def normalize(matrix):
    norm=np.sqrt(np.sum(np.square(matrix),1));
    norm=np.expand_dims(norm,1);
    matrix_n=matrix/np.tile(norm,(1,matrix.shape[1]));
    return matrix_n

def getHammingDistance(indices,indices_hash):
    ham_dist_all=np.zeros((indices_hash.shape[0],));
    for row in range(indices_hash.shape[0]):
        ham_dist_all[row]=scipy.spatial.distance.hamming(indices[row],indices_hash[row])
    return ham_dist_all    

