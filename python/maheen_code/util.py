import numpy as np;


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

