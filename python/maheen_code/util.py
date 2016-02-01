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

def normalize(matrix,gpuFlag=False):
    if gpuFlag==True:
        import cudarray as ca
        norm=ca.sqrt(ca.sum(ca.power(matrix,2),1,keepdims=True));
        matrix_n=matrix/norm
    else:
        norm=np.sqrt(np.sum(np.square(matrix),1,keepdims=True));
        matrix_n=matrix/norm
    
    return matrix_n

def getHammingDistance(indices,indices_hash):
    ham_dist_all=np.zeros((indices_hash.shape[0],));
    for row in range(indices_hash.shape[0]):
        ham_dist_all[row]=scipy.spatial.distance.hamming(indices[row],indices_hash[row])
    return ham_dist_all    

def product(arr):
    p=1;
    for l in arr:
        p *= l
    return p;

def getIOU(box_1,box_2):
    box_1=np.array(box_1);
    box_2=np.array(box_2);
    minx_t=min(box_1[0],box_2[0]);
    miny_t=min(box_1[1],box_2[1]);
    min_vals=np.array([minx_t,miny_t,minx_t,miny_t]);
    box_1=box_1-min_vals;
    box_2=box_2-min_vals;
    # print box_1,box_2
    maxx_t=max(box_1[2],box_2[2]);
    maxy_t=max(box_1[3],box_2[3]);
    img=np.zeros(shape=(maxx_t,maxy_t));
    img[box_1[0]:box_1[2],box_1[1]:box_1[3]]=1;
    img[box_2[0]:box_2[2],box_2[1]:box_2[3]]=img[box_2[0]:box_2[2],box_2[1]:box_2[3]]+1;
    # print np.min(img),np.max(img)
    count_union=np.sum(img>0);
    count_int=np.sum(img==2);
    # print count_union,count_int
    # plt.figure();
    # plt.imshow(img,vmin=0,vmax=10);
    # plt.show();
    iou=count_int/float(count_union);
    return iou

def escapeString(string):
    special_chars='!"&\'()*,:;<=>?@[]`{|}';
    for special_char in special_chars:
        string=string.replace(special_char,'\\'+special_char);
    return string

def replaceSpecialChar(string,replace_with):
    special_chars='!"&\'()*,:;<=>?@[]`{|}';
    for special_char in special_chars:
        string=string.replace(special_char,replace_with);
    return string
