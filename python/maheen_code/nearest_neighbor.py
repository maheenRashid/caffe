import sklearn;
from sklearn import preprocessing
import sklearn.neighbors
import util
import numpy as np
import time
import cudarray as ca
# import pycuda.autoinit
# import pycuda.gpuarray
# import skcuda.linalg
# skcuda.linalg.init();

def doCosineDistanceNN(features_curr,numberOfN=5,binarize=False):
    feature_len=1;
    for dim_val in list(features_curr.shape)[1:]:
        feature_len *= dim_val
    
    features_curr=np.reshape(features_curr,(features_curr.shape[0],feature_len));
    if binarize:
        features_curr[features_curr!=0]=1;
    

    features_curr=sklearn.preprocessing.normalize(features_curr, norm='l2', axis=1);
    distances=np.dot(features_curr,features_curr.T);
    np.fill_diagonal(distances,-1*float('Inf'));
    
    indices=np.argsort(distances, axis=1)[:,::-1]
    
    static_indices=np.indices(distances.shape);
    distances=distances[static_indices[0],indices]
    
    if numberOfN is not None:
        indices=indices[:,:numberOfN];
        distances=distances[:,:numberOfN];
    else:
        indices=indices[:,:-1];
        distances=distances[:,:-1];

    return indices,distances

def getNearestNeighbors(query,data,gpuFlag=False,normalize=True):

    distances = getSimpleDot(query,data,gpuFlag=gpuFlag,normalize=normalize)
    # query_n=util.normalize(query,gpuFlag=gpuFlag);
    # train_n=util.normalize(data,gpuFlag=gpuFlag);
        
    # if not gpuFlag:
    #     distances=np.dot(query_n,train_n.T);
    # else:
    #     train_n=np.ascontiguousarray(train_n.T);
    #     query_n=pycuda.gpuarray.to_gpu(query_n);
    #     train_n=pycuda.gpuarray.to_gpu(train_n);
    #     distances_gpu=skcuda.linalg.dot(query_n,train_n);
        
    #     distances=np.zeros(distances_gpu.shape,dtype=distances_gpu.dtype);
    #     distances_gpu.get(distances);
    
    indices=np.argsort(distances,axis=1)[:,::-1]
    distances=(1-np.sort(distances,axis=1))[:,::-1];        
    return indices,distances

def getSimpleDot(test,train,gpuFlag=False,normalize=True):
    if normalize:
        test = util.normalize(test,gpuFlag=gpuFlag);
        train = util.normalize(train,gpuFlag=gpuFlag);

    if gpuFlag:
        distances=ca.dot(test,ca.transpose(train));
        distances=np.array(distances);
    else:
        distances=np.dot(test,train.T);
    
    return distances