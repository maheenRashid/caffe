import sklearn;
from sklearn import preprocessing
import numpy as np
import time
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

