import util;
import numpy as np;

def sortBySmallestDistanceRank(indices_interest,neighbor_index):
    
    
    indices_interest_index=util.getIndexingArray(neighbor_index,indices_interest);
    indices_interest_index=np.array(indices_interest_index);
    
    idx_to_order=np.argsort(indices_interest_index);
    
    indices_interest_index=indices_interest_index[idx_to_order];

    indices_ordered=indices_interest[idx_to_order];
    
    nn_rank=neighbor_index[indices_interest_index]
    return indices_interest_index,idx_to_order

def getDifferenceInRank(img_paths_train,img_paths_no_train,nn_rank_train,nn_rank_no_train):
    indices_difference=[];
    for idx,img_path_train_curr in enumerate(img_paths_train):
        indices_difference_curr=[];
        img_path_no_train_curr=img_paths_no_train[idx];
        for idx_curr,img_path_curr in enumerate(img_path_train_curr):
            n_idx_train=nn_rank_train[idx][idx_curr];
            idx_temp=img_path_no_train_curr.index(img_path_curr);
            n_idx_no_train=nn_rank_no_train[idx][idx_temp];
            indices_difference_curr.append(n_idx_train-n_idx_no_train);
        indices_difference.append(indices_difference_curr);

    # indices_difference=np.array(indices_difference);
    return indices_difference
