import imagenet
import numpy as np
import os;
import time

def testFilesExcludedProperly(sub_list_files_val,meta_list_files_val,meta_val_ids_mapped,excluded_ids_all):
    idx_exclude=np.where(np.in1d(meta_val_ids_mapped,excluded_ids_all))[0];
    files_exclude=np.array(meta_list_files_val)[idx_exclude];
    idx_overlap=np.in1d(sub_list_files_val,files_exclude);
    print sum(idx_overlap);
    return sum(idx_overlap)==0;
    

def testFilesIncludedProperly(sub_list_files_val,meta_list_files_val,meta_val_ids_mapped,excluded_ids_all):
    idx_include=np.where(np.in1d(meta_val_ids_mapped,excluded_ids_all,invert=True))[0];
    files_include=np.array(meta_list_files_val)[idx_include];
    idx_overlap=np.in1d(sub_list_files_val,files_include);
    print sum(idx_overlap),len(files_include),len(sub_list_files_val);
    return sum(idx_overlap)==len(files_include)== len(sub_list_files_val);
    

def testFilesLabeledProperly(sub_val_idx_mapped,sub_list_files_val,meta_val_idx_mapped,meta_list_files_val):
    check=True
    sub_val_idx_mapped=np.array(sub_val_idx_mapped);
    meta_val_idx_mapped=np.array(meta_val_idx_mapped);
    sub_list_files_val=np.array(sub_list_files_val);
    try:
        idx_uni=np.unique(sub_val_idx_mapped);
        print len(idx_uni);
        print list(idx_uni)==range(len(idx_uni));
        assert list(idx_uni)==range(len(idx_uni));
        for idx_curr in idx_uni:
            files_curr=sub_list_files_val[sub_val_idx_mapped==idx_curr];
            files_match_idx=np.in1d(meta_list_files_val,files_curr);
            idx_meta=np.unique(meta_val_idx_mapped[files_match_idx])
            assert len(idx_meta.ravel())==1
            idx_check=meta_val_idx_mapped==idx_meta;
            # print np.array_equal(idx_check,files_match_idx);
            assert np.array_equal(idx_check,files_match_idx);
    except AssertionError:
        check=False;

    return check

def testSubsetFiles(meta_val_file,meta_synset_words_file,sub_val_file,sub_synset_words_file,excluded_ids_all_file,removePath):

    meta_list_files_val,meta_idx_val=zip(*imagenet.readLabelsFile(meta_val_file));
    meta_val_idx_mapped=[int(idx_curr) for idx_curr in meta_idx_val];
    
    meta_val_idx_mapped=np.array(meta_val_idx_mapped);

    t=time.time();
    meta_ids,meta_labels=zip(*imagenet.readLabelsFile(meta_synset_words_file));
    meta_ids=np.array(meta_ids);
    meta_val_ids_mapped=np.empty((len(meta_list_files_val),),dtype='object');
    
    for idx_curr in np.unique(meta_val_idx_mapped):
        idx_rel=meta_val_idx_mapped==idx_curr;
        meta_val_ids_mapped[idx_rel]=meta_ids[idx_curr];
    meta_val_ids_mapped=list(meta_val_ids_mapped);
    print time.time()-t

    sub_list_files_val,sub_idx_val=zip(*imagenet.readLabelsFile(sub_val_file));
    sub_val_idx_mapped=[int(idx_curr) for idx_curr in sub_idx_val];

    excluded_ids_all=imagenet.readSynsetsFile(excluded_ids_all_file);
    #all excluded files are excluded
    check_exclude=testFilesExcludedProperly(sub_list_files_val,meta_list_files_val,meta_val_ids_mapped,excluded_ids_all)
    #all included files are included
    check_include=testFilesIncludedProperly(sub_list_files_val,meta_list_files_val,meta_val_ids_mapped,excluded_ids_all)
    #all indices labeling is consistent
    check_labeling=testFilesLabeledProperly(sub_val_idx_mapped,sub_list_files_val,meta_val_idx_mapped,meta_list_files_val)
    return check_exclude,check_include,check_labeling;
