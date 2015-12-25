import os;
import numpy as np;
import scipy.stats
import scipy.io
import cPickle as pickle
from scipy import misc;
import visualize;
import math;
import random;
import time;
import util;
from tube_db import Tube, Tube_Manipulator,TubeHash_Manipulator,TubeHash
from collections import namedtuple
import lsh;
import matplotlib.pyplot as plt;
import cudarray as ca
import nearest_neighbor
import multiprocessing


def createParams(type_Experiment):
    if type_Experiment=='compareHashWithToyExperiment':
        list_params=['in_file',
                    'num_hash_tables_all',
                    'key_type',
                    'out_file_pres',
                    'out_file_indices',
                    'out_file_html',
                    'rel_path']
        params=namedtuple('Params_compareHashWithToyExperiment',list_params);
    elif type_Experiment=='visualizeNNComparisonWithHash':
        list_params=['in_file',
                    'in_file_hash',
                    'out_file_html',
                    'rel_path',
                    'topn',
                    'img_size'];
        params=namedtuple('Params_visualizeNNComparisonWithHash',list_params);
    elif type_Experiment=='visualizeHashBinDensity':
        list_params=['hash_tables',
                    'in_files',
                    'out_files',
                    'out_file_html',
                    'rel_path',
                    'bins',
                    'height_width'];
        params=namedtuple('Params_visualizeHashBinDensity',list_params);
    elif type_Experiment=='saveHashAnalysisImages':
        list_params=['path_to_db',
                    'class_labels_map',
                    'percents',
                    'out_file_class_pre',
                    'out_file_hash_simple',
                    'out_file_hash_byClass',
                    'hashtable',
                    'inc',
                    'dtype']
        params=namedtuple('Params_saveHashAnalysisImages',list_params);
    elif type_Experiment=='saveBigFeatureMats':
        list_params=['out_file_featureMats_pre',
                    'out_file_meta_pre',
                    'path_to_db',
                    'out_file_paths',
                    'num_batches']
        params = namedtuple('Params_saveBigFeatureMats',list_params);
    else:
        params=None;

    return params

def saveHash((hash_file,deep_features_path,out_file,key_type,idx)):
    print idx
    vals=np.load(deep_features_path);
    vals=vals['fc7'];
    num_frames=vals.shape[0];
    feature_dim=vals.shape[1];
    vals=np.reshape(vals,(num_frames,feature_dim));
    hp_hash=lsh.HyperplaneHash(hasher_file=hash_file,key_type=key_type);    
    hash_keys=hp_hash.hash(vals);
    np.save(out_file,hash_keys);
    
def getIndicesHash(features_test,features_train,num_hash_tables,key_type):
    assert features_test.shape[1]==features_train.shape[1]
    hp_hash=lsh.HyperplaneHash((features_test.shape[1],num_hash_tables),key_type=key_type);
    hash_test=hp_hash.hash(features_test);
    hash_train=hp_hash.hash(features_train);

    indices_hash=np.zeros((features_test.shape[0],features_train.shape[0]),dtype='int');
    for idx_row in range(hash_test.shape[0]):
        row_curr=np.tile(np.expand_dims(hash_test[idx_row,:],0),(hash_train.shape[0],1));
        bin_match=np.equal(row_curr,hash_train);
        bin_match=bin_match.sum(axis=1);
        indices_hash[idx_row,:]=np.argsort(bin_match)[::-1];

    return indices_hash

def script_compareHashWithToyExperiment(params):
    in_file = params.in_file;
    num_hash_tables_all = params.num_hash_tables_all;
    key_type = params.key_type;
    out_file_indices = params.out_file_indices;
    out_file_pres = params.out_file_pres;
    out_file_html = params.out_file_html;
    rel_path = params.rel_path;

    [features_test,features_train,labels_test,labels_train,_,_,indices,_]=pickle.load(open(in_file,'rb'));
    visualize.saveMatAsImage(indices,out_file_indices);    
    
    hammings=[];
    for out_file_pre,num_hash_tables in zip(out_file_pres,num_hash_tables_all):
        indices_hash = getIndicesHash(features_test,features_train,num_hash_tables,key_type);
        visualize.saveMatAsImage(indices_hash,out_file_pre+'.png');    
        hamming=util.getHammingDistance(indices,indices_hash);
        pickle.dump([indices_hash,indices,hamming],open(out_file_pre+'.p','wb'));

        hammings.append(np.mean(hamming));
    
    sizes = scipy.misc.imread(out_file_indices);
    sizes = sizes.shape

    im_files_html=[];
    captions_html=[];
    for idx,out_file_pre in enumerate(out_file_pres):
        out_file_curr=out_file_pre+'.png'
        key_str=str(key_type);
        key_str=key_str.replace('<type ','').replace('>','');
        caption_curr='NN Hash. Num Hash Tables: '+str(num_hash_tables_all[idx])+' '+'Hamming Distance: '+str(hammings[idx]);
        im_files_html.append([out_file_indices.replace(rel_path[0],rel_path[1]),out_file_curr.replace(rel_path[0],rel_path[1])])
        captions_html.append(['NN cosine',caption_curr]);

    visualize.writeHTML(out_file_html,im_files_html,captions_html,sizes[0]/2,sizes[1]/2);

def getImgPathsAndCaptionsNN(indices,img_paths_test,img_paths_train,labels_test,labels_train,rel_path):
    img_paths_html=[];
    captions_html=[];
    # record_wrong=[]
    for r in range(indices.shape[0]):
        img_paths_row=[img_paths_test[r].replace(rel_path[0],rel_path[1])];
        captions_row=[labels_test[r]];
        for c in range(indices.shape[1]):
            rank=indices[r,c];
            img_paths_row.append(img_paths_train[rank].replace(rel_path[0],rel_path[1]))
            captions_row.append(labels_train[rank]);
            # if labels_train[rank]!=labels_test[r]:
            #     record_wrong.append(c);
        img_paths_html.append(img_paths_row);
        captions_html.append(captions_row);

    return img_paths_html,captions_html

def script_visualizeNNComparisonWithHash(params):
    in_file = params.in_file
    in_file_hash = params.in_file_hash
    out_file_html = params.out_file_html
    rel_path = params.rel_path
    topn = params.topn
    img_size = params.img_size

    [_,_,labels_test,labels_train,img_paths_test,img_paths_train,indices,_]=pickle.load(open(in_file,'rb'));
    [indices_hash,_,_]=pickle.load(open(in_file_hash,'rb'));

    img_paths_nn,captions_nn=getImgPathsAndCaptionsNN(indices,img_paths_test,img_paths_train,labels_test,labels_train,rel_path)
    img_paths_hash,captions_hash=getImgPathsAndCaptionsNN(indices_hash,img_paths_test,img_paths_train,labels_test,labels_train,rel_path)

    
    img_paths_all=[];
    captions_all=[];
    for idx in range(len(img_paths_nn)):
        img_paths_all.append(img_paths_nn[idx][:topn]);
        img_paths_all.append(img_paths_hash[idx][:topn]);
        captions_all.append([x+' nn' for x in captions_nn[idx][:topn]]);
        captions_all.append([x+' hash' for x in captions_hash[idx][:topn]]);

    visualize.writeHTML(out_file_html,img_paths_all,captions_all,img_size[0],img_size[1]);

def script_saveHashTableDensities(hash_table,path_to_db,out_file):
    
    mani_hash=TubeHash_Manipulator(path_to_db);
    mani_hash.openSession();

    toSelect=(TubeHash.hash_val,);
    criterion=(TubeHash.hash_table==hash_table,);
    hash_vals=mani_hash.select(toSelect,criterion);
    
    mani_hash.closeSession();

    hash_vals=[hash_val[0] for hash_val in hash_vals];
    hash_vals=np.array(hash_vals,dtype=np.uint8);
    hash_dict={};
    for val_curr in np.unique(hash_vals):
        hash_dict[val_curr]=(hash_vals==val_curr).sum();

    pickle.dump(hash_dict,open(out_file,'wb'));

def script_visualizeHashBinDensity(params):

    hash_tables = params.hash_tables
    in_files = params.in_files
    out_files = params.out_files
    out_file_html = params.out_file_html
    rel_path = params.rel_path
    bins = params.bins
    height_width = params.height_width

    min_maxs=[];
    for file_idx,in_file in enumerate(in_files):
        densities=pickle.load(open(in_file,'rb'));
        densities=densities.values();
        min_maxs.append((min(densities),max(densities)))
        visualize.hist(densities,out_files[file_idx],bins=bins,normed=True,xlabel='Bin Density',
            ylabel='Frequency',title="Hash Bins' Density")

    img_files_html=[[out_file.replace(rel_path[0],rel_path[1])] for out_file in out_files];
    captions_html=[]
    for idx_hash_table,hash_table in enumerate(hash_tables):
        caption_curr=str(hash_table)+' '+str(min_maxs[idx_hash_table]);
        captions_html.append([caption_curr]);

    visualize.writeHTML(out_file_html,img_files_html,captions_html,height_width[0],height_width[1]);

def getClassIdsCount(class_ids_all,hash_vals_all):
    hash_vals_uni=list(np.unique(hash_vals_all));

    counts_all=[];
    class_ids_breakdown=[];
    
    for hash_val in hash_vals_uni:
        class_ids=class_ids_all[hash_vals_all==hash_val];
        class_ids=np.sort(class_ids);
        class_ids_uni = np.unique(class_ids)
        counts=[(class_ids==class_id).sum() for class_id in class_ids_uni];
        counts_all.append(counts);
        class_ids_breakdown.append(list(class_ids_uni));
    return counts_all,class_ids_breakdown

def getDiscriminativeScore(counts_all):
    counts_reduce=[];
    for counts in counts_all:
        c_r=[float(c)/sum(counts) for c in counts];
        counts_reduce.append(c_r);
    ranks=[util.product(counts) for counts in counts_reduce];
    return ranks

def getHashAnalysisIm(counts_all,class_ids_breakdown,inc=5,colorByClass=False):
    
    h=len(counts_all)*inc;
    w=h

    im_display=np.zeros((h,w));
    for idx in range(len(counts_all)):
        start_idx=idx*inc
        count_all=counts_all[idx];
        class_ids=class_ids_breakdown[idx];
        count_all=[float(count_curr)/sum(count_all)*w for count_curr in count_all];
        start_col=0;

        if colorByClass:
            idx_sort=np.argsort(count_all);

        count_all.sort();

        for idx_count,count_curr in enumerate(count_all):
            if colorByClass:
                color=class_ids[idx_sort[idx_count]];
            else:
                color=idx_count
            im_display[start_idx:start_idx+inc,start_col:min(start_col+round(count_curr),w)] = color
            start_col=min(start_col+round(count_curr),w);

    return im_display

def getCumulativeInfo(frequency,percents):
    frequency.sort();
    frequency=frequency[::-1];
    cum_freq=np.array([sum(frequency[:idx]) for idx in range(1,len(frequency)+1)]);
    # print len(cum_freq),cum_freq[-1],sum(frequency);
    idx_perc=[];
    for perc in percents:
        val=sum(frequency)*perc;
        idx=max(np.where(cum_freq<val)[0])+1;
        idx_perc.append(idx);
    return cum_freq,idx_perc

def savePerClassCumulativeGraph(cum_freq,idx_perc,percents,out_file,title):
    # if norm:
    cum_freq=cum_freq/float(cum_freq[-1])
    xAndYs={};
    xAndYs['Cumulative Frequency']=(range(len(cum_freq)),cum_freq);
    for idx_curr,perc_curr in zip(idx_perc,percents):
        xAndYs[str(perc_curr*100)+'%']=([0,idx_curr,idx_curr],[perc_curr,perc_curr,0]);
    xlabel='Number of HashBins'
    ylabel='Percent of Images'
    visualize.plotSimple(xAndYs.values(),out_file,title=title,
        xlabel=xlabel,ylabel=ylabel,legend_entries=xAndYs.keys())

def script_saveHashAnalysisImages(params):
    path_to_db = params.path_to_db;
    class_labels_map = params.class_labels_map;
    percents = params.percents;
    out_file_class_pre = params.out_file_class_pre;
    out_file_hash_simple = params.out_file_hash_simple;
    out_file_hash_byClass = params.out_file_hash_byClass;
    hashtable = params.hashtable;
    inc = params.inc;
    dtype = params.dtype;
    # in_file = params.in_file;

    if not os.path.exists(out_file_class_pre+'.npz'):
        mani=Tube_Manipulator(path_to_db);

        mani.openSession();
        ids=mani.selectMix((Tube.class_idx_pascal,TubeHash.hash_val),(TubeHash.hash_table==hashtable,));
        mani.closeSession();
        
        ids=np.array(ids,dtype=dtype);
        np.savez(out_file_class_pre,ids);

    ids=np.load(out_file_class_pre+'.npz')['arr_0'];
    # ids=np.load(in_file)['arr_0'];
    
    counts_all,class_ids_breakdown = getClassIdsCount(ids[:,0],ids[:,1]);
    ranks = getDiscriminativeScore(counts_all);

    sort_idx=np.argsort(ranks);
    counts_all=[counts_all[idx] for idx in sort_idx];
    class_ids_breakdown=[class_ids_breakdown[idx] for idx in sort_idx];
    im_simple = getHashAnalysisIm(counts_all,class_ids_breakdown,inc=inc,colorByClass=False);
    im_byClass = getHashAnalysisIm(counts_all,class_ids_breakdown,inc=inc,colorByClass=True);

    visualize.saveMatAsImage(im_simple,out_file_hash_simple)
    visualize.saveMatAsImage(im_byClass,out_file_hash_byClass)

    counts_all_ravel=np.array([c for counts in counts_all for c in counts]);
    class_ids_breakdown_ravel=np.array([c for class_ids in class_ids_breakdown for c in class_ids]);
    class_id_pascal,class_idx_pascal = zip(*class_labels_map);

    for class_id_idx,class_id in enumerate(class_idx_pascal):
        frequency = counts_all_ravel[class_ids_breakdown_ravel==class_id]
        out_file=out_file_class_pre+'_'+class_id_pascal[class_id_idx]+'.png'
        title=class_id_pascal[class_id_idx]+' '+str(class_id)        
        cum_freq,idx_perc=getCumulativeInfo(frequency,percents)
        savePerClassCumulativeGraph(cum_freq/float(cum_freq[-1]),idx_perc,percents,out_file,title)

def getGiantFeaturesMatGPU(paths_to_features,feature_type='fc7'):
    shape_record=[];
    for idx,path_curr in enumerate(paths_to_features):
        mat_curr=np.load(path_curr)[feature_type];
        shape_record.append(mat_curr.shape[0]);
        mat_curr=mat_curr.reshape(mat_curr.shape[0],mat_curr.shape[1]);
        if idx==0:
            train = ca.array(mat_curr);
        else:
            train = ca.extra.concatenate(train,ca.array(mat_curr),axis=0);
    return train,shape_record

def getGiantFeaturesMatNumpy(paths_to_features,feature_type='fc7'):
    shape_record=[];
    for idx,path_curr in enumerate(paths_to_features):
        mat_curr=np.load(path_curr)[feature_type];
        shape_record.append(mat_curr.shape[0]);
        mat_curr=mat_curr.reshape(mat_curr.shape[0],mat_curr.shape[1]);
        if idx==0:
            train = mat_curr.astype('float32');
        else:
            train = np.concatenate((train,mat_curr.astype('float32')),axis=0);
    return train,shape_record

def getGPUArray(path_curr):
    mat_curr=np.load(path_curr)['fc7'];
    mat_curr=mat_curr.reshape(mat_curr.shape[0],mat_curr.shape[1]);
    test=ca.array(mat_curr);
    return test

def saveDotProduct((first_shot_path,second_shot_path,out_file,idx)):
    test = getGPUArray(first_shot_path);
    train = getGPUArray(second_shot_path);
    results = nearest_neighbor.getSimpleDot(test,train,gpuFlag=True)
    np.savez(out_file,results);
    return True

def script_TEMP_savePairwiseDot(paths_to_features,out_dir_dots):
    
    print len(paths_to_features);
    # print paths_to_features[:10]
    
    total_paths = len(paths_to_features);

    input_pairs = []
    
    for first_shot_idx in range(total_paths):
        for second_shot_idx in range(first_shot_idx+1,total_paths):
            first_shot = paths_to_features[first_shot_idx];
            second_shot = paths_to_features[second_shot_idx];
            first_video = first_shot[:first_shot.rfind('_')];
            second_video = second_shot[:second_shot.rfind('_')];
            if first_video==second_video:
                continue;
            else:

                first_path_split=first_shot.split('/');
                second_path_split=second_shot.split('/');

                first_path=first_path_split[-3]+'_'+first_path_split[-2];
                second_path=second_path_split[-3]+'_'+second_path_split[-2];
                out_file=os.path.join(out_dir_dots,first_path+'_'+second_path+'.npz');

                input_pairs.append((first_shot,second_shot,out_file,len(input_pairs)));
                

    print len(input_pairs);
    t=time.time();
    for arg_curr in input_pairs:
    #     # print arg_curr
        saveDotProduct(arg_curr);
    print time.time()-t

def script_saveBigFeatureMats(params):
    out_file_featureMats_pre=params.out_file_featureMats_pre
    out_file_meta_pre=params.out_file_meta_pre
    path_to_db=params.path_to_db
    out_file_paths=params.out_file_paths
    num_batches=params.num_batches

    if not os.path.exists(out_file_paths):
        mani=Tube_Manipulator(path_to_db);
        mani.openSession()
        paths_to_features=mani.select((Tube.deep_features_path,),distinct=True);
        paths_to_features=[path_curr[0] for path_curr in paths_to_features];
        mani.closeSession();

        random.shuffle(paths_to_features);
        pickle.dump(paths_to_features,open(out_file_paths,'wb'));

    paths_to_features=pickle.load(open(out_file_paths,'rb'));
    paths_to_features.sort();

    batch_size=len(paths_to_features)/num_batches;
    idxRange=util.getIdxRange(len(paths_to_features),batch_size);
    print len(idxRange),idxRange[-1];

    # start_idx=0;
    
    for start_idx in range(len(idxRange)-1):
        out_file_curr=out_file_featureMats_pre+'_'+str(start_idx)+'.npz';
        out_file_meta_curr=out_file_meta_pre+'_'+str(start_idx)+'.p';
        print start_idx,idxRange[start_idx],idxRange[start_idx+1],out_file_curr,out_file_meta_curr,
        paths_to_features_curr=paths_to_features[idxRange[start_idx]:idxRange[start_idx+1]]
        

        t=time.time();
        
        train,shape_record=getGiantFeaturesMatGPU(paths_to_features_curr);
        train=np.array(train);
        
        np.savez(out_file_curr,train);
        pickle.dump([paths_to_features_curr,shape_record],open(out_file_meta_curr,'wb'))
        print time.time()-t
        break


def loadBigFeatureMats(big_feature_files):
    # print len(big_feature_files)
    mats=[];
    # t_big=time.time();
    for idx,file_curr in enumerate(big_feature_files):
        mat_curr=np.load(file_curr)['arr_0'];
        mats.append(mat_curr);
    # print time.time()-t_big
    # print len(mats)
    return mats

def getNNIndicesForBigFeatureMats(test_org,mats):
    
    test=ca.array(test_org);
    distances=[]
    
    for idx_mat,mat_curr in enumerate(mats):
        print idx_mat
        distances.append(nearest_neighbor.getSimpleDot(test,ca.array(mats[idx_mat]),gpuFlag=True));
    # print '';
    
    distances=np.hstack(tuple(distances));
    indices=np.argsort(distances,axis=1)[:,::-1].astype(np.uint32)        

    return indices

def getPathsArray(meta_feature_files):
    paths=[];
    sizes=[];
    for file_curr in meta_feature_files:
        paths_curr,sizes_curr = pickle.load(open(file_curr,'rb'));
        paths.extend(paths_curr);
        sizes.extend(sizes_curr);

    paths_scaled=[];
    for idx,size_curr in enumerate(sizes):
        paths_scaled.extend([paths[idx]]*size_curr);
    
    return np.array(paths_scaled);

def script_getNNIndicesForTestMat(test_paths,big_feature_files,out_paths,featureType='fc7'):
    assert len(test_paths)==len(out_paths);
    
    print 'Loading training feature mats size', len(big_feature_files),
    t=time.time();
    mats = loadBigFeatureMats(big_feature_files);
    print time.time()-t

    for idx,(test_path,out_path) in enumerate(zip(test_paths,out_paths)):
        print idx+1,'of',len(test_paths)
        t=time.time();
        test_mat=np.load(test_path)[featureType];
        test_mat=test_mat.reshape(test_mat.shape[0],test_mat.shape[1]);
        indices=getNNIndicesForBigFeatureMats(test_mat,mats);
        np.savez(out_path,indices)
        print test_path,out_path,time.time()-t;
        



def main():
    
    return
    mani_hash=TubeHash_Manipulator(path_to_db);
    mani_hash.openSession();

    toSelect=(TubeHash.hash_val,);
    criterion=(TubeHash.hash_table==1,TubeHash.hash_val<10);
    
   
    t=time.time();
    hash_vals=mani_hash.select(toSelect,criterion);
    print time.time()-t;



    hash_vals=[hash_val[0] for hash_val in hash_vals];
    hash_vals=np.array(hash_vals,dtype=np.uint8);
    print hash_vals.shape

    hash_dict={};
    t=time.time();
    for val_curr in np.unique(hash_vals):
        hash_dict[val_curr]=(hash_vals==val_curr).sum();
    print time.time()-t

    print hash_dict

    from collections import Counter

    t=time.time();
    hash_dict_counter=Counter(hash_vals)
    print time.time()-t

    print hash_dict_counter
    
    # Counter({1: 3, 8: 1, 3: 1, 4: 1, 5: 1})


    mani_hash.closeSession();

    return

    # .p';
    hash_table=0;
    out_file=out_file_pre+'_'+str(hash_table)+'.p';
    script_saveHashTableDensities(path_to_db,out_file,hash_table);
    # dict_curr=pickle.load(open(out_file,'rb'));
    # print dict_curr
    

    return
    params_dict={};
    params_dict['in_file']='/disk2/decemberExperiments/toyExample/tubes_nn_32.p'
    params_dict['in_file_hash']='/disk2/decemberExperiments/toyExample/tubes_nn_32_8_32.p'
    params_dict['out_file_html']=params_dict['in_file_hash'][:-2]+'_qualitative_comparsion.html';
    params_dict['rel_path']=['/disk2','../../..'];
    params_dict['topn']=10;
    params_dict['img_size']=[200,200];
    params=createParams('visualizeNNComparisonWithHash');
    params=params(**params_dict);
    script_visualizeNNComparisonWithHash(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));
    
    


    return
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    mani=Tube_Manipulator(path_to_db);
    mani_hash=TubeHash_Manipulator(path_to_db);
    
    mani.openSession();
    deep_features_path_all=mani.select((Tube.deep_features_path,),distinct=True);
    deep_features_path_all=[x[0] for x in deep_features_path_all];
    
    print len(deep_features_path_all);
    # print deep_features_path_all[:10]

    mani_hash.openSession();
    
    for idx_deep_features_path,deep_features_path in enumerate(deep_features_path_all[11:]):
        t=time.time();
        hash_file=deep_features_path[:-4]+'_hash.npy';
        print hash_file
        idx_info=mani.select((Tube.idx,Tube.deep_features_idx),(Tube.deep_features_path==deep_features_path,));
        # idx_all,deep_features_idx_all=zip(*idx_info);
        hash_vals=np.load(hash_file);
        # print len(idx_all),hash_vals.shape
        for idx_foreign,row in idx_info:
            # hash_vals_curr=hash_vals[row];
            for hash_table,hash_val in enumerate(hash_vals[row]):
                # pass;
                # print type(idx_foreign),type(hash_table),type(int(hash_val))
                mani_hash.insert(idx=idx_foreign,hash_table=hash_table,hash_val=int(hash_val),commit=False);
        
        if idx_deep_features_path%10==0:
            mani_hash.session.commit();

        # print time.time()-t;

    mani_hash.closeSession();
    mani.closeSession();

if __name__=='__main__':
    main();

