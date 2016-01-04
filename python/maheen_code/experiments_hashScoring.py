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
    if type_Experiment == 'scoreRandomFrames':
        list_params=['path_to_db',
                    'class_labels_map',
                    'npz_path',
                    'numberOfFrames',
                    'max_idx',
                    'n_jobs',
                    'table_idx_all',
                    'out_file_html',
                    'rel_path',
                    'width_height',
                    'out_file_frames']
        params = namedtuple('Params_scoreRandomFrames',list_params);
    else:
        params=None;

    return params

def getScoreForHashValLongerMethod(hash_table,hash_val,class_idx,video_idx):

    t=time.time();
    toSelect=(TubeHash.idx,)
    criterion=(TubeHash.hash_table==hash_table,TubeHash.hash_val==hash_val);
    total_count=mani_hash.count(toSelect=toSelect,criterion=criterion,distinct=True);
    print 'total_count',total_count, time.time()-t
    
    t=time.time();
    toSelect=(TubeHash.idx,);
    criterion=(TubeHash.hash_table==hash_table,TubeHash.hash_val==hash_val,
        Tube.class_idx_pascal==class_idx,Tube.video_id==video_idx);
    video_count=mani_hash.count(toSelect,criterion,mix=True);
    print 'count same video',video_count,time.time()-t

    t=time.time();
    toSelect=(TubeHash.idx,);
    criterion=(TubeHash.hash_table==hash_table,TubeHash.hash_val==hash_val,Tube.class_idx_pascal==class_idx);
    class_count=mani_hash.count(toSelect,criterion,mix=True);
    print 'count same class',class_count,time.time()-t

    #get the class id score
    class_count=class_count-video_count;
    total_count=total_count-video_count;
    score=class_count/float(total_count);
    print 'score',score;

    return score;

def getScoreForHashVal((path_to_db,hash_table,hash_val,class_idx,video_idx,class_idx_gt)):
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    toSelect=(Tube.class_idx_pascal,Tube.video_id)
    criterion=(TubeHash.hash_table==hash_table,TubeHash.hash_val==hash_val);
    vals=mani.selectMix(toSelect=toSelect,criterion=criterion);
    vals=np.array(vals);

    if not hasattr(class_idx,'__iter__'):
        class_idx=[class_idx];
    
    total_count_total=vals.shape[0];
    video_count_total=sum(np.logical_and(vals[:,0]==class_idx_gt,vals[:,1]==video_idx));

    scores=[];
    for class_idx_curr in class_idx:
        class_count=sum(vals[:,0]==class_idx_curr);
        if class_idx_curr==class_idx_gt:
            video_count=video_count_total;
        else:
            video_count=0;

        class_count=class_count-video_count;
        total_count=total_count_total-video_count_total;
        score=class_count/float(total_count);
        scores.append(score);

    mani.closeSession();
    return scores;

def getScoreForIdx(table_idx,path_to_db,class_idx_pascal=None,npz_path=None,n_jobs=1):
    
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();

    mani_hash=TubeHash_Manipulator(path_to_db);
    mani_hash.openSession();

    toSelect = (Tube.class_idx_pascal,Tube.video_id,Tube.frame_path)
    criterion = (Tube.idx==table_idx,);
    [(class_idx_gt,video_idx,frame_path)] = mani.select(toSelect,criterion)

    if class_idx_pascal is not None:
        class_idx=class_idx_pascal;
    else:
        class_idx=class_idx_gt

    toSelect=(TubeHash.hash_table,TubeHash.hash_val);
    criterion=(TubeHash.idx==table_idx,)
    hash_table_info=mani_hash.select(toSelect,criterion);
    print len(hash_table_info);
    mani_hash.closeSession();
    mani.closeSession();

    args=[];
    
    for hash_table_no in range(len(hash_table_info)):
        hash_table=hash_table_info[hash_table_no][0];
        hash_val=hash_table_info[hash_table_no][1];
        if npz_path is not None:
            args.append((npz_path,hash_table,hash_val,class_idx,video_idx,class_idx_gt));
        else:
            args.append((path_to_db,hash_table,hash_val,class_idx,video_idx,class_idx_gt));
            
    if n_jobs>1:
        p = multiprocessing.Pool(min(multiprocessing.cpu_count(),n_jobs))
        if npz_path is not None:
            scores=p.map(getScoreForHashValFromNpz,args)
        else:
            scores=p.map(getScoreForHashVal,args)
    else:
        scores=[];
        for arg in args:
            if npz_path is not None:
                scores.append(getScoreForHashValFromNpz(arg));
            else:
                scores.append(getScoreForHashVal(arg));
    
    return scores,class_idx_gt,frame_path

def getScoreForHashValFromNpz((npz_path,hash_table,hash_val,class_idx,video_idx,class_idx_gt)):
    fname=os.path.join(npz_path,str(hash_table)+'_'+str(hash_val)+'.npz');
    vals=np.load(fname)['arr_0'];

    if not hasattr(class_idx,'__iter__'):
        class_idx=[class_idx];

    total_count_total=vals.shape[0];

    video_count_total=sum(np.logical_and(vals[:,1]==class_idx_gt,vals[:,2]==video_idx));

    scores=[];
    for class_idx_curr in class_idx:
        class_count=sum(vals[:,1]==class_idx_curr);
        
        # total_count=vals.shape[0];
        class_count=sum(vals[:,1]==class_idx_curr);
        if class_idx_curr==class_idx_gt:
            video_count=video_count_total
        else:
            video_count=0;

        class_count=class_count-video_count;
        total_count=total_count_total-video_count_total;
        score=class_count/float(total_count);
        scores.append(score);
    # print 'score',score;
    return scores;

def script_scoreRandomFrames(params):
    path_to_db = params.path_to_db
    class_labels_map = params.class_labels_map
    npz_path = params.npz_path
    numberOfFrames = params.numberOfFrames
    max_idx = params.max_idx
    n_jobs = params.n_jobs
    table_idx_all = params.table_idx_all
    out_file_html = params.out_file_html
    rel_path = params.rel_path
    width_height = params.width_height
    out_file_frames = params.out_file_frames
    
    [class_labels,class_idx]=zip(*class_labels_map)

    if not os.path.exists(out_file_frames):
        frames_all=[];
        for table_idx in table_idx_all:
            scores,class_idx_curr,frame_path=getScoreForIdx(table_idx,path_to_db,
                class_idx_pascal=class_idx,npz_path=npz_path,n_jobs=n_jobs);
            
            frames_all.append([frame_path,class_idx_curr,scores]);

        pickle.dump(frames_all,open(out_file_frames,'wb'));

    frames_all=pickle.load(open(out_file_frames,'rb'));

    img_paths = [];
    captions = []
    for frame_path,class_idx_curr,scores in frames_all:

        scores=np.array(scores);
        avg_scores=np.mean(scores,axis=0);
        gt_idx=class_idx.index(class_idx_curr);
        gt_score=avg_scores[gt_idx];
        sort_idx=np.argsort(avg_scores)[::-1];
        max_idx=sort_idx[0];
        max_score=avg_scores[max_idx];
        max_class_idx=class_idx[max_idx];
        gt_rank=np.where(sort_idx==gt_idx)[0][0];

        caption_curr=[];
        caption_curr.append('GT');
        caption_curr.append(class_labels[class_idx.index(class_idx_curr)]);
        caption_curr.append(str(round(gt_score,4)));
        caption_curr.append(str(gt_rank+1));
        caption_curr.append('Best');
        caption_curr.append(class_labels[max_idx]);
        caption_curr.append(str(round(max_score,4)));

        caption_curr=' '.join(caption_curr)

        img_paths.append([frame_path.replace(rel_path[0],rel_path[1])]);
        captions.append([caption_curr]);

    visualize.writeHTML(out_file_html,img_paths,captions,width_height[0],width_height[1]);


def main():
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]

    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    total=0;
    for class_label,class_idx in class_labels_map:
        toSelect=(Tube.idx,)
        criterion = (Tube.class_idx_pascal==class_idx,);
        count_curr=mani.count(toSelect,criterion,distinct=True);
        total=total+count_curr;
        print class_label,class_idx,count_curr,count_curr/float(6371288)
    print total
    mani.closeSession();

if __name__=='__main__':
    main();

