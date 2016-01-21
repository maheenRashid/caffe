import os;
import numpy as np;
import scipy.stats
import scipy.io
import cPickle as pickle
import copy
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
from collections import Counter
import itertools

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
                    'out_file_frames',
                    'frameCountNorm']
        params = namedtuple('Params_scoreRandomFrames',list_params);
    elif type_Experiment == 'testNpzScoreAccuracy':
        list_params=['path_to_db',
                    'path_to_hash',
                    'total_class_counts',
                    'class_idx',
                    'video_id',
                    'shot_id',
                    'num_hash_tables',
                    'num_hash_vals',
                    'tube_id',
                    'deep_features_idx',
                    'tube_file']
        params = namedtuple('Params_testNpzSchoreAccuracy',list_params);
    elif type_Experiment == 'saveNpzScorePerShot':
        list_params=['path_to_db',
                    'total_class_counts',
                    'class_idx',
                    'video_id',
                    'shot_id',
                    'out_file_scores',
                    'path_to_hash',
                    'num_hash_tables']
        params = namedtuple('Params_saveNpzScorePerShot',list_params);
    elif type_Experiment == 'visualizeBestTubeRank':
        list_params=['class_labels_map',
                    'rel_path',
                    'out_file_html',
                    'out_dir',
                    'score_info_file'];
        params = namedtuple('Params_visualizeBestTubeRank',list_params);
    elif type_Experiment =='verifyRecordedScoreMatchesDBScore':
        list_params=['path_to_db',
                    'path_to_hash',
                    'total_class_counts',
                    'img_path',
                    'class_label',
                    'video_id',
                    'shot_id',
                    'class_idx',
                    'score_file']
        params = namedtuple('Params_verifyRecordedScoreMatchesDBScore',list_params);
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

def getScoreForIdx(table_idx,path_to_db,class_idx_pascal=None,npz_path=None,n_jobs=1,total_counts=None):
    
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();

    mani_hash=TubeHash_Manipulator(path_to_db);
    mani_hash.openSession();

    toSelect = (Tube.class_idx_pascal,Tube.video_id,Tube.img_path)
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
            args.append((npz_path,hash_table,hash_val,class_idx,video_idx,class_idx_gt,total_counts));
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

def getScoreForHashValFromNpz((npz_path,hash_table,hash_val,class_idx,video_idx,class_idx_gt,total_counts)):
    fname=os.path.join(npz_path,str(hash_table)+'_'+str(hash_val)+'.npz');
    vals=np.load(fname)['arr_0'];

    if not hasattr(class_idx,'__iter__'):
        class_idx=[class_idx];

    total_count_total=vals.shape[0];

    video_count_total=sum(np.logical_and(vals[:,1]==class_idx_gt,vals[:,2]==video_idx));

    scores=[];
    for class_idx_curr in class_idx:
        class_count=sum(vals[:,1]==class_idx_curr);
        if class_idx_curr==class_idx_gt:
            video_count=video_count_total
        else:
            video_count=0;
        if total_counts is not None:
            class_count = class_count-video_count;
            total_count = total_counts[class_idx_curr];
            total_count = total_count - video_count;
        else:
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
    frameCountNorm = params.frameCountNorm

    [class_labels,class_idx]=zip(*class_labels_map)

    if not os.path.exists(out_file_frames):
        if frameCountNorm:
            total_counts=getTotalCountsPerClass(path_to_db,list(class_idx))
        else:
            total_counts=None

        frames_all=[];
        for table_idx in table_idx_all:
            scores,class_idx_curr,frame_path=getScoreForIdx(table_idx,path_to_db,
                class_idx_pascal=class_idx,npz_path=npz_path,n_jobs=n_jobs,total_counts=total_counts);
            
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

def getTotalCountsPerClass(path_to_db,class_idx_all):

    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    total_counts={};
    # total=0;
    for class_idx in class_idx_all:
        toSelect=(Tube.idx,)
        criterion = (Tube.class_idx_pascal==class_idx,);
        count_curr=mani.count(toSelect,criterion,distinct=True);
        
        total_counts[class_idx]=count_curr;

        # total=total+count_curr;
        # print class_label,class_idx,count_curr,count_curr/float(6371288),total_counts
    # print total
    mani.closeSession();
    return total_counts

def saveClassCountsForHashTables((in_file,out_file,idx)):
    print idx
    vals=np.load(in_file)['arr_0'];
    class_counts={};
    class_idx=np.unique(vals[:,1]);

    for class_idx_curr in class_idx:
        class_count=sum(vals[:,1]==class_idx_curr);
        class_counts[class_idx_curr]=class_count;
    
    pickle.dump(class_counts,open(out_file,'wb'));

def script_saveCounts(path_to_hash,num_hash_tables,num_hash_vals,n_jobs):
    hash_files=[];
    args=[];
    idx=0;
    for hash_table in range(num_hash_tables):
        for hash_val in range(num_hash_vals):
            in_file=str(hash_table)+'_'+str(hash_val)+'.npz';
            in_file=os.path.join(path_to_hash,in_file);
            out_file=in_file[:in_file.rindex('.')]+'_counts.p';
            args.append((in_file,out_file,idx));        
            idx+=1;
            
    p = multiprocessing.Pool(min(multiprocessing.cpu_count(),n_jobs))
    p.map(saveClassCountsForHashTables,args)

def getHashValsForFrameFromDeepFeaturesIdx(path_to_db,class_idx,video_id,shot_id,tube_id,deep_features_idx):
    mani=TubeHash_Manipulator(path_to_db);
    mani.openSession();
    toSelect=(TubeHash.idx,TubeHash.hash_table,TubeHash.hash_val);
    criterion=(Tube.class_idx_pascal==class_idx,Tube.video_id==video_id,Tube.shot_id==shot_id,Tube.tube_id==tube_id,Tube.deep_features_idx==deep_features_idx);
    vals=mani.selectMix(toSelect,criterion);
    mani.closeSession();
    return vals;
    
def getShotFrameCount(path_to_db,class_idx,video_id,shot_id):
    if type(path_to_db)==str:
        mani=Tube_Manipulator(path_to_db);
        mani.openSession();
    else:
        mani=path_to_db
    toSelect=(Tube.idx,);
    criterion=(Tube.class_idx_pascal==class_idx,Tube.video_id==video_id,Tube.shot_id==shot_id);
    frame_count=mani.count(toSelect,criterion);
    if type(path_to_db)==str:
        mani.closeSession();
    return frame_count;

def script_testNpzScoreAccuracy(params):
    path_to_db = params.path_to_db
    path_to_hash = params.path_to_hash
    total_class_counts = params.total_class_counts
    class_idx  = params.class_idx 
    video_id  = params.video_id 
    shot_id  = params.shot_id 
    num_hash_tables = params.num_hash_tables
    num_hash_vals = params.num_hash_vals
    tube_id = params.tube_id
    deep_features_idx = params.deep_features_idx
    tube_file = params.tube_file

    tube_scores_all=pickle.load(open(tube_file,'rb'));

    #get hash_vals for frame
    vals=getHashValsForFrameFromDeepFeaturesIdx(path_to_db,class_idx,video_id,shot_id,tube_id,deep_features_idx);

    frame_count=getShotFrameCount(path_to_db,class_idx,video_id,shot_id);

    vals=np.array(vals);
    # print vals.shape,np.unique(vals[:,0]),np.unique(vals[:,0]).size;
    assert np.unique(vals[:,0]).size==1;
    assert vals.shape==(num_hash_tables,3);

    total_class_count=total_class_counts[class_idx];
    deno=total_class_count-frame_count;

    scores_all={};
    #count total number of that shot in that bin
    mani=TubeHash_Manipulator(path_to_db);
    mani.openSession();
    for table_no in range(vals.shape[0]):
        print table_no
        hash_table_curr=vals[table_no,1];
        hash_val_curr=vals[table_no,2];

        toSelect=(TubeHash.idx,);
        criterion=(Tube.class_idx_pascal==class_idx,Tube.video_id==video_id,Tube.shot_id==shot_id,TubeHash.hash_table==hash_table_curr,TubeHash.hash_val==hash_val_curr);

        count_shot_in_bin=mani.count(toSelect=toSelect,criterion=criterion,mix=True)

        criterion=(Tube.class_idx_pascal==class_idx,TubeHash.hash_table==hash_table_curr,TubeHash.hash_val==hash_val_curr);

        count_class_in_bin=mani.count(toSelect,criterion,mix=True);

        numo_curr=count_class_in_bin-count_shot_in_bin;
        scores_all[hash_table_curr]=numo_curr/float(deno);

    mani.closeSession();

    #check for the accuracy of frame
    for hash_table_curr in scores_all:    
        print hash_table_curr,
        print 'from tube_scores',tube_scores_all[tube_id][deep_features_idx,hash_table_curr],
        print 'calculated',scores_all[hash_table_curr];
        assert tube_scores_all[tube_id][deep_features_idx,hash_table_curr]==scores_all[hash_table_curr]

def getTubeScoresMat(tube_id,vals,hash_bin_scores,deep_features_idx,num_hash_tables):
    
    tube_scores=np.empty((len(deep_features_idx),num_hash_tables));
    tube_scores[:]=np.nan;
    idx_tube_curr= np.where(vals[:,0]==tube_id)[0];
    for idx_curr in idx_tube_curr:
        deep_features_idx_curr=vals[idx_curr,1];
        hash_table_curr=vals[idx_curr,2];
        hash_val_curr=vals[idx_curr,3];
        score_curr=hash_bin_scores[(hash_table_curr,hash_val_curr)];
        tube_scores[deep_features_idx_curr,hash_table_curr]=score_curr;
    return tube_scores

def script_saveNpzScorePerShot(params):
    path_to_db = params['path_to_db']
    total_class_counts = params['total_class_counts']
    class_idx =  params['class_idx']
    video_id =  params['video_id']
    shot_id =  params['shot_id']
    out_file_scores =  params['out_file_scores']
    path_to_hash = params['path_to_hash']
    num_hash_tables = params['num_hash_tables']
    
    class_idx_assume = params.get('class_idx_assume',None);
    if class_idx_assume is None:
        class_idx_assume=class_idx;

    print params['idx']

    # print 'getting vals and frame count from db',
    # t=time.time();
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    toSelect=(Tube.deep_features_path,Tube.tube_id,Tube.deep_features_idx,TubeHash.hash_table,TubeHash.hash_val);
    criterion=(Tube.video_id==video_id,Tube.class_idx_pascal==class_idx,Tube.shot_id==shot_id);
    vals=mani.selectMix(toSelect,criterion);
    total_frames = getShotFrameCount(mani,class_idx,video_id,shot_id);
    mani.closeSession();
    # print time.time()-t

    # print 'getting hash_counts',
    # t=time.time();
    hash_info=[(tuple_curr[3],tuple_curr[4]) for tuple_curr in vals];
    hash_counts = dict(Counter(hash_info))
    # print time.time()-t

    # print 'getting hash_bin_scores',
    # t=time.time();
    if class_idx_assume ==class_idx:
        total_class_count=total_class_counts[class_idx_assume]-total_frames;
    else:
        total_class_count=total_class_counts[class_idx_assume];

    hash_bin_scores={};
    
    for idx,k in enumerate(hash_counts.keys()):
        in_file=str(k[0])+'_'+str(k[1])+'_counts.p'
        class_id_counts=pickle.load(open(os.path.join(path_to_hash,in_file),'rb'));
        
        if class_idx_assume ==class_idx:
            hash_bin_count=class_id_counts.get(class_idx_assume,0)-hash_counts[k];
        else:
            hash_bin_count=class_id_counts.get(class_idx_assume,0)

        hash_bin_scores[k]=hash_bin_count/float(total_class_count);
    # print time.time()-t;

    # print 'getting tube_scores_all',
    # t=time.time(); 
    vals_org=np.array(vals);
    deep_features_paths=vals_org[:,0];
    vals=np.array(vals_org[:,1:],dtype=int);
    # Tube.tube_id,Tube.deep_features_idx,TubeHash.hash_table,TubeHash.hash_val
    tube_ids=np.unique(vals[:,0])    
    # deep_features_idx=np.unique(vals[:,1]);
    tube_scores_all={};
    for tube_id in tube_ids:
        # print tube_id,len(deep_features_idx),len(np.unique(vals[vals[:,0]==tube_id,1]));
        deep_features_idx=np.unique(vals[vals[:,0]==tube_id,1])
        tube_scores_all[tube_id]=getTubeScoresMat(tube_id,vals,hash_bin_scores,deep_features_idx,num_hash_tables);
    # print time.time()-t;

    # for tube_id in tube_scores_all:
    #     tube_scores=tube_scores_all[tube_id];
    #     print tube_id,np.sum(np.isnan(tube_scores)),tube_scores.shape
    #     # print tube_scores[0,:]
    #     assert np.sum(np.isnan(tube_scores))==0;

    # out_file_temp='/disk2/temp/temp.p';
    # np.savez_compressed(out_file_scores,tube_scores_all.values(),tube_scores_all.keys())
    pickle.dump(tube_scores_all,open(out_file_scores,'wb'));

def visualizeBestTubeRank(params):
    class_labels_map = params.class_labels_map
    rel_path = params.rel_path
    out_file_html = params.out_file_html
    out_dir = params.out_dir
    score_info_file = params.score_info_file
    
    [class_labels,class_idx_map]=zip(*class_labels_map);
    class_labels=list(class_labels);
    class_idx_map=list(class_idx_map);
    [score_files,score_files_info]=pickle.load(open(score_info_file,'rb'));
    class_idx=np.unique(score_files_info[:,0]);
    tubes=np.unique(score_files_info[:,3])
    best_tubes_overall=score_files_info[:,3];
    out_file=os.path.join(out_dir,'overall.png');
    visualize.hist(best_tubes_overall,out_file,bins=tubes,normed=True,xlabel='Tube_idx',ylabel='Frequency',title='Best Tube Over All',cumulative=False)

    img_paths=[];
    captions=[];
    for class_idx_curr in class_idx:

        label=class_labels[class_idx_map.index(class_idx_curr)]
        out_file=os.path.join(out_dir,label+'.png');
        img_paths.append([out_file.replace(rel_path[0],rel_path[1])]);
        captions.append([label]);
        rel_tubes=score_files_info[score_files_info[:,0]==class_idx_curr,3];
        print class_idx_curr,rel_tubes.shape,min(rel_tubes),max(rel_tubes);
        visualize.hist(rel_tubes,out_file,bins=tubes,normed=True,xlabel='Tube Idx',ylabel='Frequency',title='Best Tube '+label,cumulative=False)        
    # visualize.save

    visualize.writeHTML(out_file_html,img_paths,captions,400,400);

def saveAverageScoresInfo(score_files,out_file):

    score_files_info=np.empty((len(score_files),9),dtype=int);
    for idx,file_curr in enumerate(score_files):
        file_name=file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')];
        
        class_idx=int(file_name[:file_name.index('_')]);
        video_id=int(file_name[file_name.index('_')+1:file_name.rindex('_')]);
        shot_id=int(file_name[file_name.rindex('_')+1:]);

        scores=pickle.load(open(file_curr,'rb'));

        tube_ids=scores.keys();
        tube_scores=[np.mean(scores[tube_id]) for tube_id in tube_ids];

        max_scores_idx =[np.argmax(np.mean(scores[tube_id],axis=1)) for tube_id in tube_ids];
        max_scores =[np.max(np.mean(scores[tube_id],axis=1)) for tube_id in tube_ids];

        min_scores_idx =[np.argmin(np.mean(scores[tube_id],axis=1)) for tube_id in tube_ids];
        min_scores =[np.min(np.mean(scores[tube_id],axis=1)) for tube_id in tube_ids];

        best_tube=tube_ids[np.argmax(tube_scores)]
        best_tube_idx=np.argmax(tube_scores)
        best_tube_best_frame_idx=max_scores_idx[best_tube_idx];

        best_frame_tube=tube_ids[np.argmax(max_scores)]
        best_frame_idx=max_scores_idx[np.argmax(max_scores)];

        worst_frame_tube=tube_ids[np.argmin(min_scores)]
        worst_frame_idx=min_scores_idx[np.argmin(min_scores)]

        score_files_info[idx,0]=class_idx;
        score_files_info[idx,1]=video_id;
        score_files_info[idx,2]=shot_id;
        score_files_info[idx,3]=best_tube;
        score_files_info[idx,4]=best_tube_best_frame_idx
        score_files_info[idx,5]=best_frame_tube
        score_files_info[idx,6]=best_frame_idx
        score_files_info[idx,7]=worst_frame_tube
        score_files_info[idx,8]=worst_frame_idx

    pickle.dump([score_files,score_files_info],open(out_file,'wb'));

def getNRankedPatches(list_scores,list_files,num_to_display):
    # print len(list_scores),len(list_files);
    
    assert len(list_scores)==len(list_files)
    num_patches=len(list_files);
    list_scores=np.array(list_scores);
    sort_idx=np.argsort(list_scores)[::-1];
    list_scores_sorted=list_scores[sort_idx]
    list_files=np.array(list_files);
    list_files_sorted=list_files[sort_idx];
    idx_display=np.linspace(0,num_patches-1,num_to_display,dtype=int);
    img_paths=list(list_files_sorted[idx_display]);
    scores_picked=list_scores_sorted[idx_display];
    return img_paths,scores_picked,idx_display

def visualizeRankedPatchesPerClass(class_score_info,num_to_display,out_file_html,rel_path,height_width):
    
    img_paths_all=[];
    captions_all=[];
    
    for selected_class,class_label,out_file in class_score_info:
        [list_scores,list_files] = pickle.load(open(out_file,'rb'));
        num_patches=len(list_files);
        img_paths,scores_picked,idx_display=getNRankedPatches(list_scores,list_files,num_to_display)
        img_paths=[img_path.replace(rel_path[0],rel_path[1]) for img_path in img_paths];
        captions=[];
        for idx_idx_curr,idx_curr in enumerate(idx_display):
            score_curr=round(scores_picked[idx_idx_curr],5)
            caption_curr=str(idx_idx_curr)+' Rank '+str(idx_curr+1)+' of '+str(num_patches)+' Score: '+str(score_curr);
            caption_curr=class_label+' '+caption_curr;
            captions.append(caption_curr);
        # captions=[[list_scores[sort_idx[idx_curr]],5))] for idx_idx_curr,idx_curr in enumerate(idx_display)]
        # print captions
        img_paths_all.append(img_paths);
        captions_all.append(captions);

    img_paths_all=np.array(img_paths_all).T
    captions_all=np.array(captions_all).T
    visualize.writeHTML(out_file_html,img_paths_all,captions_all,height_width[0],height_width[0]);
    print out_file_html

def getListScoresAndPatches_multiProc((idx_file_curr,file_curr,class_label,path_to_patches)):
    list_scores=[];
    list_files=[];
    print idx_file_curr,file_curr
    file_name=file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')];
    class_idx=int(file_name[:file_name.index('_')]);
    video_id=int(file_name[file_name.index('_')+1:file_name.rindex('_')]);
    shot_id=int(file_name[file_name.rindex('_')+1:]);

    scores = pickle.load(open(file_curr,'rb'));

    shot_string = class_label
    shot_string = os.path.join(path_to_patches,shot_string+'_'+str(video_id)+'_'+str(shot_id));
    
    for tube_id in scores:
        scores_curr = scores[tube_id];
        scores_curr = np.mean(scores_curr,axis=1);
        list_scores.extend(list(scores_curr))
        
        txt_file = os.path.join(path_to_patches,shot_string,str(tube_id),str(tube_id)+'.txt');
        patch_paths = util.readLinesFromFile(txt_file)
        assert len(patch_paths)==scores_curr.shape[0]
        list_files.extend(patch_paths);
    return (list_scores,list_files)

def getListScoresAndPatches(score_files,class_label,path_to_patches,n_jobs=None):
    # print len(score_files);
    list_scores=[];
    list_files=[];
    args=[];

    for idx_file_curr,file_curr in enumerate(score_files):
        args.append((idx_file_curr,file_curr,class_label,path_to_patches));

    if n_jobs is None:
        for arg in args:
            list_scores_curr,list_files_curr=getListScoresAndPatches_multiProc(arg);
            list_scores.append(list_scores_curr);
            list_files.append(list_files_curr);
    else:
        p=multiprocessing.Pool(n_jobs);
        results=p.map(getListScoresAndPatches_multiProc,args);
        [list_scores,list_files]=zip(*results);

    list_scores=list(itertools.chain.from_iterable(list_scores))
    list_files=list(itertools.chain.from_iterable(list_files))
        # print idx_file_curr,file_curr
        # file_name=file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')];
        # class_idx=int(file_name[:file_name.index('_')]);
        # video_id=int(file_name[file_name.index('_')+1:file_name.rindex('_')]);
        # shot_id=int(file_name[file_name.rindex('_')+1:]);

        # scores = pickle.load(open(file_curr,'rb'));

        # shot_string = class_label
        # # class_labels[class_idx_all.index(class_idx)];
        # shot_string = os.path.join(path_to_patches,shot_string+'_'+str(video_id)+'_'+str(shot_id));
        
        # for tube_id in scores:
        #     scores_curr = scores[tube_id];
        #     scores_curr = np.mean(scores_curr,axis=1);
        #     list_scores.extend(list(scores_curr))
            
        #     txt_file = os.path.join(path_to_patches,shot_string,str(tube_id),str(tube_id)+'.txt');
        #     patch_paths = util.readLinesFromFile(txt_file)
        #     assert len(patch_paths)==scores_curr.shape[0]
        #     list_files.extend(patch_paths);

    return list_scores,list_files

def script_fixHorseCountError(path_to_record):
    record=pickle.load(open(path_to_record,'rb'))
    [txt_files,score_files,tube_ids,scores_size,patch_paths_size]=zip(*record);

    for idx,t in enumerate(txt_files):
        score_file=score_files[idx];
        tube_id=tube_ids[idx]
        
        scores_all=pickle.load(open(score_file,'rb'));
        scores=scores_all[tube_id];
        
        scores_size_curr=scores_size[idx];
        patch_paths_size_curr=patch_paths_size[idx];
        diff=scores_size_curr-patch_paths_size_curr

        top = scores[:patch_paths_size_curr,:]
        bottom =scores[patch_paths_size_curr:,:];
        

        top_nan_count=np.sum(np.isnan(top))
        bottom_nan_count=np.sum(np.isnan(bottom))

        assert diff>0;
        assert top.shape[0]+bottom.shape[0]==scores.shape[0];
        assert bottom.shape[0]==diff;
        assert bottom_nan_count==bottom.size
        assert bottom_nan_count==diff*num_hash_tables
        assert top_nan_count==0;

        scores_all[tube_id]=top;
        pickle.dump(scores_all,open(score_file,'wb'));

def saveRecordOfCountErrorFiles(score_files,class_label,path_to_patches,out_file):
    record=[];
    for idx_file_curr,file_curr in enumerate(score_files):
        print idx_file_curr,len(score_files)
        file_name=file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')];
        class_idx=int(file_name[:file_name.index('_')]);
        video_id=int(file_name[file_name.index('_')+1:file_name.rindex('_')]);
        shot_id=int(file_name[file_name.rindex('_')+1:]);

        scores = pickle.load(open(file_curr,'rb'));

        shot_string = class_label
        # class_labels[class_idx_all.index(class_idx)];
        shot_string = os.path.join(path_to_patches,shot_string+'_'+str(video_id)+'_'+str(shot_id));
        
        for tube_id in scores:
            scores_curr = scores[tube_id];
            txt_file = os.path.join(path_to_patches,shot_string,str(tube_id),str(tube_id)+'.txt');
            patch_paths = util.readLinesFromFile(txt_file)
            if scores_curr.shape[0]!=len(patch_paths):
                print 'PROBLEM'
                record.append((txt_file,file_curr,tube_id,scores_curr.shape[0],len(patch_paths)));
    print len(record)
    pickle.dump(record,open(out_file,'wb'));

def getPatchInfoFromPath(path):
    path_broken=path.split('/');
    video_broken=path_broken[-3].split('_');
    class_label=video_broken[0];
    video_id=int(video_broken[1]);
    shot_id=int(video_broken[2]);
    tube_id=int(path_broken[-2]);
    frame_id=path_broken[-1]
    frame_id=int(frame_id[:frame_id.rindex('.')]);
    return class_label,video_id,shot_id,tube_id,frame_id

def script_verifyRecordedScoreMatchesDBScore(params):
    
    path_to_db = params.path_to_db
    path_to_hash = params.path_to_hash
    total_class_counts = params.total_class_counts
    img_path = params.img_path
    class_label = params.class_label
    video_id = params.video_id
    shot_id = params.shot_id
    class_idx = params.class_idx
    score_file = params.score_file
    
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    
    toSelect=(Tube.idx,);
    criterion=(Tube.class_idx_pascal==class_idx,Tube.video_id==video_id,Tube.shot_id==shot_id);
    total_shot_patches=mani.count(toSelect,criterion);
    
    #get patch id
    hash_info_patch=getHashInfoForImg(path_to_db,img_path);
    # patch_id=mani.select((Tube.idx,),(Tube.img_path==img_path,));
    # assert len(patch_id)==1;
    # patch_id=patch_id[0][0];
    
    # #get hash vals
    # mani_hash=TubeHash_Manipulator(path_to_db);
    # mani_hash.openSession();
    # toSelect=(TubeHash.hash_table,TubeHash.hash_val)
    # criterion=(TubeHash.idx==patch_id,);
    # hash_info_patch=mani_hash.select(toSelect,criterion);
    
    #get hash_info of all patches in shot
    criterion=(Tube.class_idx_pascal==class_idx,Tube.video_id==video_id,Tube.shot_id==shot_id)
    hash_info_all=mani_hash.selectMix(toSelect,criterion);
    
    mani_hash.closeSession();
    mani.closeSession()

    hash_info_all=list(hash_info_all);
    hash_scores_patch=[];
    for idx_hash_info,hash_info_curr in enumerate(hash_info_patch):

        hash_file_curr=str(hash_info_curr[0])+'_'+str(hash_info_curr[1])+'_counts.p';
        hash_file_curr=os.path.join(path_to_hash,hash_file_curr);
        hash_bin_class_counts=pickle.load(open(hash_file_curr,'rb'));
        hash_bin_class_count=hash_bin_class_counts[class_idx];

        numo=hash_bin_class_count-hash_info_all.count(hash_info_curr);
        deno=total_class_counts[class_idx]-total_shot_patches;
        hash_scores_patch.append(numo/float(deno));

    score_db=np.mean(hash_scores_patch);
    print len(hash_scores_patch),score_db,score_file
    assert np.isclose(score_db,score_file);

def saveTotalClassBreakdowns(path_to_db,out_file):
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    toSelect=(Tube.class_idx_pascal,Tube.video_id,Tube.shot_id,Tube.tube_id);
    vals=mani.select(toSelect,distinct=True);
    mani.closeSession();
    vals=np.array(vals);
    class_idx_db=vals[:,0];
    ids_db=vals[:,1:];

    column_names=['video','shot','tube'];
    counts=getClassCountsByIdType(class_idx_db,ids_db,column_names)
    pickle.dump(counts,open(out_file,'wb'));

def getClassCountsByIdType(class_idx_db,ids_db,column_names):
    counts={};
    class_idx_all=np.unique(class_idx_db);
    for column_idx,column_name in enumerate(column_names):
        counts[column_name]={};
        for class_idx in class_idx_all:
            rel_rows=class_idx_db==class_idx
            rel_cols=ids_db[rel_rows,:column_idx+1];
            unique_rows=np.vstack({tuple(row) for row in rel_cols})
            counts[column_name][class_idx]=unique_rows.shape[0];
    return counts

def getHashBinClassBreakdowns((hash_table,hash_val,path_to_db,out_file,idx)):
    print idx
    mani=Tube_Manipulator(path_to_db)
    mani.openSession();
    toSelect=(Tube.class_idx_pascal,Tube.video_id,Tube.shot_id,Tube.tube_id);
    criterion=(TubeHash.hash_table==hash_table,TubeHash.hash_val==hash_val);
    vals=mani.selectMix(toSelect,criterion=criterion,distinct=True);
    mani.closeSession();
    vals=np.array(vals);
    class_idx_db=vals[:,0];
    ids_db=vals[:,1:];

    column_names=['video','shot','tube'];
    counts=getClassCountsByIdType(class_idx_db,ids_db,column_names)
    # for k in counts.keys():
    #     for k2 in counts[k].keys():
    #         print k,k2,counts[k][k2];

    # return counts
    pickle.dump(counts,open(out_file,'wb'));

def verifyTotalClassBreakdowns(path_to_db,out_file):
    counts=pickle.load(open(out_file,'rb'));
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    for class_idx in range(10):
        print class_idx

        toSelect=(Tube.video_id,);
        criterion=(Tube.class_idx_pascal==class_idx,);
        count_video=mani.count(toSelect,criterion,distinct=True);
        toSelect=(Tube.video_id,Tube.shot_id);
        count_shot=mani.count(toSelect,criterion,distinct=True);
        toSelect=(Tube.video_id,Tube.shot_id,Tube.tube_id);
        count_tube=mani.count(toSelect,criterion,distinct=True);

        print counts['video'][class_idx],count_video,
        print counts['shot'][class_idx],count_shot,
        print counts['tube'][class_idx],count_tube

        assert counts['video'][class_idx]==count_video
        assert counts['shot'][class_idx]==count_shot
        assert counts['tube'][class_idx]==count_tube

    mani.closeSession();

def getAllClassesShotScoresByImgPath():

    path_to_db = 'sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    out_dir='/disk2/januaryExperiments/class_breakdowns';
    num_hash_tables=32;
    total_counts_file=os.path.join(out_dir,'total_counts_breakdown.p');
    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)];
    # [class_labels,class_idx_all]=zip(*class_labels_map);

    # print 'best_dog'
    # img_path='/disk2/res11/tubePatches/dog_21_19/6/63.jpg'
    print 'not_best_dog'
    img_path='/disk2/res11/tubePatches/dog_35_1/0/33.jpg';
    class_idx=6;
    type_count='shot';

    total_counts=pickle.load(open(total_counts_file,'rb'));

    vals=getHashInfoForImg(path_to_db,img_path);
    # mani=Tube_Manipulator(path_to_db);
    # mani.openSession();
    # [(db_idx,)]=mani.select((Tube.idx,),(Tube.img_path==img_path,));
    # mani.closeSession();
    # mani=TubeHash_Manipulator(path_to_db);
    # mani.openSession();
    # vals=mani.select((TubeHash.hash_table,TubeHash.hash_val),(TubeHash.idx==db_idx,));
    # mani.closeSession();

    scores={};

    for class_label,class_idx_assume in class_labels_map:
        totals_in_bins=[]
        for hash_table,hash_val in vals:
            file_curr=str(hash_table)+'_'+str(hash_val)+'_counts.p';
            file_curr=os.path.join(out_dir,file_curr);
            counts=pickle.load(open(file_curr,'rb'));
            if class_idx_assume==class_idx:
                counts_numo=counts[type_count][class_idx_assume]-1;
                counts_deno=total_counts[type_count][class_idx_assume]-1;
            else:
                counts_numo=counts[type_count][class_idx_assume];
                counts_deno=total_counts[type_count][class_idx_assume];
            score_curr=counts_numo/float(counts_deno);
            if class_idx_assume not in scores:
                scores[class_idx_assume]=[];    
            scores[class_idx_assume].append(score_curr);
            totals_in_bins.append(sum(counts[type_count].values()));
            
    print len(totals_in_bins),sum(totals_in_bins),np.mean(totals_in_bins);
    for class_label,class_idx_assume in class_labels_map:
        print class_label,class_idx_assume,len(scores[class_idx_assume]),np.mean(scores[class_idx_assume])

def saveHashBinFrameCountsAll(out_file,hash_dir,class_idx_all,num_hash_tables,num_hash_vals,layer=None):
    hash_counts_all=np.zeros(shape=(num_hash_tables*num_hash_vals,len(class_idx_all)),dtype=int);
    hash_counts_all_keys=[];
    print hash_counts_all.shape

    for hash_table in range(num_hash_tables):
        for hash_val in range(num_hash_vals):
            hash_counts_all_keys.append((hash_table,hash_val));
            idx_curr=(hash_table*num_hash_vals)+hash_val
            # print idx_curr
            file_curr=str(hash_table)+'_'+str(hash_val)+'_counts.p';
            file_curr=os.path.join(hash_dir,file_curr);
            counts=pickle.load(open(file_curr,'rb'));
            
            if layer is not None:
                counts=counts[layer]

            for key_curr in counts:
                hash_counts_all[idx_curr,key_curr]=counts[key_curr];

    pickle.dump([hash_counts_all_keys,hash_counts_all],open(out_file,'wb'));

def getHashInfoForImg(path_to_db,img_path):
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    
    #get patch id
    patch_id=mani.select((Tube.idx,),(Tube.img_path==img_path,));
    assert len(patch_id)==1;
    patch_id=patch_id[0][0];
    mani.closeSession();

    #get hash vals
    mani_hash=TubeHash_Manipulator(path_to_db);
    mani_hash.openSession();
    toSelect=(TubeHash.hash_table,TubeHash.hash_val)
    criterion=(TubeHash.idx==patch_id,);
    hash_info_patch=mani_hash.select(toSelect,criterion);
    mani_hash.closeSession();
    return hash_info_patch
    
def script_saveNpzScorePerShot_normalized(params):
    path_to_db = params['path_to_db']
    file_binCounts = params['file_binCounts']
    class_idx =  params['class_idx']
    video_id =  params['video_id']
    shot_id =  params['shot_id']
    out_file_scores =  params['out_file_scores']
    num_hash_tables = params['num_hash_tables']
    total_counts = params['total_class_counts']
    
    class_idx_assume = params.get('class_idx_assume',None);
    if class_idx_assume is None:
        class_idx_assume=class_idx;

    print params['idx']

    mani=Tube_Manipulator(path_to_db);

    mani.openSession();
    toSelect=(Tube.tube_id,Tube.deep_features_idx,TubeHash.hash_table,TubeHash.hash_val);
    criterion=(Tube.video_id==video_id,Tube.class_idx_pascal==class_idx,Tube.shot_id==shot_id);
    vals=mani.selectMix(toSelect,criterion);
    total_frames = getShotFrameCount(mani,class_idx,video_id,shot_id);
    mani.closeSession();

    hash_count_keys,hash_counts=pickle.load(open(file_binCounts,'rb'));

    hash_info=[tuple(r) for r in vals[:,2:]];
    hash_counts = dict(Counter(hash_info))
    # total_counts=np.sum(hash_counts,axis=0);

    scores_all={};
    vals=np.array(vals)
    tube_ids_uni=np.unique(vals[:,0]);

    for tube_id in tube_ids_uni:
        vals_rel=vals[vals[:,0]==tube_id,1:];
        deep_features_idx_uni = np.unique(vals_rel[:,0]);
        scores_tube=np.empty((len(deep_features_idx_uni),num_hash_tables));
        scores_tube[:]=np.nan;

        for deep_features_idx in deep_features_idx_uni:
            hash_info=vals_rel[vals_rel[:,0]==deep_features_idx,1:];
            
            assert len(hash_info)==num_hash_tables

            scores=[];
            
            for hash_info_curr in hash_info:
                idx_curr=hash_count_keys.index(tuple(hash_info_curr));
                counts_curr=hash_counts[idx_curr,:];
                deno=counts_curr/total_counts.astype(dtype=float);
                numo=deno[class_idx_assume];
                deno=sum(deno);
                score_curr=numo/float(deno);
                scores.append(score_curr);
            
            scores_tube[deep_features_idx,:]=scores;

        scores_all[tube_id]=scores_tube;

    for tube_id in scores_all:
        tube_scores=scores_all[tube_id];
        assert np.sum(np.isnan(tube_scores))==0;

    pickle.dump(scores_all,open(out_file_scores,'wb'));

def meta_script_saveNpzScorePerShot_normalized(params,out_dir_scores,out_dir_actual):

    mani=TubeHash_Manipulator(params['path_to_db']);
    mani.openSession();
    vals=mani.select((Tube.class_idx_pascal,Tube.video_id,Tube.shot_id),distinct=True);
    mani.closeSession()

    print len(vals)

    args=[];
    for idx,(class_idx,video_id,shot_id) in enumerate(vals):
        params_curr = copy.deepcopy(params);
        params_curr['class_idx'] = class_idx
        params_curr['video_id'] = video_id
        params_curr['shot_id'] = shot_id
        file_curr = str(params_curr['class_idx'])+'_'+str(params_curr['video_id'])+'_'+str(params_curr['shot_id'])+'.p';
        params_curr['out_file_scores'] = os.path.join(out_dir_scores,file_curr)
        params_curr['idx'] = idx;

        if os.path.exists(params_curr['out_file_scores']):
        #     continue;
        # else:
            params_curr['out_file_scores']=os.path.join(out_dir_actual,file_curr)
            args.append(params_curr);    

    print len(args)
    # args=args[3000:];
    print len(args)
    n_jobs=12
    p = multiprocessing.Pool(min(multiprocessing.cpu_count(),n_jobs))
    p.map(script_saveNpzScorePerShot_normalized,args);


def main():
    out_dir='/disk2/januaryExperiments/shot_score_normalized_perShot_analysis';
    class_labels_map = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    [class_labels,class_idx_all]=zip(*class_labels_map);
    
    for num_to_display in [10,100,1000]:
        out_file_html=os.path.join(out_dir,'all_scores_patches_sorted_'+str(num_to_display))+'.html';
        rel_path=['/disk2','../../..'];
        height_width=[400,400];
        
        class_score_info=[];
        for class_label,class_idx in class_labels_map:
            out_file=os.path.join(out_dir,'all_scores_patches_'+class_label+'.p')
            tuple_curr=(class_idx,class_label,out_file)
            print tuple_curr
            class_score_info.append(tuple_curr);
        
        visualizeRankedPatchesPerClass(class_score_info,num_to_display,out_file_html,rel_path,height_width)

    return
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    score_dir='/disk2/januaryExperiments/shot_score_normalized_perShot';
    out_dir='/disk2/januaryExperiments/shot_score_normalized_perShot_analysis';
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);

    path_to_patches='/disk2/res11/tubePatches';
    class_labels_map = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    n_jobs=12;

    [class_labels,class_idx_all]=zip(*class_labels_map);
    
    for selected_class in range(10):
        class_label=class_labels[class_idx_all.index(selected_class)]
        out_file=os.path.join(out_dir,'all_scores_patches_'+class_label+'.p')
        score_files=[os.path.join(score_dir,file_curr) for file_curr in os.listdir(score_dir) if file_curr.endswith('.p') and file_curr.startswith(str(selected_class)+'_')];
        list_scores,list_files=getListScoresAndPatches(score_files,class_label,path_to_patches,n_jobs=n_jobs)
        pickle.dump([list_scores,list_files],open(out_file,'wb'));
    

    return
    # total_class_counts = {0: 622034, 1: 245763, 2: 664689, 3: 125286, 4: 311316, 5: 500093, 6: 889816, 7: 839481, 8: 358913, 9: 1813897}
    # total_counts=[];
    # for class_idx in range(10):
    #     total_counts.append(total_class_counts[class_idx]);
    # total_counts=np.array(total_counts)
    # print total_counts

    # params={};
    # params['total_class_counts'] = total_counts
    # params['path_to_db'] = 'sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    # params['file_binCounts'] = '/disk2/januaryExperiments/frameCounts/frameCounts_all.p';
    # params['num_hash_tables'] = 32

    # # out_dir_scores='/disk2/januaryExperiments/shot_score_normalized/';
    # # params['class_idx'] = 9
    # # params['video_id'] = 6
    # # params['shot_id'] = 15
    # # file_curr = str(params['class_idx'])+'_'+str(params['video_id'])+'_'+str(params['shot_id'])+'.p';
    # # params['out_file_scores'] = os.path.join(out_dir_scores,file_curr)
    # # params['idx'] = 0
    # # script_saveNpzScorePerShot_normalized(params)
    # out_dir_scores='/disk2/januaryExperiments/shot_score_normalized/';
    # if not os.path.exists(out_dir_scores):
    #     os.mkdir(out_dir_scores);
    # out_dir_actual='/disk1/maheen_data/shot_score_normalized/';
    # meta_script_saveNpzScorePerShot_normalized(params,out_dir_scores,out_dir_actual)

    return
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    out_dir='/disk2/januaryExperiments/class_breakdowns'
    out_file='shotCounts_all.p'
    out_file=os.path.join(out_dir,out_file);
    print out_file
    class_idx_all=range(10);
    num_hash_tables=32;
    num_hash_vals=256;
    layer='shot'
    # saveHashBinFrameCountsAll(out_file,out_dir,class_idx_all,num_hash_tables,num_hash_vals,layer=layer)
    file_curr=os.path.join(out_dir,'total_counts_breakdown.p');
    k=pickle.load(open(file_curr,'rb'));
    total_counts=[];
    for class_idx in class_idx_all:
        total_counts.append(k[layer][class_idx]);

    total_counts=np.array(total_counts);
    print total_counts,sum(total_counts)

    params={};
    params['total_class_counts'] = total_counts
    params['path_to_db'] = path_to_db
    params['file_binCounts'] = out_file
    params['num_hash_tables'] = 32
    
    out_dir_scores='/disk2/januaryExperiments/shot_score_normalized/';
    if not os.path.exists(out_dir_scores):
        os.mkdir(out_dir_scores);
    out_dir_actual='/disk1/maheen_data/shot_score_normalize_shotLevel';
    meta_script_saveNpzScorePerShot_normalized(params,out_dir_scores,out_dir_actual)
    # a=pickle.load(open(os.path.join(out_dir_scores,'2_15_48.p'),'rb'));
    # print a.keys();
    # for k in a.keys():
    #     print k,a[k].shape,np.mean(a[k]);
    # print a[0].shape,np.mean(a[0],axis=1);
    # print a[8].shape,np.mean(a[8],axis=1);


        

if __name__=='__main__':
    main();

