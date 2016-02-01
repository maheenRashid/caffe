import os;
import numpy as np;
import scipy.stats
import scipy.io
import cPickle as pickle
import copy
from scipy import misc;
import youtube;
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
import Image, ImageDraw
from functools import reduce


def createParams(type_Experiment):
    if type_Experiment == 'drawTubeBB':
        list_params=['path_to_data',
                    'score_dir',
                    'mat_dir',
                    'out_dir',
                    'to_exclude',
                    'class_labels_map',
                    'n_jobs']
        params = namedtuple('Params_drawTubeBB',list_params);
    elif type_Experiment=='saveIOUInfo':
        list_params=['class_labels_map',
                    'out_file',
                    'path_to_tube_ranks',
                    'path_to_mats',
                    'out_file_crude_info',
                    'out_file_bestOfBoth']
        params = namedtuple('Params_saveIOUInfo',list_params);
    else:
        params=None;

    return params

def script_drawTubeBB(params):
    path_to_data = params.path_to_data
    score_dir = params.score_dir
    mat_dir = params.mat_dir
    out_dir = params.out_dir
    to_exclude = params.to_exclude
    class_labels_map = params.class_labels_map
    n_jobs = params.n_jobs

    [class_labels,class_idx_all]=zip(*class_labels_map);
    files=[file_curr for file_curr in os.listdir(score_dir) if file_curr.endswith('.p')];
    
    print len(files);

    args=[];
    for file_curr in files:
        class_idx=int(file_curr[:file_curr.index('_')]);
        rest=file_curr[file_curr.index('_'):file_curr.index('.p')];
        class_label=class_labels[class_idx_all.index(class_idx)];
        mat_file=os.path.join(mat_dir,class_label+rest+'.mat');
        if mat_file in to_exclude:
            continue;
        out_file=os.path.join(out_dir,file_curr[:file_curr.index('.')]+'.png');
        if os.path.exists(out_file):
            continue

        tubes,scores=pickle.load(open(os.path.join(score_dir,file_curr),'rb'))
        tubes=tubes[::-1]
        fills=[(0,0,255)]*(len(tubes)-1)+[(255,0,0)]

        arg_curr=(mat_file,path_to_data,out_file,tubes,fills);
        args.append(tuple(arg_curr));
    
    print len(args);

    p=multiprocessing.Pool(n_jobs);
    p.map(drawTubeBB,args);


def getTubeCoordsForFrame(res,frame_id):
    num_tubes=len(res);
    coords={};

    for tube_no in range(num_tubes):
        coords[tube_no]=[];

        bboxes=res[tube_no].bboxes
        bin_idx=bboxes[:,-1]==frame_id
        
        if sum(bin_idx)==0:
            continue;
        
        bbox=bboxes[bin_idx,:4].ravel()
        # print bbox.shape
        bbox=bbox-1;
        bbox=[math.floor(x/0.4) for x in bbox];
        coords[tube_no]=bbox;
        # minx=bbox[0];miny=bbox[1];maxx=bbox[2];maxy=bbox[3];
    return coords
    
def getLineCoords(bbox):
    minx=bbox[0];miny=bbox[1];maxx=bbox[2];maxy=bbox[3];
    line_coords=[(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny),(minx,miny)];
    return line_coords

def drawTubeBB((mat_file,path_to_data,out_file,tubes,fills)):
    
    just_mat_file=mat_file[mat_file.rindex('/')+1:];

    category=just_mat_file[:just_mat_file.index('_')];
    video_id=just_mat_file[just_mat_file.index('_')+1:just_mat_file.rindex('_')];
    shot_id=just_mat_file[just_mat_file.rindex('_')+1:just_mat_file.rindex('.')];
    
    res=scipy.io.loadmat(mat_file,squeeze_me=True, struct_as_record=False);
    
    res=res['res'];
    if not hasattr(res,'__iter__'):
        res=[res];
    num_frames=res[0].bboxes.shape[0];
    num_tubes=len(res);

    frame_idx_all = tuple([res_curr.bboxes[:,-1] for res_curr in res]);
    frame_idx_all = reduce(np.intersect1d,frame_idx_all);
    
    assert frame_idx_all.size>0
    idx_idx=len(frame_idx_all)/2
    frame_id=int(frame_idx_all[idx_idx])
    print mat_file,category,video_id,shot_id,frame_id
    frame_path= youtube.getFramePath(path_to_data,category,video_id,shot_id,frame_id);
    im = Image.open(frame_path)
    draw = ImageDraw.Draw(im)

    coords=getTubeCoordsForFrame(res,frame_id);

    for idx_tube_no,tube_no in enumerate(tubes):
        bbox = coords[tube_no];

        # bboxes=res[tube_no].bboxes
        # bbox=bboxes[bboxes[:,-1]==frame_id,:4].ravel()
        # # print bbox.shape
        # bbox=bbox-1;
        # bbox=[math.floor(x/0.4) for x in bbox];
        
        if len(im.getbands())==1:
            draw.line(getLineCoords(bbox),width=1,fill=fills[idx_tube_no][0])
        else:
            draw.line(getLineCoords(bbox),width=1,fill=fills[idx_tube_no])
        
    im = np.array(im)
    misc.imsave(out_file,im);


def saveBestTubeAvgScore((score_file,out_file,idx)):
    if idx is not None:
        print idx
    scores=pickle.load(open(score_file));
    tube_keys=scores.keys();
    tube_scores=[np.mean(scores[tube_id]) for tube_id in tube_keys];
    sort_idx=np.argsort(tube_scores)[::-1];
    tubes_ranked=np.array(tube_keys)[sort_idx];
    scores_ranked=np.array(tube_scores)[sort_idx];
    # return tubes_ranked[0],tubes_ranked,scores_ranked
    pickle.dump([tubes_ranked,scores_ranked],open(out_file,'wb'));

def saveGTData(path_to_txt,class_labels_map,out_file):

    [class_labels,class_idx_all]=zip(*class_labels_map);

    sticks=util.readLinesFromFile(path_to_txt)
    print len(sticks);
    
    meta_info=[];
    coords=[];

    for stick in sticks:
        stick_split=stick.split('/');
        stick_split=[curr for curr in stick_split if curr!=''];
        class_label=stick_split[3];
        class_idx=class_idx_all[class_labels.index(class_label)];
        video_id=int(stick_split[5]);
        shot_id=int(stick_split[7]);
        frame_id=stick_split[-1];
        frame_id=int(frame_id[:frame_id.index('.')].strip('frame'));
        meta_info_curr=[class_idx,video_id,shot_id,frame_id]
        res=scipy.io.loadmat(stick,squeeze_me=True, struct_as_record=False);
        boxes=res['coor'];
        if not hasattr(boxes[0],'__iter__'):
            boxes=[boxes];
        else:
            print 'found it!',boxes

        for box in boxes:
            meta_info.append(meta_info_curr);
            coords.append(list(box));

    pickle.dump([meta_info,coords],open(out_file,'wb'));

def getTubeGTOverlap(gt_box,tube_coords):

    # tube_coords_keys=tube_coords.keys();
    tube_coords_keys=[]
    for k in tube_coords.keys():
        if len(tube_coords[k])>0:
            tube_coords_keys.append(k);

    # for gt_box in gt_coords:
    ious=[];
    for tube_id in tube_coords_keys:    
        ious.append(util.getIOU(gt_box,tube_coords[tube_id]));
    
    ious=np.array(ious);
    sort_idx=np.argsort(ious)[::-1];
    ious_sorted=ious[sort_idx];
    tubes_sorted=[tube_coords_keys[idx_curr] for idx_curr in sort_idx]
    return ious_sorted,tubes_sorted


def getBestOfBothRank(ious,tubes_gt,tubes_ranked):
    # print tubes_sorted

    if tubes_ranked[0] not in tubes_gt:
        gt_idx_best_rank=None
        iou_best_rank=None
    else:
        gt_idx_best_rank=tubes_gt.index(tubes_ranked[0])
        iou_best_rank=ious[gt_idx_best_rank]

    # print gt_idx_best_rank,ious[gt_idx_best_rank]

    assert tubes_gt[0] in tubes_ranked
    rank_idx_best_gt=tubes_ranked.index(tubes_gt[0]);
    iou_best_gt=ious[0]
    # print rank_idx_best_gt,iou_best_gt
    return gt_idx_best_rank,iou_best_rank,rank_idx_best_gt,iou_best_gt



def script_saveIOUInfo(params):
    class_labels_map = params.class_labels_map
    out_file = params.out_file
    path_to_tube_ranks = params.path_to_tube_ranks
    path_to_mats = params.path_to_mats
    out_file_crude_info = params.out_file_crude_info
    out_file_bestOfBoth = params.out_file_bestOfBoth

    [class_labels,class_idx_all]=zip(*class_labels_map);
    [meta_info,coords] = pickle.load(open(out_file,'rb'));

    ious_all=[];
    tubes_gt_all=[];
    best_of_both_all=[];
    meta_info_rec = [];

    for idx_gt in range(len(meta_info)):
    # [40]:
    # range(len(meta_info)):
        # [40]:
        print idx_gt
        meta_curr=meta_info[idx_gt];
        gt_box=coords[idx_gt]
        
        meta_curr_str=[str(curr) for curr in meta_curr];
        class_label=class_labels[class_idx_all.index(meta_curr[0])];
        tube_file=os.path.join(path_to_tube_ranks,'_'.join(meta_curr_str[:3])+'.p')
        mat_file=os.path.join(path_to_mats,'_'.join([class_label]+meta_curr_str[1:3])+'.mat');
        if not os.path.exists(tube_file):
            continue;
        # print mat_file

        res=scipy.io.loadmat(mat_file,squeeze_me=True, struct_as_record=False)['res']
        tube_coords=getTubeCoordsForFrame(res,meta_curr[-1])
        # print 'tube_coords'
        # print tube_coords
        ious,tubes_gt = getTubeGTOverlap(gt_box,tube_coords)
        tubes_ranked,tubes_scores=pickle.load(open(tube_file,'rb'));
        tubes_ranked=list(tubes_ranked);
        # print 'tubes_ranked'
        # print tubes_ranked
        # print 'tubes_gt,ious'
        # print tubes_gt,ious
        if len(tubes_gt)==0:
            gt_idx_best_rank =None; 
            iou_best_rank =None; 
            rank_idx_best_gt =None; 
            iou_best_gt =None; 
        else:
            gt_idx_best_rank,iou_best_rank,rank_idx_best_gt,iou_best_gt = getBestOfBothRank(ious,tubes_gt,tubes_ranked)
        
        meta_info_rec.append(meta_curr)
        ious_all.append(ious);
        tubes_gt_all.append(tubes_gt);

        best_of_both_all.append([gt_idx_best_rank,iou_best_rank,rank_idx_best_gt,iou_best_gt]);

    print len(meta_info),len(best_of_both_all),len(tubes_gt_all),len(ious_all)
    # print best_of_both_all
    # [:10];
    pickle.dump([meta_info_rec,best_of_both_all],open(out_file_bestOfBoth,'wb'))
    pickle.dump([meta_info_rec,ious_all,tubes_gt_all],open(out_file_crude_info,'wb'))


    # print tubes_ranked
    # print tube_coords.keys();

def getAverageIOUPerClassBOB(out_file_bestOfBoth,class_labels_map):
    [meta_info_rec,best_of_both_all] = pickle.load(open(out_file_bestOfBoth,'rb'))
    meta_info_rec=np.array(meta_info_rec);
    print meta_info_rec.shape
    best_of_both_all=np.array(best_of_both_all,dtype=float);
    print best_of_both_all.shape

    nan_bin=np.sum(np.isnan(best_of_both_all),axis=1);
    print nan_bin.shape
    meta_info_rec = meta_info_rec[nan_bin==0,:];
    best_of_both_all = best_of_both_all[nan_bin==0,:];
    print meta_info_rec.shape,best_of_both_all.shape

    avg_iou_pred={};
    avg_iou_best={};

    for class_label,class_idx in class_labels_map:
        idx_rel=meta_info_rec[:,0]==class_idx;
        iou_pred=np.mean(best_of_both_all[idx_rel,1]);
        iou_best=np.mean(best_of_both_all[idx_rel,3]);
        avg_iou_pred[class_label]=iou_pred;
        avg_iou_best[class_label]=iou_best;
    avg_iou_pred['all']=np.mean(best_of_both_all[:,1]);
    avg_iou_best['all']=np.mean(best_of_both_all[:,3]);

    return avg_iou_pred,avg_iou_best

def getAverageIOUBestFanyi(out_file_crude,class_labels_map):
    [meta_info_rec, ious_all,tubes_gt_all] = pickle.load(open(out_file_crude,'rb'));
    zero_iou=[];
    for ious_curr,tubes_curr in zip(ious_all,tubes_gt_all):
        if 0 in tubes_curr:
            zero_iou.append(ious_curr[tubes_curr.index(0)])
        else:
            zero_iou.append(None);
    meta_info_rec=np.array(meta_info_rec);
    zero_iou=np.array(zero_iou,dtype=float);
    nan_bin=np.isnan(zero_iou);
    meta_info_rec = meta_info_rec[nan_bin==0,:];
    zero_iou = zero_iou[nan_bin==0];
    
    avg_iou_pred={};
    for class_label,class_idx in class_labels_map:
        idx_rel=meta_info_rec[:,0]==class_idx;
        iou_pred=np.mean(zero_iou[idx_rel]);
        avg_iou_pred[class_label]=iou_pred;
    avg_iou_pred['all']=np.mean(zero_iou);

    return avg_iou_pred
    

def main():
    out_file_crude='/disk2/januaryExperiments/tube_scoring/iou_crude_info.p';
    out_file_bestOfBoth='/disk2/januaryExperiments/tube_scoring/iou_bestOfBoth.p';
    out_file_bestOfBoth_perShot='/disk2/januaryExperiments/tube_scoring/iou_bestOfBoth_perShot.p';
    out_file_file='/disk2/januaryExperiments/tube_scoring/iou_crude_info.p';

    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    
    avg_iou_pred,avg_iou_best = getAverageIOUPerClassBOB(out_file_bestOfBoth,class_labels_map)
    avg_iou_pred_perShot,_ = getAverageIOUPerClassBOB(out_file_bestOfBoth_perShot,class_labels_map)
    avg_iou_fanyi = getAverageIOUBestFanyi(out_file_crude,class_labels_map);
    
    dict_vals={};
    label_keys=['Fanyi','Shot','Frame','Best']
    avg_ious=[avg_iou_fanyi,avg_iou_pred_perShot,avg_iou_pred,avg_iou_best]
    xtick_labels=avg_iou_pred.keys();
    print xtick_labels

    for k in avg_iou_pred:
        for idx in range(len(label_keys)):
            if label_keys[idx] in dict_vals:
                dict_vals[label_keys[idx]].append(avg_ious[idx][k])
            else:
                dict_vals[label_keys[idx]]=[avg_ious[idx][k]]
        # dict_vals['Shot']=avg_iou_fanyi[k]
        # dict_vals['Frame']=avg_iou_fanyi[k]
        # dict_vals['Best']=avg_iou_fanyi[k]
        print k,avg_iou_fanyi[k],avg_iou_pred_perShot[k],avg_iou_pred[k],avg_iou_best[k]

    out_file='/disk2/januaryExperiments/tube_scoring/avg_iou_comparison.png';
    
    colors=['r','g','b','y']
    visualize.plotGroupBar(out_file,dict_vals,xtick_labels,label_keys,colors,ylabel='Average IOU',title='Average IOU Comparison',width=0.25,ylim=[0.2,0.9])


    return
    # path_to_txt='/disk2/youtube/categories/gt_list.txt';
    params_dict={};
    params_dict['class_labels_map']=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    params_dict['out_file']='/disk2/januaryExperiments/tube_scoring/gt_data.p';
    
    params_dict['path_to_tube_ranks'] = '/disk2/januaryExperiments/tube_scoring/scores_perShot';
    params_dict['path_to_mats'] = '/disk2/res11';
    params_dict['out_file_crude_info']='/disk2/januaryExperiments/tube_scoring/iou_crude_info_perShot.p';
    params_dict['out_file_bestOfBoth']='/disk2/januaryExperiments/tube_scoring/iou_bestOfBoth_perShot.p';
    params=createParams('saveIOUInfo');
    params=params(**params_dict);
    
    script_saveIOUInfo(params)

    return
    # path_to_data = '/disk2/youtube/categories'
    # saveGTData(path_to_txt,class_labels_map,out_file)

    # return
    
    [class_labels,class_idx_all]=zip(*class_labels_map);
    [meta_info,coords] = pickle.load(open(out_file,'rb'));

    meta_curr=meta_info[0];
    gt_box=coords[0]
    meta_curr_str=[str(curr) for curr in meta_curr];
    class_label=class_labels[class_idx_all.index(meta_curr[0])];
    tube_file=os.path.join(path_to_tube_ranks,'_'.join(meta_curr_str[:3])+'.p')
    mat_file=os.path.join(path_to_mats,'_'.join([class_label]+meta_curr_str[1:3])+'.mat');


    res=scipy.io.loadmat(mat_file,squeeze_me=True, struct_as_record=False)['res']
    tube_coords=getTubeCoordsForFrame(res,meta_curr[-1])
    
    ious,tubes_gt = getTubeGTOverlap(gt_box,tube_coords)
    tubes_ranked,tubes_scores=pickle.load(open(tube_file,'rb'));
    tubes_ranked=list(tubes_ranked);

    gt_idx_best_rank,iou_best_rank,rank_idx_best_gt,iou_best_gt = getBestOfBothRank(ious,tubes_gt,tubes_ranked)
    print gt_idx_best_rank,iou_best_rank,rank_idx_best_gt,iou_best_gt
    print tubes_ranked

    return
    frame_path = youtube.getFramePath(path_to_data,class_label,meta_curr[1],meta_curr[2],meta_curr[3])

    im = Image.open(frame_path)
    draw = ImageDraw.Draw(im)

    for idx_tube,tube in enumerate(tubes_ranked):
        draw.line(getLineCoords(tube_coords[tube]),width=2,fill=(255,255,0));        
    draw.line(getLineCoords(tube_coords[tubes_ranked[0]]),width=2,fill=(255,0,0));

    # for gt_box in gt_coords:
        # print gt_box
    draw.line(getLineCoords(gt_box),width=2,fill=(0,255,0));

    out_file='/disk2/temp/temp.png';
    misc.imsave(out_file,np.array(im));

    # gt_boxes=coords[0];



        # break
        # video_id=;
        # shot_id=;
        # frame_id=;

        # 

        # print res['coor']
        # break
    # print sticks[0];
    # videos=[vid for vid in os.listdir(path) if os.path.isdir(os.path.join(path,vid))];
    # for vid in videos:
    #     path_vid = os.path.join(path,vid);
    #     shots=[shot for shot in os.listdir(path_vid) if os.path.isdir(os.path.join(path_vid,shot))];
    #     for shot in shots:
    #         path_shot = os.path.join(path_vid,shot);


    return
    meta_dir='/disk2/januaryExperiments/tube_scoring'
    out_file_html_pre='/disk2/januaryExperiments/tube_scoring/best_tubes_comparison';
    rel_path=['/disk2','../../..'];
    paths_to_im=[os.path.join(meta_dir,'images'),os.path.join(meta_dir,'images_perShot')]
    height_width=[500,800]
    columns=['Frame Level','Shot Level']
    for class_idx in range(10):
        out_file_html=out_file_html_pre+'_'+str(class_idx)+'.html';
        files=[file_curr for file_curr in os.listdir(paths_to_im[0]) if file_curr.endswith('.png') and file_curr.startswith(str(class_idx)+'_')];
        img_paths=[];
        captions=[];
        for file_curr in files:
            img_paths_row=[];
            captions_row=[];
            for idx_path_to_im,path_to_im in enumerate(paths_to_im):
                im_curr=os.path.join(path_to_im,file_curr).replace(rel_path[0],rel_path[1]);
                img_paths_row.append(im_curr);
                captions_row.append(columns[idx_path_to_im]+' '+file_curr);
            img_paths.append(img_paths_row);
            captions.append(captions_row);
        visualize.writeHTML(out_file_html,img_paths,captions,height_width[0],height_width[1])

    return 
    params_dict={};
    params_dict['path_to_data']='/disk2/youtube/categories'
    params_dict['score_dir']='/disk2/januaryExperiments/tube_scoring/scores_perShot';
    params_dict['mat_dir']='/disk2/res11';
    params_dict['out_dir']='/disk2/januaryExperiments/tube_scoring/images_perShot'
    if not os.path.exists(params_dict['out_dir']):
        os.mkdir(params_dict['out_dir']);
    params_dict['to_exclude'] = ['/disk2/res11/horse_7_16.mat','/disk2/res11/horse_7_14.mat','/disk2/res11/horse_4_49.mat']
    params_dict['class_labels_map'] = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    params_dict['n_jobs']=12

    params=createParams('drawTubeBB');
    params=params(**params_dict);

    script_drawTubeBB(params)

    return
    shot_dir='/disk2/januaryExperiments/shot_score_normalized_perShot';
    out_dir='/disk2/januaryExperiments/tube_scoring/scores_perShot';
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    
    class_labels_map = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    n_jobs=12

    [class_labels,class_idx_all]=zip(*class_labels_map);
    
    check_list=[str(class_idx) for class_idx in class_idx_all];
    # print check_list

    file_list=[file_curr for file_curr in os.listdir(shot_dir) if file_curr.endswith('.p') and file_curr[0] in check_list];
    args=[(os.path.join(shot_dir,file_curr),os.path.join(out_dir,file_curr),idx) for idx,file_curr in enumerate(file_list)];
    
    print len(args);
    
    p=multiprocessing.Pool(n_jobs);
    p.map(saveBestTubeAvgScore,args);

    # for class_idx in class_idx_all:
    #     rel_files=[file_curr for file_curr in file_list if file_curr.startswith(str(class_idx)+'_') and file_curr.endswith('.p')];
    #     for rel_file in rel_files:
    #         score_file=os.path.join(shot_dir,rel_file);
    #         out_file=os.path.join(out_dir,rel_file);
    #         best_tube_rank,tubes_ranked,scores_ranked = getBestTubeAvgScore(score_file)    
    #         pickle.dump([best_tube_rank,tubes_ranked,scores_ranked],open(out_file,'wb'));

    #     out_files=[os.path.join(shot_dir,file_curr) for file_curr in file_list if file_curr.startswith(str(class_idx)+'_') and file_curr.endswith('.p')];


    #     score_file=os.path.join(shot_dir,str(class_idx)+'_'+str(video_id)+'_'+str(shot_id)+'.p');
    #     best_tube_rank,tubes_ranked,scores_ranked = getBestTubeAvgScore(os.path.join(shot_dir,score_file))

    return





    path_to_data='/disk2/youtube/categories'
    to_exclude = ['/disk2/res11/horse_7_16.mat','/disk2/res11/horse_7_14.mat','/disk2/res11/horse_4_49.mat']

    meta_dir='/disk2/res11'
    mat_files=pickle.load(open('/disk2/temp/horse_problem.p','rb'))
    mat_files=[os.path.join(meta_dir,file_curr[file_curr.rindex('/')+1:]+'.mat') for file_curr in mat_files];
    # for mat_file in mat_files:

    mat_file = '/disk2/res11/horse_7_11.mat'
    out_file = '/disk2/temp.png';

    drawTubeBB(mat_file,path_to_data,out_file)


    return
    path_to_data='/disk2/youtube/categories'
    out_dir_patches='/disk2/res11/tubePatches';
    path_to_mat='/disk2/res11';

    if not os.path.exists(out_dir_patches):
        os.mkdir(out_dir_patches);

    mat_files=[os.path.join(path_to_mat,file_curr) for file_curr in os.listdir(path_to_mat) if file_curr.endswith('.mat')]
    mat_file=mat_files[0];
    print mat_file
    drawTubeBB(mat_file)
    # script_saveTubePatches(mat_files,path_to_data,out_dir_patches,numThreads=8)

    return

    shot_dir='/disk2/januaryExperiments/shot_score_normalized';
    class_idx=0
    video_id=1
    shot_id=1
    score_file=os.path.join(shot_dir,str(class_idx)+'_'+str(video_id)+'_'+str(shot_id)+'.p');
    best_tube_rank,tubes_ranked,scores_ranked = getBestTubeAvgScore(os.path.join(shot_dir,score_file))
    print best_tube_rank,tubes_ranked,scores_ranked



if __name__=='__main__':
    main();