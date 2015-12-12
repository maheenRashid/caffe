import os;
import numpy as np;
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import scipy.io
import cPickle as pickle
from scipy import misc;
import visualize;
import math;
import random;
import time;
import caffe_wrapper;
import multiprocessing;
import script_top5error;
import util;
import subprocess;
import nearest_neighbor
from tube_db import Tube, Tube_Manipulator,TubeHash_Manipulator,TubeHash
from collections import namedtuple
import lsh;


def addLeadingZeros(num,sizeOfStr):
    num_str=str(num);
    zeros='0'*(sizeOfStr-len(num_str));
    str_formatted=zeros+num_str;
    return str_formatted;

def getFramePath(path_to_data,category,video_id,shot_id,frame_id):
    video_id=addLeadingZeros(video_id,4);
    shot_id=addLeadingZeros(shot_id,3);
    frame_id=addLeadingZeros(frame_id,4);
    frame_name='frame'+frame_id+'.jpg'
    path_to_frame=os.path.join(path_to_data,category,'data',video_id,'shots',shot_id,frame_name);
    return path_to_frame;

def saveTubePatches((out_dir_meta,file_name,path_to_data,idx_debug)):
    mat_file=file_name[file_name.rindex('/')+1:];
    print mat_file,idx_debug
    
    out_dir_mat=os.path.join(out_dir_meta,mat_file[:mat_file.index('.')]);
    if not os.path.exists(out_dir_mat):
        os.mkdir(out_dir_mat);
    category=mat_file[:mat_file.index('_')];
    video_id=mat_file[mat_file.index('_')+1:mat_file.rindex('_')];
    shot_id=mat_file[mat_file.rindex('_')+1:mat_file.rindex('.')];
    
    res=scipy.io.loadmat(file_name,squeeze_me=True, struct_as_record=False);
    
    res=res['res'];
    if not hasattr(res,'__iter__'):
        res=[res];
    num_frames=res[0].bboxes.shape[0];
    num_tubes=len(res);

    # print num_frames

    check_files=[];
    for num_tube in range(num_tubes):
        for num_frame in range(num_frames):
            file_curr=os.path.join(out_dir_mat,str(num_tube),str(num_frame)+'.jpg');
            check_files.append(os.path.exists(file_curr));
    # print check_files;
    if sum(check_files)==len(check_files):
        print 'returning';
        return;

    # return
    out_folders=[];
    for tube_no in range(len(res)):
        out_dir_curr=os.path.join(out_dir_mat,str(tube_no));
        if not os.path.exists(out_dir_curr):
            os.mkdir(out_dir_curr);
        out_folders.append(out_dir_curr);
    

    frame_id=1
    frame_path= getFramePath(path_to_data,category,video_id,shot_id,frame_id);
    im=misc.imread(frame_path);
        
    max_r=max([max(res_curr.bboxes[:,3]) for res_curr in res])-1;
    max_c=max([max(res_curr.bboxes[:,2]) for res_curr in res])-1;
    
    max_r=math.floor(max_r/0.4);
    max_c=math.floor(max_c/0.4);
    if max_r>=im.shape[0] or max_c>=im.shape[1]:
        print 'resize_error'
        pickle.dump('',open(os.path.join(out_dir_mat,'resize_error.p'),'wb'));
        return

    for frame_num in range(num_frames):
        frame_id=frame_num+1;
        frame_path= getFramePath(path_to_data,category,video_id,shot_id,frame_id);
        im=misc.imread(frame_path);
        try:    
            for tube_no in range(len(res)):
                out_dir=os.path.join(out_dir_mat,str(tube_no));
                out_file=os.path.join(out_dir,str(frame_num)+'.jpg');
                bboxes=res[tube_no].bboxes

                if sum(bboxes[:,-1]==frame_id)==0:
                    continue;

                idx=np.where(bboxes[:,-1]==frame_id)[0][0]
                bbox=bboxes[idx,:4]
                bbox=bbox-1;
                bbox=[math.floor(x/0.4) for x in bbox];
                
                if len(im.shape)<3:
                    im_curr=im[bbox[1]:bbox[3],bbox[0]:bbox[2]];
                else:
                    im_curr=im[bbox[1]:bbox[3],bbox[0]:bbox[2],:];      
                misc.imsave(out_file,im_curr);
        except Exception:
            print 'exception',
            pickle.dump('',open(os.path.join(out_dir_mat,str(tube_no),'problem.p'),'wb'));
    
    for out_folder in out_folders:
        visualize.writeHTMLForFolder(out_folder);

def writeInputFilesForFeatureExtraction(path_to_patches, out_file):
    
    folders_all=[];
    video_names=[name_curr for name_curr in os.listdir(path_to_patches) if os.path.isdir(os.path.join(path_to_patches,name_curr))];

    in_files=[];
    resize_dirs=[];
    problem_dirs=[];

    for idx,video_name in enumerate(video_names):
        if idx%100==0:
            print idx

        parent_dir=os.path.join(path_to_patches,video_name);
        
        if os.path.exists(os.path.join(parent_dir,'resize_error.p')):
            resize_dirs.append(parent_dir);
            continue;

        tracks=[track_curr for track_curr in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir,track_curr))];
        for track_curr in tracks:
            curr_dir=os.path.join(parent_dir,track_curr);
            if os.path.exists(os.path.join(curr_dir,'problem.p')):
                problem_dirs.append(curr_dir);
                continue;

            file_names=[file_name for file_name in os.listdir(curr_dir) if file_name.endswith('jpg')];

            out_file_text=os.path.join(curr_dir,track_curr+'.txt');
            with open(out_file_text,'wb') as f:
                for file_name in file_names:
                    f.write(os.path.join(curr_dir,file_name)+'\n');

            in_files.append(out_file_text);


    print len(in_files),len(resize_dirs),len(problem_dirs);
    pickle.dump([in_files,resize_dirs,problem_dirs],open(out_file,'wb'));

def script_breakUpInFilesListForFeatureExtraction(file_index,in_file_meta_pre,out_file_meta_pre):
    [in_files,_,_]=pickle.load(open(file_index,'rb'));
    batch_size=len(in_files)/10;
    idx_range=util.getIdxRange(len(in_files),batch_size)
    for idx,idx_begin in enumerate(idx_range[:-1]):
        idx_end=idx_range[idx+1];
        in_files_rel=in_files[idx_begin:idx_end];
        in_file_meta_curr=in_file_meta_pre+'_'+str(idx)+'.p';
        out_file_meta_curr=out_file_meta_pre+'_'+str(idx)+'.p';
        pickle.dump(in_files_rel,open(in_file_meta_curr,'wb'));
        print in_file_meta_curr,out_file_meta_curr

def script_saveTubePatches(mat_files,path_to_data,out_dir_patches,numThreads=8):
    args=[];
    for idx,file_name in enumerate(mat_files):
        # file_name=os.path.join(path_to_mat,mat_file);
        arg_curr=(out_dir_patches,file_name,path_to_data,idx);
        args.append(arg_curr);
        if numThreads<=1:
            saveTubePatches(arg_curr);
    if numThreads>1:
        p = multiprocessing.Pool(min(multiprocessing.cpu_count(),numThreads))
        p.map(saveTubePatches,args)

def writeMetaInfoToDb(path_to_db,out_files,idx_global,class_ids_all,path_to_data):
    mani=Tube_Manipulator(path_to_db);
    
    mani.openSession();
    for out_file_idx,out_file in enumerate(out_files):
        if out_file_idx%100==0:
            print out_file_idx,len(out_files)
        in_file_text=out_file.replace('.npz','.txt');
        patch_files=util.readLinesFromFile(in_file_text);
        # print out_file,in_file_text,len(patch_files);
        
        
        for idx_img_file,img_file in enumerate(patch_files):
            img_path=img_file;
            
            img_path_split=img_path.split('/');
            img_path_split=[segment for segment in img_path_split if segment!=''];
            mat_name=img_path_split[-3];
            
            class_id_pascal=mat_name[:mat_name.index('_')];
            
            video_id=int(mat_name[mat_name.index('_')+1:mat_name.rindex('_')]);
            shot_id=int(mat_name[mat_name.rindex('_')+1:]);
            tube_id=int(img_path_split[-2]);

            frame_id=img_path_split[-1];
            frame_id=int(frame_id[:frame_id.index('.')]);
            # frame_id+=1

            class_idx_pascal=class_ids_all.index(class_id_pascal);
            deep_features_path=out_file;
            deep_features_idx=idx_img_file;
            layer='fc7';

            frame_path=getFramePath(path_to_data,class_id_pascal,video_id,shot_id,frame_id+1)
            assert os.path.exists(frame_path);
            
            

            mani.insert(idx_global, img_path, frame_id, video_id, tube_id, shot_id, frame_path=frame_path, layer=layer, deep_features_path=deep_features_path, deep_features_idx=deep_features_idx, class_id_pascal=class_id_pascal, class_idx_pascal=class_idx_pascal,commit=False);
            idx_global+=1;
    mani.session.commit();
    mani.closeSession();
    return idx_global;

def getTubePathsForShot(path_to_db,class_id_pascal,video_id,shot_id,frame_to_choose='middle'):
    mani=Tube_Manipulator(path_to_db);
    
    mani.openSession();
    frame_ids=mani.select((Tube.frame_id,),(Tube.class_id_pascal==class_id_pascal,Tube.shot_id==shot_id,Tube.video_id==video_id),distinct=True);
    
    frame_ids=[frame_id[0] for frame_id in frame_ids];
    frame_ids.sort();
    
    if frame_to_choose=='middle':
        middle_idx=len(frame_ids)/2;
        frame_id=frame_ids[middle_idx];
    else:
        frame_id=0;

    paths=mani.select((Tube.img_path,),(Tube.class_id_pascal==class_id_pascal,Tube.shot_id==shot_id,Tube.video_id==video_id,Tube.frame_id==frame_id),distinct=True);
    paths=[path[0] for path in paths];

    mani.closeSession();
    return paths;

def getNVideosByPascalIds(path_to_db,pascal_ids,numberofVideos):
    
    dict_out={};

    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    for pascal_id in pascal_ids:
        total_ids=mani.select((Tube.video_id,),(Tube.class_id_pascal==pascal_id,),distinct=True,limit=numberofVideos);
        total_ids=[total_id[0] for total_id in total_ids];
        # random.shuffle(total_ids);
        # selected_ids=total_ids[:numberofVideos];
        dict_out[pascal_id]=total_ids;
    mani.closeSession();
    return dict_out

def setUpFeaturesMat(info_for_extraction,dtype='float64'):
    img_paths = np.array([info_curr[0] for info_curr in info_for_extraction]);
    labels = np.array([info_curr[1] for info_curr in info_for_extraction]);
    paths = np.array([info_curr[2] for info_curr in info_for_extraction]);
    deep_idx = np.array([info_curr[3] for info_curr in info_for_extraction]);

    features=np.zeros((len(info_for_extraction),4096),dtype=dtype);
    paths_uni=np.unique(paths);
    
    for path_curr in paths_uni:
        vals=np.load(path_curr);
        vals=vals['fc7'];
        idx_rel=np.where(paths==path_curr);
        idx_to_extract=deep_idx[idx_rel];
        features[idx_rel[0],:]=vals[idx_to_extract,:,:,:].reshape((len(idx_to_extract),4096));
    return features,labels,img_paths    

def getInfoForFeatureExtractionForVideo(path_to_db,video_info,numberOfFrames):
    info_for_extraction=[];
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    for pascal_id in video_info:
        video_ids=video_info[pascal_id];
        for video_id in video_ids:
            info=mani.select((Tube.img_path,Tube.class_id_pascal,Tube.deep_features_path,Tube.deep_features_idx),(Tube.video_id==video_id,Tube.class_id_pascal==pascal_id),distinct=True,limit=numberOfFrames);
            info_for_extraction=info_for_extraction+info;
    mani.closeSession();
    return info_for_extraction

def getInfoForExtractionForTube(path_to_db,pascal_id,video_id,shot_id,tube_id):
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    info=mani.select((Tube.img_path,Tube.class_id_pascal,Tube.deep_features_path,Tube.deep_features_idx),(Tube.video_id==video_id,Tube.class_id_pascal==pascal_id,Tube.tube_id==tube_id,Tube.shot_id==shot_id),distinct=True);
    mani.closeSession();
    return info


def script_toyNNExperiment(params):
    path_to_db = params.path_to_db;
    class_id_pascal = params.class_id_pascal;
    video_id = params.video_id;
    shot_id = params.shot_id;
    tube_id = params.tube_id;
    numberofVideos = params.numberofVideos;
    numberOfFrames = params.numberOfFrames;
    out_file_html = params.out_file_html;
    rel_path = params.rel_path;
    out_file_hist = params.out_file_hist;
    gpuFlag = params.gpuFlag;
    dtype = params.dtype;
    pascal_ids = params.pascal_ids;
    video_info = params.video_info;
    out_file_pickle = params.out_file_pickle

    info_for_extraction=getInfoForFeatureExtractionForVideo(path_to_db,video_info,numberOfFrames);    
    video_info={class_id_pascal:[video_id]}
    info_for_extraction_query=getInfoForExtractionForTube(path_to_db,class_id_pascal,video_id,shot_id,tube_id)
    features_train,labels_train,img_paths_train=setUpFeaturesMat(info_for_extraction,dtype=dtype);
    features_test,labels_test,img_paths_test=setUpFeaturesMat(info_for_extraction_query,dtype=dtype);
    # features_test,labels_test,img_paths_test=setUpFeaturesMat(info_for_extraction,dtype=dtype);
    indices,distances=nearest_neighbor.getNearestNeighbors(features_test,features_train,gpuFlag=gpuFlag);
    
    img_paths_html=[];
    captions_html=[];
    record_wrong=[]
    for r in range(indices.shape[0]):
        img_paths_row=[img_paths_test[r].replace(rel_path[0],rel_path[1])];
        captions_row=[labels_test[r]];
        for c in range(indices.shape[1]):
            rank=indices[r,c];
            img_paths_row.append(img_paths_train[rank].replace(rel_path[0],rel_path[1]))
            captions_row.append(labels_train[rank]);
            if labels_train[rank]!=labels_test[r]:
                record_wrong.append(c);
        img_paths_html.append(img_paths_row);
        captions_html.append(captions_row);

    visualize.writeHTML(out_file_html,img_paths_html,captions_html);
    visualize.hist(record_wrong,out_file_hist,bins=20,normed=True,xlabel='Rank of Incorrect Class',ylabel='Frequency',title='')
    pickle.dump([features_test,features_train,labels_test,labels_train,img_paths_test,img_paths_train,indices,distances],open(out_file_pickle,'wb'));

def createParams(type_Experiment):
    if type_Experiment=='toyNNExperiment':
        list_params=['path_to_db',
                    'class_id_pascal',
                    'video_id',
                    'shot_id',
                    'tube_id',
                    'numberofVideos',
                    'numberOfFrames',
                    'out_file_html',
                    'rel_path',
                    'out_file_hist',
                    'gpuFlag',
                    'dtype',
                    'pascal_ids',
                    'video_info',
                    'out_file_pickle']
        params=namedtuple('Params_toyNNExperiment',list_params);
    elif type_Experiment=='compareHashWithToyExperiment':
        list_params=['in_file',
                    'num_hash_tables_all',
                    'key_type',
                    'out_file_pres',
                    'out_file_indices',
                    'out_file_html',
                    'rel_path']
        params=namedtuple('Params_compareHashWithToyExperiment',list_params);
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
    
    for out_file_pre,num_hash_tables in zip(out_file_pres,num_hash_tables_all):
        
        indices_hash = getIndicesHash(features_test,features_train,num_hash_tables,key_type);
        visualize.saveMatAsImage(indices_hash,out_file_pre+'.png');    
        pickle.dump([indices_hash,indices],open(out_file_pre+'.p','wb'));
    
    sizes = scipy.misc.imread(out_file_indices);
    sizes = sizes.shape

    im_files_html=[];
    captions_html=[];
    for idx,out_file_pre in enumerate(out_file_pres):
        out_file_curr=out_file_pre+'.png'
        key_str=str(key_type);
        key_str=key_str.replace('<type ','').replace('>','');
        caption_curr='NN Hash. KeyType: '+key_str+' Num Hash Tables: '+str(num_hash_tables_all[idx])
        im_files_html.append([out_file_indices.replace(rel_path[0],rel_path[1]),out_file_curr.replace(rel_path[0],rel_path[1])])
        captions_html.append(['NN cosine',caption_curr]);

    visualize.writeHTML(out_file_html,im_files_html,captions_html,sizes[0]/2,sizes[1]/2);


def main():


    
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
    return
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_test.db';

    path_to_data='/disk2/youtube/categories';
    
    out_file_pre='/disk2/res11/featureExtractionInputOutputFiles/out_files_'
    out_files_list=[out_file_pre+str(x)+'.p' for x in range(2,11)];
    for out_file in out_files_list:
        print out_file
    # print out_files_list;

    # idx_global=0;
    class_ids_all=['aeroplane','bird','boat','car','cat','cow','dog','horse','motorbike','train'];

    # import cProfile, pstats, StringIO
    # pr = cProfile.Profile()
    # pr.enable()
    
    idx_global=1244925;
    for out_file_meta in out_files_list:

        out_files=pickle.load(open(out_file_meta,'rb'));
        
        print out_file_meta,len(out_files);
        # out_files=out_files;
        t=time.time();
        idx_global=writeMetaInfoToDb(path_to_db,out_files,idx_global,class_ids_all,path_to_data);
        print time.time()-t;
    # ... do something ...
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
                
    return
    out_file_pre='/disk2/res11/featureExtractionInputOutputFiles/out_files_'
    in_file_pre='/disk2/res11/featureExtractionInputOutputFiles/in_files_'

    out_files_list=[out_file_pre+str(x)+'.p' for x in range(8,10)];

    for idx_file in range(8,10):
        out_file_meta=out_file_pre+str(idx_file)+'.p';
        in_file_meta=in_file_pre+str(idx_file)+'.p';
        in_files=pickle.load(open(in_file_meta,'rb'));
        out_files=[in_file.replace('.txt','.npz') for in_file in in_files];
        pickle.dump(out_files,open(out_file_meta,'wb'));
        # out_files=pickle.load(open(out_file_meta,'rb'));
        # print out_file_meta
        # print len(out_files),len(in_files);
        # print out_files[:10],in_files[:10];

    return
    



        


if __name__=='__main__':
    main();