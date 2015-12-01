import os;
import numpy as np;
import matplotlib as plt;
import scipy.io
import cPickle as pickle
from scipy import misc;
import visualize;
import math;
import time;
import caffe_wrapper;
import multiprocessing;
import script_top5error;
import util;
import subprocess;
from tube_db import Tube, Tube_Manipulator

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

def main():
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