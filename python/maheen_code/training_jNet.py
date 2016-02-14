
import os;
import util;
import numpy as np;
import matplotlib
import subprocess;
import scipy.io
from math import ceil
import cPickle as pickle;
from collections import namedtuple
import random;

def createParams(type_Experiment):
    pass;

def getMovedDirPath(moved_dir,orig_dir,sub_dirs_file_name):
    sub_dirs=[util.readLinesFromFile(os.path.join(dir_curr,sub_dirs_file_name)) for dir_curr in [moved_dir,orig_dir]];
    full_paths=[os.path.join(orig_dir,dir_curr) for dir_curr in sub_dirs[1] if dir_curr not in sub_dirs[0]];
    full_paths=full_paths+[os.path.join(moved_dir,dir_curr) for dir_curr in sub_dirs[0]]
    list_intersection=set(sub_dirs[0]+sub_dirs[1]);
    return full_paths

def saveCorrespondingDirs(meta_dirs_image,meta_dirs_flo,sub_dirs_file,out_file_correspondences):
    image_dirs = getMovedDirPath(meta_dirs_image[0],meta_dirs_image[1],sub_dirs_file);
    flo_dirs = getMovedDirPath(meta_dirs_flo[0],meta_dirs_flo[1],sub_dirs_file);
    flo_idx=[];
    image_dirs_stripped=[dir_curr.rsplit('/',2)[1] for dir_curr in image_dirs];
    flo_dirs_stripped=[dir_curr.rsplit('/',2)[1] for dir_curr in flo_dirs];
    index_array= util.getIndexingArray(flo_dirs_stripped,image_dirs_stripped);
    flo_dirs=np.array(flo_dirs);
    flo_dirs=flo_dirs[index_array];
    flo_dirs=list(flo_dirs);
    pickle.dump(zip(image_dirs,flo_dirs),open(out_file_correspondences,'wb'));

def getBatchSizeFromDeploy(proto_file):
    with open(proto_file,'rb') as f:
        data=f.read();
    idx=data.index('batch_size')+len('batch_size');
    data=data[idx+2:];
    data=data[:data.index('\n')];
    data=int(data);
    return data; 

def script_saveImFloFileInfo(out_file_correspondences,proto_file,out_file):
    dirs=pickle.load(open(out_file_correspondences,'rb'));
    
    im_files_all=[];
    flo_files_all=[];
    batch_size_all=[];
    num_batches_all=[];
    for im_dir,flo_dir in dirs:
        im_files=[os.path.join(im_dir,file_curr) for file_curr in os.listdir(im_dir) if file_curr.endswith('.ppm')];
        flo_files=[os.path.join(flo_dir,file_curr) for file_curr in os.listdir(flo_dir) if file_curr.endswith('.flo')];
        data = getBatchSizeFromDeploy(os.path.join(flo_dir,proto_file));
        im_files_all.append(im_files);
        flo_files_all.append(flo_files);
        batch_size_all.append(data);
        num_batches_all.append((len(im_files)-1)/data);
        i+=1;

    # print min(num_batches_all),max(num_batches_all);
    pickle.dump([im_files_all,flo_files_all,batch_size_all,num_batches_all],open(out_file,'wb'));

def script_saveMatFiles(flo_dir,im_dir,out_dir,mat_file,proto_file):
    #get video name
    video_name=flo_dir[:-1];
    video_name=video_name[video_name.rindex('/')+1:];
    print video_name

    #get flo files
    flo_files=[os.path.join(flo_dir,file_curr) for file_curr in os.listdir(flo_dir) if file_curr.endswith('.flo')];
    flo_files.sort();

    #get im files
    im_files=util.readLinesFromFile(os.path.join(flo_dir,'im_1.txt'));
    old_dir=im_files[0][:im_files[0].rindex('/')+1];
    
    #if dirs have changed, replace the paths
    if im_dir!=old_dir:
        im_files=[im_curr.replace(old_dir,im_dir) for im_curr in im_files];

    #get batch size
    batch_size=getBatchSizeFromDeploy(os.path.join(flo_dir,proto_file));

    #get batch info
    batch_num=[int(file_curr[file_curr.rindex('-')+1:file_curr.rindex('(')]) for file_curr in flo_files];
    batch_num=np.array(batch_num);
    batch_ids=list(set(batch_num))
    batch_ids.sort();
    
    flo_files_all = [];
    im_files_all = []
    for batch_no in batch_ids:
        idx_rel=np.where(batch_num==batch_no)[0];
        
        flo_files_curr=[];
        im_files_curr=[];
        for idx_curr in idx_rel:
            flo_file=flo_files[idx_curr];
            im_no=int(flo_file[flo_file.rindex('(')+1:flo_file.rindex(')')]);
            im_corr=im_files[batch_no*batch_size+im_no];
            flo_files_curr.append(flo_file);
            im_files_curr.append(im_corr);
        
        flo_files_all.append(flo_files_curr);
        im_files_all.append(im_files_curr);

    #save as mat with flofiles, im_files, and out_dir;
    for idx_batch_no,batch_no in enumerate(batch_ids):
        flo_files=flo_files_all[idx_batch_no];
        im_files=im_files_all[idx_batch_no];

        out_dir_mat = os.path.join(out_dir,video_name+'_'+str(batch_no));
        # print out_dir_mat

        if not os.path.exists(out_dir_mat):
            os.mkdir(out_dir_mat);
        out_file=os.path.join(out_dir_mat,mat_file);
        print out_file
        mat_data={'flo_files':flo_files,'im_files':im_files}
        
        scipy.io.savemat(out_file,mat_data)


def getRemainingDirs(all_dirs,check_file):
    remainingDirs=[];

    for dir_curr in all_dirs:
        if not os.path.exists(os.path.join(dir_curr,check_file)):
            remainingDirs.append(dir_curr);
            # continue;    
    return remainingDirs;

def script_writeCommandsForPreprocessing(all_dirs_file,command_file_pre,num_proc,check_file=None):
    all_dirs=util.readLinesFromFile(all_dirs_file);
    all_dirs=[dir_curr[:-1] for dir_curr in all_dirs];
    
    if check_file is not None:
        all_dirs=getRemainingDirs(all_dirs,check_file);

    command_pre='echo '
    command_middle_1=';cd ~/Downloads/opticalflow; matlab -nojvm -nodisplay -nosplash -r "out_folder=\''
    command_middle='\';saveTrainingData" > '
    command_end=' 2>&1';

    commands=[];
    for dir_curr in all_dirs:
        dir_curr=util.escapeString(dir_curr);
        log_file=os.path.join(dir_curr,'log.txt');
        command=command_pre+dir_curr+command_middle_1+dir_curr+command_middle+log_file+command_end;
        commands.append(command);
    
    idx_range=util.getIdxRange(len(commands),len(commands)/num_proc)
    command_files=[];
    for i,start_idx in enumerate(idx_range[:-1]):
        command_file_curr=command_file_pre+str(i)+'.txt'
        end_idx=idx_range[i+1]
        commands_rel=commands[start_idx:end_idx];
        util.writeFile(command_file_curr,commands_rel);
        command_files.append(command_file_curr);
    return command_files;


def main():

    all_dirs_file='/disk2/februaryExperiments/training_jacob/all_dirs.txt';
    command_file_pre='/disk2/februaryExperiments/training_jacob/commands_training_data_';
    train_data_file='/disk2/februaryExperiments/training_jacob/caffe_files/train.txt';
    check_file='done.mat'
    num_proc=12;
    # command_files = script_writeCommandsForPreprocessing(all_dirs_file,command_file_pre,num_proc,check_file);

    all_dirs=util.readLinesFromFile(all_dirs_file);
    # all_dirs=all_dirs[:10];
    random.shuffle(all_dirs);

    strings=[];
    for no_dir_curr,dir_curr in enumerate(all_dirs):
        print no_dir_curr,dir_curr
        dir_curr=dir_curr[:-1];
        curr_flos=[os.path.join(dir_curr,curr_flo) for curr_flo in os.listdir(dir_curr) if curr_flo.endswith('.tif')];
        for curr_flo in curr_flos:
            curr_im=curr_flo.replace('.tif','.jpg');
            assert os.path.exists(curr_im);
            string_curr=curr_im+'  '+curr_flo+' '
            strings.append(string_curr);
    print len(strings);
    # print strings[:3];

    # random.shuffle(strings);
    util.writeFile(train_data_file,strings);
    # with open (train_data_file,'wb') as f:
    #     for im_curr,flo_curr in zip(ims,flos):
    #         string_curr=im_curr+' '+flo_curr+'\n';
    #         f.write(string_curr);


    return
    dirs = getRemainingDirs(util.readLinesFromFile(all_dirs_file),check_file);
    last_lines=[];
    for dir_curr in dirs:
        last_lines.append(util.readLinesFromFile(os.path.join(dir_curr,'log.txt'))[-2]);
    print set(last_lines);




    return
    meta_dirs_image=['/disk2/image_data_moved',
                    '/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data'];
    meta_dirs_flo=['/disk2/flow_data',
                    '/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data'];
    sub_dirs_file='all_sub_dirs.txt';
    out_dir='/disk2/februaryExperiments/training_jacob'
    out_file_correspondences=os.path.join(out_dir,'im_flo_correspondences.p');
    proto_file='deploy.prototxt';
    out_file=os.path.join(out_dir,'im_flo_files.p');

    out_dir='/disk2/februaryExperiments/training_jacob/training_data';
    mat_file='im_flo_files.mat';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);

    im_flo_dirs=pickle.load(open(out_file_correspondences,'rb'))
    [im_dirs,flo_dirs]=zip(*im_flo_dirs);

    for im_dir,flo_dir in im_flo_dirs:
        script_saveMatFiles(flo_dir,im_dir,out_dir,mat_file,proto_file)
        
    




    # for batch_id in batch_num:


    # print len(im_files);
    # print len(flo_files);
    # batch_size=221;

    # flo_files.sort();
    # i=0;
    # for flo_file in flo_files[:10]:
    #     i=i+1;
    #     batch_no=int(flo_file[flo_file.rindex('-')+1:flo_file.rindex('(')]);
    #     im_no=int(flo_file[flo_file.rindex('(')+1:flo_file.rindex(')')]);
    #     im_corr=im_files[batch_no*batch_size+im_no];
    #     print flo_file,batch_no,im_no,im_corr,i


    
    
    


if __name__=='__main__':
    main();


