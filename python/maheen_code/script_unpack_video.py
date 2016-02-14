import os;
import util;
import numpy as np;
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import subprocess;
from math import ceil
import cPickle as pickle;
from collections import namedtuple

def createParams(type_Experiment):
    if type_Experiment == 'writeFlownetCommands':
        list_params=['video_list_file',
                    'path_to_video_meta',
                    'in_dir_meta',
                    'out_dir_meta',
                    'path_to_deploy',
                    'out_file_commands',
                    'dir_flownet_meta',
                    'path_to_sizer',
                    'caffe_bin',
                    'path_to_model',
                    'text_1',
                    'text_2',
                    'deploy_file',
                    'gpu']
        params = namedtuple('Params_writeFlownetCommands',list_params);
    else:
        params=None;

    return params;

def script_saveCommands(in_dir_meta,out_dir_meta,out_file_text):
    cats=[dir_curr for dir_curr in os.listdir(in_dir_meta) if os.path.isdir(os.path.join(in_dir_meta,dir_curr))];
    # print len(cats);
    commands=[];
    command_text='ffmpeg -i '
    for cat in cats:
        dir_curr=os.path.join(in_dir_meta,cat);
        dir_out_curr=os.path.join(out_dir_meta,cat);
        print dir_out_curr;
        
        if not os.path.exists(dir_out_curr):
            os.mkdir(dir_out_curr);
        videos=[file_curr for file_curr in os.listdir(dir_curr) if file_curr.endswith('.avi')];
        for video in videos:
            out_dir_video_org=os.path.join(dir_out_curr,video[:-4])
            out_dir_video = util.escapeString(out_dir_video_org)
            command_curr='mkdir '+out_dir_video+';'+command_text+util.escapeString(os.path.join(dir_curr,video))+' '+out_dir_video+'/image_%d.ppm'
            commands.append(command_curr);
    with open(out_file_text,'wb') as f:
        for command in commands:
            f.write(command+'\n');


def getImageListForFlow(image_dir):
    image_list=[im for im in os.listdir(image_dir) if im.endswith('.ppm')];
    image_list_sorted=[];
    for idx in range(len(image_list)):
        idx_im=idx+1;
        im_curr='image_'+str(idx_im)+'.ppm';
        assert im_curr in image_list;
        image_list_sorted.append(im_curr);
    list_1=[];
    list_2=[];
    for idx in range(len(image_list_sorted)-1):
        list_1.append(os.path.join(image_dir,image_list_sorted[idx]));
        list_2.append(os.path.join(image_dir,image_list_sorted[idx+1]));
    assert len(list_1)==len(list_2);
    return list_1,list_2;
    
def readFloFile(file_name):
    with open(file_name, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (w, h, 2))
    return data2D;

def replaceProto(path_to_deploy,out_deploy,dim_list,text_1,text_2,batchsize,out_dir_curr):
    width=dim_list[0];
    height=dim_list[1];
    divisor = 64.
    adapted_width = ceil(width/divisor) * divisor
    adapted_height = ceil(height/divisor) * divisor
    rescale_coeff_x = width / adapted_width
    rescale_coeff_y = height / adapted_height

    replacement_list = {
        '$ADAPTED_WIDTH': ('%d' % adapted_width),
        '$ADAPTED_HEIGHT': ('%d' % adapted_height),
        '$TARGET_WIDTH': ('%d' % width),
        '$TARGET_HEIGHT': ('%d' % height),
        '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
        '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y),
        '$LIST1_TXT' : '"'+text_1+'"',
        '$LIST2_TXT' : '"'+text_2+'"',
        '$BATCHSIZE' : ('%d' % batchsize),
        '$OUT_FOLDER' : '"'+out_dir_curr+'"'      
    }

    proto = ''
    with open(path_to_deploy, "r") as tfile:
        proto = tfile.read()

    for r in replacement_list:
        proto = proto.replace(r, replacement_list[r])

    with open(out_deploy, "w") as tfile:
        tfile.write(proto)


def script_writeFlownetCommands(params):
    video_list_file= params.video_list_file;
    path_to_video_meta= params.path_to_video_meta;
    in_dir_meta= params.in_dir_meta;
    out_dir_meta= params.out_dir_meta;
    path_to_deploy= params.path_to_deploy;
    out_file_commands= params.out_file_commands;
    dir_flownet_meta= params.dir_flownet_meta;
    path_to_sizer= params.path_to_sizer;
    caffe_bin = params.caffe_bin;
    path_to_model = params.path_to_model;
    text_1_org= params.text_1;
    text_2_org= params.text_2;
    deploy_file= params.deploy_file;
    gpu= params.gpu;

    im_dirs=util.readLinesFromFile(video_list_file);
    im_dirs=[im_dir.replace(path_to_video_meta,in_dir_meta)[:-4] for im_dir in im_dirs];
    
    commands=[];
    # im_dirs=['/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data/hmdb/pick/THE_WALLET_TRICK!!!_pick_f_cm_np2_ba_med_1'];
    for idx_im_dir,im_dir in enumerate(im_dirs):
        print idx_im_dir,len(im_dirs);
        out_dir_curr=im_dir.replace(in_dir_meta,out_dir_meta);
        text_1=os.path.join(out_dir_curr,text_1_org);
        text_2=os.path.join(out_dir_curr,text_2_org);
        out_deploy=os.path.join(out_dir_curr,deploy_file);

        subprocess.call('mkdir -p '+util.escapeString(out_dir_curr), shell=True)
        
        list_1,list_2 = getImageListForFlow(im_dir)
        util.writeFile(text_1,list_1);
        util.writeFile(text_2,list_2);
        
        # im_test=util.escapeString(list_1[0]);
        dim_list = [int(dimstr) for dimstr in str(subprocess.check_output([path_to_sizer, list_1[0]])).split(',')]
        replaceProto(path_to_deploy,out_deploy,dim_list,text_1,text_2,len(list_1),out_dir_curr)
        
        args = [caffe_bin, 'test', '-model', util.escapeString(out_deploy),
            '-weights', path_to_model,
            '-iterations', '1',
            '-gpu', str(gpu)]

        cmd = str.join(' ', args)
        commands.append(cmd);

    # print('Executing %s' % cmd)
    util.writeFile(out_file_commands,commands);

def writeCommands_hacky(out_file_commands,dirs,caffe_bin,deploy_name,path_to_model,gpu):
    commands=[];
    for dir_curr in dirs:
        out_deploy=os.path.join(dir_curr,deploy_name);
        args = [caffe_bin, 'test', '-model', util.escapeString(out_deploy),
            '-weights', path_to_model,
            '-iterations', '1',
            '-gpu', str(gpu)]

        cmd = str.join(' ', args)
        commands.append(cmd)
    util.writeFile(out_file_commands,commands);

def main():
    # dir_meta='/disk2/flow_data';
    # dir_meta_old='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data';
    # deploy_file='deploy.prototxt';
    # dir_mids=[os.path.join(dir_meta,dir_mid) for dir_mid in os.listdir(dir_meta) if os.path.isdir(os.path.join(dir_meta,dir_mid))]
    # dirs_left=[os.path.join(dir_mid,dir_curr) for dir_mid in dir_mids for dir_curr in os.listdir(dir_mid) if os.path.isdir(os.path.join(dir_mid,dir_curr))]

    # dirs_left=[os.path.join(dir_mid,dir_curr) for dir_mid in dirs_left for dir_curr in os.listdir(dir_mid) if os.path.isdir(os.path.join(dir_mid,dir_curr))]
    
    # print len(dirs_left);
    # print dirs_left[0];

    # for dir_curr in dirs_left:
    #     deploy_curr=os.path.join(dir_curr,deploy_file);
    #     print deploy_curr
    #     data=[];
    #     with open(deploy_curr,'r') as f:
    #         data = f.read()
        
    #     with open(deploy_curr+'_backup','w') as f:
    #         f.write(data);

    #     data = data.replace(dir_meta_old, dir_meta)
    #     with open(deploy_curr, "w") as f:
    #         f.write(data);

    # return
    # video_list_file='/disk2/video_data/video_list.txt'
    # path_to_video_meta='/disk2/video_data';

    # path_to_flo_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data';
    # path_to_im_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data';
    
    # video_files=util.readLinesFromFile(video_list_file);
    # # image_dirs=[dir_curr.replace(path_to_video_meta,path_to_im_meta)[:-4] for dir_curr in video_files];
    # # flo_dirs=[dir_curr.replace(path_to_video_meta,path_to_flo_meta)[:-4] for dir_curr in video_files];
    # flo_dirs=pickle.load(open('/disk2/temp/dirs_done.p','rb'));
    # image_dirs=[dir_curr.replace(path_to_flo_meta,path_to_im_meta) for dir_curr in flo_dirs];
    # print len(image_dirs)
    # out_dir='/disk2/image_data_moved';

    # out_file='/disk2/image_data_moved/mv_commands_2.txt'
    # commands=[];
    # image_dirs_to_move=image_dirs[5000:7000];
    # for image_dir in image_dirs_to_move:
    #     image_dir=util.escapeString(image_dir);
    #     new_dir=image_dir.replace(path_to_im_meta,out_dir);
    #     command='mkdir -p '+new_dir+';';
    #     command=command+'mv '+image_dir+'/* '+new_dir;
    #     commands.append(command);
    # util.writeFile('/disk2/image_data_moved/dirs_moved_2.txt',image_dirs_to_move);
    # util.writeFile(out_file,commands);

    # return
    video_list_file='/disk2/video_data/video_list.txt'
    path_to_video_meta='/disk2/video_data';

    # path_to_flo_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data';
    path_to_flo_meta='/disk2/flow_data';
    path_to_im_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data';
    
    video_files=util.readLinesFromFile(video_list_file);
    # image_dirs=[dir_curr.replace(path_to_video_meta,path_to_im_meta)[:-4] for dir_curr in video_files];
    # flo_dirs=[dir_curr.replace(path_to_video_meta,path_to_flo_meta)[:-4] for dir_curr in video_files];
    flo_dirs=pickle.load(open('/disk2/temp/dirs_done_disk2.p','rb'));
    image_dirs=[dir_curr.replace(path_to_flo_meta,path_to_im_meta) for dir_curr in flo_dirs];
    print len(image_dirs);
    finished=[];
    i=0;
    for image_dir,flo_dir in zip(image_dirs,flo_dirs):
        print i
        count_im_command='ls '+os.path.join(util.escapeString(image_dir),'*.ppm')+'| wc -l';
        count_flo_command='ls '+os.path.join(util.escapeString(flo_dir),'*.flo')+'| wc -l';
        
        # im_count=int(subprocess.check_output(count_im_command,shell=True));
        # flo_count=int(subprocess.check_output(count_flo_command,shell=True));
        im_count=len([file_curr for file_curr in os.listdir(image_dir) if file_curr.endswith('.ppm')]);
        flo_count=len([file_curr for file_curr in os.listdir(flo_dir) if file_curr.endswith('.flo')]);
        print i,flo_count,im_count
        if flo_count+1==im_count:
            finished.append(1);
        else:
            finished.append(0);
        
        i+=1;

    finished=np.array(finished);
    print 'done',sum(finished==1);
    print 'not done',sum(finished==0);

    pickle.dump([finished,image_dirs],open('/disk2/temp/to_rerun.p','wb'));
    
    return
    dir_flownet_meta='/home/maheenrashid/Downloads/flownet/flownet-release/models/flownet';
    caffe_bin = os.path.join(dir_flownet_meta,'bin/caffe')
    path_to_model = os.path.join(dir_flownet_meta,'model/flownet_official.caffemodel')

    video_list_file='/disk2/video_data/video_list.txt'
    path_to_video_meta='/disk2/video_data';

    in_dir_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data';
    in_dir_meta='/disk2/flow_data';
    # if not os.path.exists(new_in_dir_meta):
    #     os.mkdir(new_in_dir_meta);

    deploy_name='deploy.prototxt';
    gpu=0;

    dirs=[dir_curr.replace(path_to_video_meta,in_dir_meta)[:-4] for dir_curr in util.readLinesFromFile(video_list_file)];
    dirs=[dir_curr for dir_curr in dirs if os.path.exists(dir_curr)]
    counts=[len(os.listdir(dir_curr)) for dir_curr in dirs if os.path.exists(dir_curr)];
    dirs_left=[]
    dirs_done=[];
    for idx_count,count in enumerate(counts):
        if count==4:
            dirs_left.append(dirs[idx_count])
            # dir_curr=dirs[idx_count]
            # deploy_curr=os.path.join(dir_curr,deploy_name);
            # im_file=os.path.join(dir_curr,'im_1.txt');
            # batch_size = sum(1 for line in open(im_file))
            
            # old_str='batch_size: '+str(int(ceil(batch_size/5)));
            # print old_str,
            
            # batch_size = int(ceil(batch_size/8));
            # new_str='batch_size: '+str(batch_size);
            # print new_str

            # data=[];
            # with open(deploy_curr,'r') as f:
            #     data = f.read()
            # # print data[:300];
            # assert old_str in data;
            # data = data.replace(old_str, new_str)
            # # print data[:300];
            # with open(deploy_curr, "w") as f:
            #     f.write(data);

            # out_dir_curr=dir_curr.replace(in_dir_meta,new_in_dir_meta);
            #mkdir of new location
            # mkdir_command='mkdir -p '+util.escapeString(out_dir_curr)
            # print mkdir_command
            # subprocess.call(mkdir_command, shell=True)

            #mv contents from old to new
            # mv_command='mv '+util.escapeString(dir_curr)+'/* '+util.escapeString(out_dir_curr);
            # print mv_command
            # subprocess.call(mv_command, shell=True)
            #append new to dirs_left
            # dirs_left.append(out_dir_curr);
            # raw_input();
        else:
            dirs_done.append(dirs[idx_count])



    print min(counts);
    counts=np.array(counts);
    print sum(counts==4);
    print len(dirs_left)
    
    mid_point=len(dirs_left)/2;
    
    print mid_point,len(dirs_left)-mid_point
    out_file_commands='/disk2/januaryExperiments/gettingFlows/flownet_commands_left_0.txt';
    gpu=0;
    # writeCommands_hacky(out_file_commands,dirs_left[:mid_point],caffe_bin,deploy_name,path_to_model,gpu)
    
    out_file_commands='/disk2/januaryExperiments/gettingFlows/flownet_commands_left_1.txt';
    gpu=1;
    # writeCommands_hacky(out_file_commands,dirs_left[mid_point:],caffe_bin,deploy_name,path_to_model,gpu)
    
    print len(dirs_done);
    pickle.dump(dirs_done,open('/disk2/temp/dirs_done_disk2.p','wb'));

    return
    video_list_file='/disk2/video_data/video_list.txt'
    path_to_video_meta='/disk2/video_data';

    in_dir_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data';
    out_dir_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/flow_data';
    path_to_deploy='/disk2/januaryExperiments/gettingFlows/deploy_template.prototxt';
    out_file_commands='/disk2/januaryExperiments/gettingFlows/flownet_commands.txt';

    dir_flownet_meta='/home/maheenrashid/Downloads/flownet/flownet-release/models/flownet';
    path_to_sizer=os.path.join(dir_flownet_meta,'bin/get_image_size')
    caffe_bin = os.path.join(dir_flownet_meta,'bin/caffe')
    path_to_model = os.path.join(dir_flownet_meta,'model/flownet_official.caffemodel')
    
    text_1='im_1.txt';
    text_2='im_2.txt';
    deploy_file='deploy.prototxt';
    gpu=0;

    params_dict={};
    params_dict['video_list_file'] = video_list_file;
    params_dict['path_to_video_meta'] = path_to_video_meta;
    params_dict['in_dir_meta'] = in_dir_meta;
    params_dict['out_dir_meta'] = out_dir_meta;
    params_dict['path_to_deploy'] = path_to_deploy;
    params_dict['out_file_commands'] = out_file_commands;
    params_dict['dir_flownet_meta'] = dir_flownet_meta;
    params_dict['path_to_sizer'] = path_to_sizer;
    params_dict['caffe_bin']  = caffe_bin;
    params_dict['path_to_model']  = path_to_model;
    params_dict['text_1'] = text_1;
    params_dict['text_2'] = text_2;
    params_dict['deploy_file'] = deploy_file;
    params_dict['gpu'] = gpu;
    
    params=createParams('writeFlownetCommands');
    params=params(**params_dict);
    # script_writeFlownetCommands(params);
    commands=util.readLinesFromFile(params.out_file_commands);
    commands=[c.replace('-gpu 1','-gpu 0') for c in commands];
    util.writeFile(params.out_file_commands,commands);
    pickle.dump(params._asdict(),open(params.out_file_commands+'_meta_experiment.p','wb'));




    return
    video_list_file='/disk2/video_data/video_list.txt'
    path_to_im_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data';
    path_to_video_meta='/disk2/video_data';
    commands_file_text='/disk2/januaryExperiments/gettingFlows/resize_commands.txt';

    video_list=util.readLinesFromFile(video_list_file);
    print len(video_list);
    image_dirs=[video_curr.replace(path_to_video_meta,path_to_im_meta)[:-4] for video_curr in video_list];
    print len(image_dirs),image_dirs[0];
    image_dirs=image_dirs[:1];

    commands=[];
    command_conv=['convert','-resize 512x384'];
    for image_dir in image_dirs:
        image_list=[os.path.join(image_dir,im) for im in os.listdir(image_dir) if im.endswith('.ppm')];
        for image_curr in image_list:
            command_curr=[command_conv[0],image_curr,command_conv[1],image_curr];
            command_curr=' '.join(command_curr);
            commands.append(command_curr);

    print len(commands);
    print commands[0];
    util.writeFile(commands_file_text,commands);



    return
    video_list_file='/disk2/video_data/video_list.txt'
    path_to_im_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data';
    path_to_video_meta='/disk2/video_data';
    path_to_txt_1='/disk2/januaryExperiments/gettingFlows/temp_im_1.txt';
    path_to_txt_2='/disk2/januaryExperiments/gettingFlows/temp_im_2.txt';

    video_list=util.readLinesFromFile(video_list_file);
    print len(video_list);
    image_dirs=[video_curr.replace(path_to_video_meta,path_to_im_meta)[:-4] for video_curr in video_list];
    print len(image_dirs),image_dirs[0];
    
    list_1=[];list_2=[];
    for image_dir in image_dirs[:10]:
        list_1_curr,list_2_curr=getImageListForFlow(image_dir);
        list_1.extend(list_1_curr[:3]);
        list_2.extend(list_2_curr[:3]);
    
    assert len(list_1)==len(list_2);

    util.writeFile(path_to_txt_1,list_1);
    util.writeFile(path_to_txt_2,list_2);

    



    # for im_curr in image_list_sorted[:10]:
    #   print im_curr;
    # print len(image_list),len(image_list_sorted);

    # cats=[dir_curr for dir_curr in os.listdir(out_dir_meta) if os.path.isdir(os.path.join(out_dir_meta,dir_curr))];

    # im_counts=[];
    # video_dirs=[];

    # for cat in cats:
    #   dir_cat=os.path.join(out_dir_meta,cat);
    #   videos=[dir_curr for dir_curr in os.listdir(dir_cat) if os.path.isdir(os.path.join(dir_cat,dir_curr))];
    #   im_counts_curr=[len(os.listdir(os.path.join(dir_cat,video_curr))) for video_curr in videos]
    #   video_dirs_curr=[os.path.join(dir_cat,video_curr) for video_curr in videos];

    #   im_counts.extend(im_counts_curr);
    #   video_dirs.extend(video_dirs_curr);

    # print len(im_counts);
    # print len(video_dirs);
    # print im_counts[:10];

    # return
    # flo_file='/home/maheenrashid/Downloads/flownet/flownet-release/models/flownet/flownets-pred-0000000.flo'
    # data2D = readFloFile(flo_file);
    # data2D = np.dstack((data2D,np.zeros((data2D.shape[0],data2D.shape[1]))));
    # print data2D.shape
    # f=plt.figure();
    # plt.imshow(data2D);
    # plt.savefig('/disk2/temp/flo_x.png')
    # plt.close(f);
    # f=plt.figure();
    # plt.imshow(data2D[:,:,1]);
    # plt.savefig('/disk2/temp/flo_y.png')
    # plt.close(f);


    # return

if __name__=='__main__':
    main();