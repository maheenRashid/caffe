import os;
import scipy;
import mat4py;
from scipy import misc;
import scipy.io
import visualize;
import numpy as np;
import glob
import script_nearestNeigbourExperiment
import pickle
import matplotlib.pyplot as plt;
def saveBBImages(path_to_im,path_to_anno,file_names,out_dir):
    curr_dict={};
    
    for idx_file_name,file_name in enumerate(file_names):
        
        curr_dict=scipy.io.loadmat(os.path.join(path_to_anno,file_name+'.mat'),squeeze_me=True, struct_as_record=False);
        im=misc.imread(os.path.join(path_to_im,file_name+'.jpg'));
        objects=curr_dict['record'].objects

        if not hasattr(objects, '__iter__'):
            objects=[objects]
        
        for idx,object_curr in enumerate(objects):
            if object_curr.viewpoint != []:
                curr_class= object_curr.__dict__['class']
                bbs_curr= object_curr.bbox
                bbs_curr=[b-1 for b in bbs_curr];
                im_curr=im[bbs_curr[1]:bbs_curr[3],bbs_curr[0]:bbs_curr[2],:];
                file_name_out=file_name[file_name.rindex('/')+1:]
                out_file=os.path.join(out_dir,file_name_out+'_'+curr_class+'_'+str(idx)+'.jpg');
                misc.imsave(out_file,im_curr);


def script_saveAzimuthInfo(file_name,path_to_anno):
    test_set,_=pickle.load(open(file_name+'.p','rb'));
    test_set=sorted(test_set,key=lambda x: x[0])
    test_set=zip(*test_set);
    img_paths=list(test_set[0]);
    gt_labels=list(test_set[1]);
    azimuths=[];
    for idx_img_path,img_path in enumerate(img_paths):
        if idx_img_path%10==0:
            print idx_img_path,len(img_paths);
        image_name=img_path[img_path.rindex('/')+1:-4];
        image_name_split=image_name.split('_');
        image_name_pre=image_name_split[0]+'_'+image_name_split[1];
        pascal_class=image_name_split[2];
        obj_index=int(image_name_split[3]);
        mat_file=os.path.join(os.path.join(path_to_anno,pascal_class+'_pascal'),image_name_pre+'.mat')
        try:
            azimuth=getCoarseAzimuth(mat_file,obj_index)
        except:
            print 'error'
            azimuth=-1;
        azimuths.append(azimuth);
    pickle.dump([img_paths,gt_labels,azimuths],open(file_name+'_azimuths.p','wb'));
        
        


def script_saveIndicesAll(file_name,layers):
    test_set,_=pickle.load(open(file_name+'.p','rb'));
    vals=np.load(file_name+'.npz');

    test_set=sorted(test_set,key=lambda x: x[0])
    test_set=zip(*test_set);
    
    img_paths=list(test_set[0]);
    gt_labels=list(test_set[1]);
    
    numberOfN=None;
    for layer in layers:
        print layer
        file_name_l=file_name+'_'+layer+'_all';
        indices,conf_matrix=script_nearestNeigbourExperiment.doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute')
        pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));

def getCoarseAzimuth(file_name,obj_index):
    curr_dict=scipy.io.loadmat(file_name,squeeze_me=True, struct_as_record=False);
    objects=curr_dict['record'].objects
    
    if not hasattr(objects, '__iter__'):
        objects=[objects]

    assert len(objects)>obj_index;
    obj=objects[obj_index];
    assert 'viewpoint' in obj.__dict__.keys();
    assert 'azimuth_coarse' in obj.viewpoint.__dict__.keys();
    return obj.viewpoint.azimuth_coarse



def script_compareAzimuth():
    path_to_anno='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations/chair_pascal';
    im_dir='/disk2/pascal_3d/PASCAL3D+_release1.0/Images_BB';
    out_file_html='/disk2/pascal_3d/PASCAL3D+_release1.0/Images_BB/chair_angle_check.html'
    anno_files=[os.path.join(path_to_anno,file_name) for file_name in os.listdir(path_to_anno) if file_name.endswith('.mat')];

    list_of_chairs=[];
    for anno_file in anno_files:
        just_file=anno_file[anno_file.rindex('/')+1:];
        just_file=just_file[:-4];

        curr_dict=scipy.io.loadmat(anno_file,squeeze_me=True, struct_as_record=False);
        objects=curr_dict['record'].objects
        if not hasattr(objects, '__iter__'):
            objects=[objects]
        for idx,obj in enumerate(objects):
            if obj.__dict__['class']=='chair':
                im_file=os.path.join(im_dir,just_file+'_chair_'+str(idx)+'.jpg');
                list_of_chairs.append((im_file,obj.viewpoint.azimuth_coarse));
    angles=list(zip(*list_of_chairs)[1]);
    images=list(zip(*list_of_chairs)[0]);
    angles=np.array(angles)
    angles_uni=np.unique(angles);
    col_im=[];
    col_caption=[];
    for angle_uni in angles_uni:
        idx_uni=np.where(angles==angle_uni)[0];
        row_im_curr=[];
        row_caption_curr=[];
        for idx_curr in range(min(5,len(idx_uni))):
            idx_im=idx_uni[idx_curr]
            image_just_name=images[idx_im]
            image_just_name=image_just_name[image_just_name.rindex('/')+1:];
            row_im_curr.append(image_just_name);
            row_caption_curr.append(str(angle_uni));
        col_im.append(row_im_curr);
        col_caption.append(row_caption_curr);

    print col_im[:5];
    print col_caption[:5];
    
    visualize.writeHTML(out_file_html,col_im,col_caption)

def script_createComparativeHtmls():
    layers=['pool5','fc6','fc7'];
    path_to_anno='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations';
    file_dir='/disk2/pascal_3d/PASCAL3D+_release1.0/Images_BB';
    dirs=[dir[:-7] for dir in os.listdir(path_to_anno) if dir.endswith('pascal')];
    file_name='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    file_name_alt='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    replace_paths=['/disk2','../../../..']
    out_file_pre='nn_performance_comparison_trained_notrained'
    out_file_pre=os.path.join('/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d',out_file_pre);
    for layer in layers:
        file_name_l=file_name+'_'+layer;
        [img_paths,gt_labels,indices,_]=pickle.load(open(file_name_l+'.p','rb'));

        idx_sort_binned=script_nearestNeigbourExperiment.sortByPerformance(indices,gt_labels,1,perClass=True);
        
        img_paths=[x.replace(replace_paths[0],replace_paths[1]) for x in img_paths];
        im_paths,captions=visualize.createImageAndCaptionGrid(img_paths,gt_labels,indices,dirs)

        file_name_l=file_name_alt+'_'+layer;
        [img_paths_alt,gt_labels_alt,indices,_]=pickle.load(open(file_name_l+'.p','rb'));

        img_paths_alt=[x.replace(replace_paths[0],replace_paths[1]) for x in img_paths_alt];
        
        im_paths_alt,captions_alt=visualize.createImageAndCaptionGrid(img_paths,gt_labels,indices,dirs)        
        
        im_paths_alt=[im_paths_alt[img_paths_alt.index(curr_img_path)] for curr_img_path in img_paths];
        captions_alt=[captions_alt[img_paths_alt.index(curr_img_path)] for curr_img_path in img_paths];

        im_paths_big=[];
        captions_big=[];
        for idx_curr in idx_sort_binned:
            im_paths_big.append(im_paths[idx_curr]);
            im_paths_big.append(im_paths_alt[idx_curr]);
            captions_big.append(captions[idx_curr]);
            captions_big.append(captions_alt[idx_curr]);
            
        visualize.writeHTML(out_file_pre+'_'+layer+'.html',im_paths_big,captions_big)

def script_visualizePerformanceDifference():
    trained_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    notrained_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    out_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/nn_performance_comparison_trained_notrained.png';
    out_file_diff='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/nn_performance_comparison_trained_notrained_diff.png';

    file_pres=[trained_file,notrained_file]
    layers=['pool5','fc6','fc7'];
    legend_pres=['Trained','Unfamiliar'];
    legend_entries=[];
    for leg in legend_pres:
        legend_entries.extend([leg+' '+layer for layer in layers]);

    vecs_to_plot=[];
    file_names=[file_pre+'_'+layer for file_pre in file_pres for layer in layers];
    for idx,file_name in enumerate(file_names):
        print file_name,legend_entries[idx];
        [img_paths,gt_labels,indices,_]=pickle.load(open(file_name+'.p','rb'));
        no_correct,_=script_nearestNeigbourExperiment.getNumberOfCorrectNNMatches(indices,gt_labels)
        with open(file_name+'.txt','wb') as f:
            for num in no_correct:
                f.write(str(num)+' ');
            f.write('\n');
        # with open(file_name+'.txt','rb') as f:
        #     no_correct=f.readline();
        # no_correct=[float(no_correct_curr) for no_correct_curr in no_correct.strip('\n').split()];
        vecs_to_plot.append(no_correct);

    # print legend_entries;
    
    print vecs_to_plot;
    # print len(vecs_to_plot);
    # print len(legend_entries);

    plt.figure();
    plt.xlabel('Number of Nearest Neighbours K');
    plt.ylabel('Accuracy');
    plt.title('NN Accuracy DNN Features for Pascal Classes'); 
    plt.xlim(0,6);
    plt.ylim(min([min(vec) for vec in vecs_to_plot])-0.05,max([max(vec) for vec in vecs_to_plot])+0.05);
    handles=[];
    for vec in vecs_to_plot:
        handle,=plt.plot(range(1,len(vec)+1),vec);
        handles.append(handle);
        
    plt.legend(handles, legend_entries,loc=2,prop={'size':10});
    
    plt.savefig(out_file);

    legend_entries=['Trained-Untrained '+layer for layer in layers];
    diffs=[];
    for idx in range(3):
        a=vecs_to_plot[idx];
        b=vecs_to_plot[idx+3];
        diff=[a[idx_curr]-b[idx_curr] for idx_curr in range(len(a))];
        diffs.append(diff);
    vecs_to_plot=diffs;
    plt.figure();
    plt.xlabel('Number of Nearest Neighbours K');
    plt.ylabel('Accuracy Difference');
    plt.title('NN Accuracy DNN Features for Pascal Classes'); 
    plt.xlim(0,6);
    plt.ylim(min([min(vec) for vec in vecs_to_plot])-0.01,max([max(vec) for vec in vecs_to_plot])+0.01);
    handles=[];
    for vec in vecs_to_plot:
        handle,=plt.plot(range(1,len(vec)+1),vec);
        handles.append(handle);
        
    plt.legend(handles, legend_entries,loc=2,prop={'size':10});

    plt.savefig(out_file_diff);


def main():
    train_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    layers=['pool5','fc6','fc7'];
    path_to_anno='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations'
    script_saveAzimuthInfo(train_file,path_to_anno);
    script_saveAzimuthInfo(non_train_file,path_to_anno);
    # script_saveIndicesAll(train_file,layers)
    # script_saveIndicesAll(non_train_file,layers)

    return
    out_dir='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    # out_dir=os.path.join(out_dir,'no_trained');
    out_dir=os.path.join(out_dir,'trained');
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);

    path_to_anno='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations';
    file_dir='/disk2/pascal_3d/PASCAL3D+_release1.0/Images_BB';
    dirs=[dir[:-7] for dir in os.listdir(path_to_anno) if dir.endswith('pascal')];
    test_set=[];
    for dir_idx,dir in enumerate(dirs):
        ims=[filename for filename in glob.glob(file_dir + '/*'+dir+'*.jpg')]
        test_set.extend(zip(ims,[dir_idx]*len(ims)));
    
    print len(test_set);

    layers=['pool5','fc6','fc7'];
    gpu_no=1
    path_to_classify='..';
    numberOfN=5
    relativePaths=['/disk2','../../../../..'];
    deployFile='/disk2/octoberExperiments/nn_performance_without_pascal/deploy.prototxt'
    meanFile='/disk2/octoberExperiments/nn_performance_without_pascal/mean.npy'
    modelFile='/disk2/octoberExperiments/nn_performance_without_pascal/snapshot_iter_450000.caffemodel'
    # file_name=script_nearestNeigbourExperiment.runClassificationTestSet(test_set,out_dir,path_to_classify,gpu_no,layers,deployFile=deployFile,meanFile=meanFile,modelFile=modelFile,ext='jpg')
    # file_name=script_nearestNeigbourExperiment.runClassificationTestSet(test_set,out_dir,path_to_classify,gpu_no,layers,ext='jpg')
    file_name=os.path.join(out_dir,'20151027204114');
    test_set,_=pickle.load(open(file_name+'.p','rb'));
    vals=np.load(file_name+'.npz');

    test_set=sorted(test_set,key=lambda x: x[0])
    test_set=zip(*test_set);
    
    img_paths=list(test_set[0]);
    gt_labels=list(test_set[1]);
    
    numberOfN=5;


    # file_name_alt='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203004'
    for layer in layers:
        print layer
        file_name_l=file_name+'_'+layer;
        # indices,conf_matrix=script_nearestNeigbourExperiment.doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute')
        # pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));
        [img_paths,gt_labels,indices,_]=pickle.load(open(file_name_l+'.p','rb'));
        
        idx_sort_binned=script_nearestNeigbourExperiment.sortByPerformance(indices,gt_labels,0,perClass=True);

        img_paths=[x.replace('/disk2','../../../../..') for x in img_paths];
        im_paths,captions=visualize.createImageAndCaptionGrid(img_paths,gt_labels,indices,dirs)
        im_paths=[im_paths[idx] for idx in idx_sort_binned];
        captions=[captions[idx] for idx in idx_sort_binned];

        visualize.writeHTML(file_name_l+'_sorted.html',im_paths,captions)





    return
    path_to_anno='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations';
    path_to_im='/disk2/pascal_3d/PASCAL3D+_release1.0/Images';
    dirs=[dir for dir in os.listdir(path_to_anno) if dir.endswith('pascal')];
    
    out_dir='/disk2/pascal_3d/PASCAL3D+_release1.0/Images_BB';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    
    for dir in dirs:
        file_names=[os.path.join(dir,file_name)[:-4] for file_name in os.listdir(os.path.join(path_to_im,dir)) if file_name.endswith('.jpg')];
        saveBBImages(path_to_im,path_to_anno,file_names,out_dir);

if __name__=='__main__':
    main();