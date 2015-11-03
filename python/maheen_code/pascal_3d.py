# 
import os;
import scipy;
import mat4py;
from scipy import misc;
import scipy.io
import visualize;
import numpy as np;
import glob
import script_nearestNeigbourExperiment
import cPickle as pickle;
import matplotlib.pyplot as plt;

import time;

def script_savePerClassPerDegreeHistograms():
    train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    dirs=[dir[:-7] for dir in os.listdir('/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations') if dir.endswith('_pascal')];
    layers=['pool5','fc6','fc7'];
    degrees=[0,45,90,135,180];
    # degrees=[90];
    # dirs=['train']
    delta=5;
    for file_pre in [train_pre,non_train_pre]:
        for layer in layers:
            curr_dir=os.path.join(file_pre+'_'+layer+'_all_azimuths');
            for dir in dirs:
                print dir
                curr_file=os.path.join(curr_dir,dir+'_data.p');
                [diffs_curr,dists_curr]=pickle.load(open(curr_file,'rb'));
                for degree in degrees:
                    
                    title=dir+' '+str(degree)+' delta '+str(delta)

                    out_file=os.path.join(curr_dir,dir+'_'+str(degree)+'_'+str(delta)+'_compress_data.p')
                    print out_file
                    try:
                        hists,bins=getDistanceHistograms(diffs_curr,degree,delta=delta,normed=True,bins=10);
                        pickle.dump([hists,bins],open(out_file,'wb'));                        

                        out_file=os.path.join(curr_dir,dir+'_'+str(degree)+'_'+str(delta)+'_non_compress_data.p')
                        hists,bins=getDistanceHistograms(diffs_curr,degree,dists_curr=dists_curr,delta=delta,normed=True,bins=10);

                        pickle.dump([hists,bins],open(out_file,'wb'));
                    except:
                        print 'error'
                        print out_file

                    # out_file=os.path.join(curr_dir,dir+'_'+str(degree)+'_'+str(delta)+'_compress.png');
                    # # print out_file

                    # try:
                    #     visualize.plotDistanceHistograms(diffs_curr,degree,out_file,title=title,delta=delta,dists_curr=None,bins=10,normed=True)
                    #     out_file=os.path.join(curr_dir,dir+'_'+str(degree)+'_'+str(delta)+'_non_compress.png');

                    #     visualize.plotDistanceHistograms(diffs_curr,degree,out_file,title=title,delta=delta,dists_curr=dists_curr,bins=10,normed=True)
                    # except:
                    #     print 'error'
                    #     print out_file

def getDistanceHistograms(diffs_curr,degree,delta=0,dists_curr=None,bins=10,normed=False):

    if dists_curr is None:
        dists_curr=np.array(range(1,diffs_curr.shape[1]+1));
        dists_curr=np.expand_dims(dists_curr,0);
        dists_curr=np.repeat(dists_curr,diffs_curr.shape[0],0);

    diffs=diffs_curr-degree;
    diffs=abs(diffs);
    idx=np.where(diffs<=delta)
    dists=dists_curr[idx[0],idx[1]];
    hist,bin_edges=np.histogram(dists, bins=bins, normed=normed)
    return hist,bin_edges



def script_createHistDifferenceHTML():
    out_dir_meta='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d';
    train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    dirs=[dir[:-7] for dir in os.listdir('/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations') if dir.endswith('_pascal')];
    layers=['pool5','fc6','fc7'];
    degrees=[0,45,90,135,180];
    delta=5;
    caption_text=['Trained','Not Trained'];
    replace=[out_dir_meta+'/',''];
    degree=90;
    for layer in layers:
        out_file_html=os.path.join(out_dir_meta,layer+'_all_azimuths'+'.html')

        img_paths=[];
        caption_paths=[];
        for dir in dirs:
            

            img_paths_row=[];
            caption_paths_row=[];    

            for idx,file_pre in enumerate([train_pre,non_train_pre]):        
                curr_dir=os.path.join(file_pre+'_'+layer+'_all_azimuths');
                im_file=os.path.join(curr_dir,dir+'_'+str(degree)+'_'+str(delta)+'_compress.png');
                
                img_paths_row.append(im_file.replace(replace[0],replace[1]));
                caption_paths_row.append(caption_text[idx]+' '+layer+' '+dir);

            img_paths.append(img_paths_row);
            caption_paths.append(caption_paths_row);
        
        visualize.writeHTML(out_file_html,img_paths,caption_paths,height=400,width=400);
        print out_file_html

def script_createHistsWithSpecificAngleHtml():
    out_dir_meta='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d';
    train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    dirs=[dir[:-7] for dir in os.listdir('/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations') if dir.endswith('_pascal')];
    layers=['pool5','fc6','fc7'];
    deg_to_see=0;
    degree=90;
    delta=5;

    out_file_html=os.path.join('/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d','hist_angle_restrict_'+str(deg_to_see)+'_'+str(degree)+'_comparison_non_compress.html');
    replace=['/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/',''];

    img_paths=[];
    captions=[];
    for dir in dirs:
        for layer in layers:
            single_row=[];
            single_row_caption=[];
            for caption_curr,file_pre in [('Trained',train_pre),('Not trained',non_train_pre)]:
                curr_dir=file_pre+'_'+layer+'_all_azimuths'
                img_path=os.path.join(curr_dir,dir+'_'+str(deg_to_see)+'_'+str(degree)+'_'+str(delta)+'_non_compress.png');
                img_path=img_path.replace(replace[0],replace[1]);
                single_row.append(img_path);
                single_row_caption.append(caption_curr+' '+dir+' '+layer);
            img_paths.append(single_row);
            captions.append(single_row_caption);

    visualize.writeHTML(out_file_html,img_paths,captions,height=300,width=400)


def script_createHistsWithSpecificAngle():
    out_dir_meta='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d';
    train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    dirs=[dir[:-7] for dir in os.listdir('/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations') if dir.endswith('_pascal')];
    layers=['pool5','fc6','fc7'];
    deg_to_see=0;
    degree=90;
    delta=5;

    for file_pre in [train_pre,non_train_pre]:
        azimuth_file=file_pre+'_azimuths.p';
        for dir in dirs:
            # dir='car';
            for layer in layers:
                # ='fc7';
                
                curr_dir=file_pre+'_'+layer+'_all_azimuths'
                class_data_file=os.path.join(curr_dir,dir+'_data.p');
                [img_paths,gt_labels,azimuths]=pickle.load(open(azimuth_file,'rb'));
                [diffs_all,dists_all]=pickle.load(open(class_data_file,'rb'));
                idx=np.array(gt_labels);
                idx=np.where(idx==dirs.index(dir))[0];
                azimuths_rel=np.array(azimuths);
                azimuths_rel=azimuths_rel[idx];
                idx_deg=np.where(azimuths_rel==deg_to_see)[0];
                diffs_curr=diffs_all[idx_deg,:];
                dists_curr=dists_all[idx_deg,:]

                print diffs_curr.shape
                print dists_curr.shape

                out_file=os.path.join(curr_dir,dir+'_'+str(deg_to_see)+'_'+str(degree)+'_'+str(delta)+'_compress.png');
                                # # print out_file
                title=dir+' with angle '+str(deg_to_see)+' with '+str(degree)+' difference'
                visualize.plotDistanceHistograms(diffs_curr,degree,out_file,title=title,delta=delta,dists_curr=None,bins=10,normed=True)
                hists,bin_edges=getDistanceHistograms(diffs_curr,degree,delta=delta,dists_curr=None,bins=10,normed=True);
                pickle.dump([hists,bin_edges],open(out_file[:-2],'wb'));

                out_file=os.path.join(curr_dir,dir+'_'+str(deg_to_see)+'_'+str(degree)+'_'+str(delta)+'_non_compress.png');
                visualize.plotDistanceHistograms(diffs_curr,degree,out_file,title=title,delta=delta,dists_curr=dists_curr,bins=10,normed=True)
                hists,bin_edges=getDistanceHistograms(diffs_curr,degree,delta=delta,dists_curr=dists_curr,bins=10,normed=True);
                pickle.dump([hists,bin_edges],open(out_file[:-2],'wb'));

                

def script_createHistComparative():
    out_dir_meta='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d';
    train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_pre='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    dirs=[dir[:-7] for dir in os.listdir('/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations') if dir.endswith('_pascal')];
    layers=['pool5','fc6','fc7'];
    delta=5;
    caption_text=['Trained','Not Trained'];
    replace=[out_dir_meta+'/',''];
    degree=90;
    deg_to_see=0;
    # train_files=[os.path.join(train_pre+'_'+layer+'_all_azimuths',dir+'_'+str(degree)+'_'+str(delta)+'_compress_data.p') for layer in layers for dir in dirs];
    # non_train_files=[os.path.join(non_train_pre+'_'+layer+'_all_azimuths',dir+'_'+str(degree)+'_'+str(delta)+'_compress_data.p') for layer in layers for dir in dirs];
    # for idx in range(len(train_files)):

    combos=[(dir,layer) for dir in dirs for layer in layers];
    out_file_html=os.path.join(out_dir_meta,'hist_by_degree_'+str(degree)+'_comparisons_compress.html');
    img_paths=[];
    captions=[];

    for dir,layer in combos:

        file_train=os.path.join(train_pre+'_'+layer+'_all_azimuths',dir+'_'+str(degree)+'_'+str(delta)+'_compress_data.p');
        # train_files[idx];
        file_non_train=os.path.join(non_train_pre+'_'+layer+'_all_azimuths',dir+'_'+str(degree)+'_'+str(delta)+'_compress_data.p');
        # non_train_files[idx];

        hists_train,bins_train=pickle.load(open(file_train,'rb'));
        hists_non_train,bins_non_train=pickle.load(open(file_non_train,'rb'));
        
        mid_points_train=[bins_train[i]+bins_train[i+1]/float(2) for i in range(len(bins_train)-1)];
        mid_points_non_train=[bins_non_train[i]+bins_non_train[i+1]/float(2) for i in range(len(bins_non_train)-1)];
        
        # dir=file_train[file_train.rindex('/')+1:];
        # dir=dir[:dir.index('_')];
        out_file_just_file=layer+'_'+dir+'_'+str(degree)+'_'+str(delta)+'.png'
        out_file=os.path.join(out_dir_meta,out_file_just_file)
        title=dir+' Comparison';
        xlabel='Distance Rank';
        ylabel='Frequency';

        # print out_file
        img_paths.append([out_file_just_file]);
        captions.append([dir+' '+layer]);

        visualize.plotSimple(zip([mid_points_train,mid_points_non_train],[hists_train,hists_non_train]),out_file,title=title,xlabel=xlabel,ylabel=ylabel,legend_entries=['Trained','Non Trained'],loc=0);
    print out_file_html
    visualize.writeHTML(out_file_html,img_paths,captions,width=400,height=400);
        # return

    # for layer in layers:
    #     out_file_html=os.path.join(out_dir_meta,layer+'_all_azimuths'+'.html')

    #     img_paths=[];
    #     caption_paths=[];
    #     for dir in dirs:
            

    #         img_paths_row=[];
    #         caption_paths_row=[];    

    #         for idx,file_pre in enumerate([train_pre,non_train_pre]):        
    #             curr_dir=os.path.join(file_pre+'_'+layer+'_all_azimuths');
    #             im_file=os.path.join(curr_dir,dir+'_'+str(degree)+'_'+str(delta)+'_compress.png');
                
    #             img_paths_row.append(im_file.replace(replace[0],replace[1]));
    #             caption_paths_row.append(caption_text[idx]+' '+layer+' '+dir);

    #         img_paths.append(img_paths_row);
    #         caption_paths.append(caption_paths_row);
        
    #     visualize.writeHTML(out_file_html,img_paths,caption_paths,height=400,width=400);
    #     print out_file_html


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


def getAzimuthInfo(im_paths,gt_labels,indices,azimuth):
    diffs_all=[];
    dists_all=[];
    for r in range(indices.shape[0]):
        gt_label=gt_labels[r];
        gt_azi=azimuth[r];
        diffs=[];
        dists=[];
        for c in range(indices.shape[1]):
            pred_idx=indices[r,c];
            pred_label=gt_labels[pred_idx]
            pred_azi=azimuth[pred_idx];
            if pred_label==gt_label:
                diff=abs(gt_azi-pred_azi);
                if diff>180:
                    diff=(360-diff)%180;
                diffs.append(diff);
                dists.append(c);
        diffs_all.append(diffs);
        dists_all.append(dists);

    return diffs_all,dists_all;

def getPerClassInfo(gt_labels,diffs_all,dists_all):
    gt_labels=np.array(gt_labels);
    gt_labels_uni=np.unique(gt_labels);
    diffs_dists_by_label=[];
    for gt_label in gt_labels_uni:
        idx_gt=np.where(gt_labels==gt_label)[0];
        diffs_curr=[diffs_all[idx] for idx in idx_gt];
        dists_curr=[dists_all[idx] for idx in idx_gt];
        diffs_curr=np.array(diffs_curr);
        diffs_curr=diffs_curr[:,:-1];
        dists_curr=np.array(dists_curr);
        dists_curr=dists_curr[:,:-1];
        curr_combo=(diffs_curr,dists_curr);
        diffs_dists_by_label.append(curr_combo);
    return diffs_dists_by_label,gt_labels_uni

def script_visualizePerClassAzimuthPerformance():
    train_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    dirs=[dir[:-7] for dir in os.listdir('/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations') if dir.endswith('_pascal')];
    file_pres=[train_file,non_train_file];
    layers=['pool5','fc6','fc7'];
    for file_name in file_pres:
        for layer in layers:
            in_file=file_name+'_'+layer+'_all_azimuths.p';
            print in_file
            t=time.time();
            [img_paths,gt_labels,azimuths,diffs_all,dists_all]=pickle.load(open(in_file,'rb'));
            t=time.time()-t;
            # print t;
            diffs_dists_by_label,gt_labels_uni=getPerClassInfo(gt_labels,diffs_all,dists_all);

            
            out_dir=in_file[:-2];
            if not os.path.exists(out_dir):
                os.mkdir(out_dir);
                
            for idx_gt_label,gt_label in enumerate(gt_labels_uni):
                diffs_curr,dists_curr=diffs_dists_by_label[idx_gt_label];
                out_file=os.path.join(out_dir,dirs[gt_label]+'_data.p');
                pickle.dump([diffs_curr,dists_curr],open(out_file,'wb'));

                title=dirs[gt_label]+' Distances Versus Viewpoint Difference'
                xlabel='Distance'
                ylabel='Viewpoint Difference in Degree'

                out_file=os.path.join(out_dir,dirs[gt_label]+'_compress.png');
                visualize.createScatterOfDiffsAndDistances(diffs_curr,title,xlabel,ylabel,out_file);
                out_file=os.path.join(out_dir,dirs[gt_label]+'_non_compress.png');
                visualize.createScatterOfDiffsAndDistances(diffs_curr,title,xlabel,ylabel,out_file,dists_curr);


def main():
    script_createHistComparative();
    # script_createHistsWithSpecificAngle()
    return
    script_createHistComparative()
    # script_createHistDifferenceHTML()
    # script_savePerClassPerDegreeHistograms()
    return
    script_visualizePerClassAzimuthPerformance();
    return
    train_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/trained/20151027204114'
    non_train_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547'
    layers=['pool5','fc6','fc7'];
    path_to_anno='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations'

    for file_name in [train_file,non_train_file]:
        [img_paths,gt_labels,azimuths]=pickle.load(open(file_name+'_azimuths.p','rb'));
        for layer in layers:
            print layer;
            file_name_l=file_name+'_'+layer+'_all';
            out_file=file_name_l+'_azimuths.p';

            t=time.time()
            [img_paths,gt_labels,indices,_]=pickle.load(open(file_name_l+'.p','rb'));
            t=time.time()-t
            print t
            # raw_input();
            
            diffs_all,dists_all=getAzimuthInfo(img_paths,gt_labels,indices,azimuths)
            pickle.dump([img_paths,gt_labels,azimuths,diffs_all,dists_all],open(out_file,'wb'));

    return    
    text_labels=[dir[:-7] for dir in os.listdir(path_to_anno) if dir.endswith('pascal')];
    for file_name in [train_file,non_train_file]:
        [img_paths,gt_labels,azimuths]=pickle.load(open(file_name+'_azimuths.p','rb'));
        for layer in layers:
            print layer;
            file_name_l=file_name+'_'+layer+'_all';
            out_dir=file_name_l+'_azimuths';
            if not os.path.exists(out_dir):
                os.mkdir(out_dir);
            t=time.time()
            [img_paths,gt_labels,indices,_]=pickle.load(open(file_name_l+'.p','rb'));
            t=time.time()-t
            print t
            # raw_input();
            createAzimuthGraphs(img_paths,gt_labels,indices,azimuths,out_dir,text_labels)

    for layer in layers:
        print layer
        out_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/azimuths_'+layer+'_all'+'_comparison.html';
        rel_train='trained/20151027204114_'+layer+'_all'+'_azimuths'
        rel_notrain='no_trained/20151027203547_'+layer+'_all'+'_azimuths';
        out_dir='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_3d/no_trained/20151027203547_'+layer+'_all'+'_azimuths'

        im_paths=[[os.path.join(rel_train,file_curr),os.path.join(rel_notrain,file_curr)] for file_curr in os.listdir(out_dir) if file_curr.endswith('.jpg')];
        
        captions=[['train','no_train']]*len(im_paths);
        visualize.writeHTML(out_file,im_paths,captions,height=500,width=500)


    # script_saveAzimuthInfo(train_file,path_to_anno);
    # script_saveAzimuthInfo(non_train_file,path_to_anno);
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