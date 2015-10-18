
import os;
import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)));
import urllib
import urllib2
import re
import subprocess
import numpy as np;
import random;
import time
import shutil
import pickle
import sklearn
import sklearn.neighbors as neighbors
import collections
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt;
import math;
import glob;

def doNN(img_paths,gt_labels,features_curr,numberOfN=5,distance='cosine',algo='brute',binarize=False): 
    gt_labels_uni=list(set(gt_labels));
    gt_labels_uni.sort();

    conf_matrix=np.zeros((len(gt_labels_uni),len(gt_labels_uni)));

    nn=neighbors.NearestNeighbors(n_neighbors=numberOfN+1,metric=distance,algorithm=algo);

    feature_len=1;
    for dim_val in list(features_curr.shape)[1:]:
        feature_len *= dim_val
    
    features_curr=np.reshape(features_curr,(features_curr.shape[0],feature_len));
    if binarize:
        features_curr[features_curr!=0]=1;
    
    features_curr=sklearn.preprocessing.normalize(features_curr, norm='l2', axis=1);
    distances=np.dot(features_curr,features_curr.T);
    np.fill_diagonal(distances,0.0);

    indices=np.argsort(distances, axis=1)[:,::-1]
    indices=indices[:,:numberOfN];
    
    for row in range(indices.shape[0]):
        for col in range(len(indices[0])):
                gt_label=gt_labels[row];
                pred_label=gt_labels[indices[row,col]];
                conf_matrix[gt_labels_uni.index(gt_label),gt_labels_uni.index(pred_label)]+=1;
    
    return indices, conf_matrix;

def randomlySelectTestSet(path_to_val,val_gt_file,no_classes=50,no_im=5):

    
    f=open(val_gt_file,'rb');
    gt_classes=[x.strip('\n').split(' ') for x in f.readlines()];
    f.close();
    
    im_list=[row[0] for row in gt_classes]
    gt_classes=[int(row[1]) for row in gt_classes]

    gt_classes=np.array(gt_classes);
    uni_classes=np.unique(gt_classes);
    rand_classes=random.sample(uni_classes,min(no_classes,len(uni_classes)));

    idx_chosen_images=[];
    for rand_class in rand_classes:
        idx_rand_class=np.where(gt_classes==rand_class)[0];
        idx_chosen_images.extend(random.sample(idx_rand_class,min(no_im,len(idx_rand_class))));
        
    im_list_chosen=[os.path.join(path_to_val,im_list[idx_curr]) for idx_curr in idx_chosen_images];
    gt_class_chosen=gt_classes[idx_chosen_images];

    return zip(im_list_chosen,gt_class_chosen);

def runClassificationTestSet(test_set,out_dir,path_to_classify,gpu_no,layers,ext='JPEG',central_crop=True):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    
    # test_set=randomlySelectTestSet(path_to_val,val_gt_file,no_classes=no_classes,no_im=no_im)
    
    out_file= time.strftime("%Y%m%d%H%M%S");
    pickle.dump([test_set,layers],open(os.path.join(out_dir,out_file+'.p'),'wb'));
    
    temp_dir=out_dir+'_temp';
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir);

    for im_path,gt_val in test_set:
        file_name=im_path[im_path.rfind('/')+1:]
        shutil.copy(im_path,os.path.join(temp_dir,file_name));

    out_file=os.path.join(out_dir,out_file);
    command=[os.path.join(path_to_classify,'classify.py'),temp_dir,out_file,'--ext',ext,'--gpu',str(gpu_no),'--layer']+layers;

    if central_crop:
        command=command+["--center_only"]
    command_formatted=' '.join(command);
    subprocess.call(command_formatted, shell=True)
    return out_file;


def runClassification(path_to_val,out_dir,val_gt_file,path_to_classify,gpu_no,layers,no_classes=50,no_im=5,ext='JPEG',central_crop=True):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    
    test_set=randomlySelectTestSet(path_to_val,val_gt_file,no_classes=no_classes,no_im=no_im)
    
    out_file= time.strftime("%Y%m%d%H%M%S");
    pickle.dump([test_set,layers],open(os.path.join(out_dir,out_file+'.p'),'wb'));
    
    temp_dir=out_dir+'_temp';
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir);

    for im_path,gt_val in test_set:
        file_name=im_path[im_path.rfind('/')+1:]
        shutil.copy(im_path,os.path.join(temp_dir,file_name));

    out_file=os.path.join(out_dir,out_file);
    command=[os.path.join(path_to_classify,'classify.py'),temp_dir,out_file,'--ext',ext,'--gpu',str(gpu_no),'--layer']+layers;

    if central_crop:
        command=command+["--center_only"]
    command_formatted=' '.join(command);
    subprocess.call(command_formatted, shell=True)
    return out_file;

def writeHTML(file_name,im_paths,captions,height=200,width=200):
    f=open(file_name,'w');
    html=[];
    f.write('<!DOCTYPE html>\n');
    f.write('<html><body>\n');
    f.write('<table>\n');
    for row in range(len(im_paths)):
        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td>');
            f.write(captions[row][col]);
            f.write('</td>');
            f.write('    ');
        f.write('\n</tr>\n');

        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td><img src="');
            f.write(im_paths[row][col]);
            f.write('" height='+str(height)+' width='+str(width)+'"/></td>');
            f.write('    ');
        f.write('\n</tr>\n');
        f.write('<p></p>');
    f.write('</table>\n');
    f.close();

def createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels):
    im_paths=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    captions=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    for r in range(indices.shape[0]):
        im_paths[r][0]=img_paths[r];
        captions[r][0]='GT class \n'+text_labels[gt_labels[r]]+' '+str(gt_labels[r]);
        for c in range(indices.shape[1]):
            pred_idx=indices[r][c]
            im_paths[r][c+1]=img_paths[pred_idx];
            if gt_labels[pred_idx] !=gt_labels[r]:
                captions[r][c+1]='wrong \n'+text_labels[gt_labels[pred_idx]]+' '+str(gt_labels[pred_idx]);
            else:
                captions[r][c+1]='';
    return im_paths,captions

def script_classificationOld():

    # return
    path_to_classify='..';
    path_to_val='/disk2/imagenet/val';
    out_dir='/disk2/octoberExperiments/nn_trained_vs_notrained';
    no_classes=100;
    no_im=50;
    gpu_no=1;
    # val_gt_file='/disk2/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    val_gt_file='../../data/ilsvrc12/val.txt'
    file_text_labels='../../data/ilsvrc12/synset_words.txt'
    ext='JPEG';
    layers=['pool5','fc6','fc7'];
    central_crop=True;
    
    text_labels= np.loadtxt(file_text_labels, str, delimiter='\t')
    
    test_set=randomlySelectTestSet(path_to_val,val_gt_file,no_classes=no_classes,no_im=no_im)
    
    file_name=runClassification(path_to_val,out_dir,val_gt_file,path_to_classify,gpu_no,layers,no_classes=no_classes,no_im=no_im,ext=ext,central_crop=central_crop);

    # file_name=os.path.join(out_dir,'20151009153119');
    # file_name=os.path.join(out_dir,'20151009172027');
    # file_name=os.path.join(out_dir,'20151009202006');
    # file_name=os.path.join(out_dir,'20151010000109');
    test_set,_=pickle.load(open(file_name+'.p','rb'));
    vals=np.load(file_name+'.npz');

    test_set=sorted(test_set,key=lambda x: x[0])
    test_set=zip(*test_set);
    
    img_paths=list(test_set[0]);
    gt_labels=list(test_set[1]);
    
    numberOfN=5;

    for layer in layers:

        file_name_l=file_name+'_'+layer;
        indices,conf_matrix=doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute')
        pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));
        img_paths=[x.replace('/disk2','../..') for x in img_paths];
        im_paths,captions=createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels)
        writeHTML(file_name_l+'.html',im_paths,captions)

def script_createBigGTFile():
    path_to_xml='/disk2/imagenet/structure_files/structure_released.xml';
    with open(path_to_xml,'rb') as f:
        content=f.read();
    content=content.split('<');

    path_to_big_gt='/disk2/imagenet/structure_files/id_to_text_desc.txt';
    with open(path_to_big_gt,'wb') as f:
        for c in content:
            if c.startswith('synset wnid="n'):
                c=re.split('[=]',c);
                # print c
                id=c[1].rsplit(' ',1)[0].strip('"');
                words=c[2].rsplit(' ',1)[0].strip('"');
                # print id, words;
                f.write(id+' '+words+'\n');

def script_getPerformanceDiff(in_dir,layer='pool5'):
    files=os.listdir(in_dir);
    files=[f[:-4] for f in files if f.endswith('npz')];
    files=list(set(files));
    print len(files);
    difs_all=[];
    for f in files:
        file_name=os.path.join(in_dir,f+'_'+layer+'.p');
        file_name_bin=os.path.join(in_dir,f+'_binary_'+layer+'.p');
        [_,gt_labels,indices,_]=pickle.load(open(file_name,'rb'));
        no_correct=getNumberOfCorrectNNMatches(indices,gt_labels);
        [_,gt_labels,indices,_]=pickle.load(open(file_name_bin,'rb'));
        no_correct_bin=getNumberOfCorrectNNMatches(indices,gt_labels);
        difs=[no_correct_bin[idx]-no_correct[idx] for idx in range(len(no_correct))];
        difs_all.append(difs);
    return difs_all;

def getNumberOfCorrectNNMatches(indices,gt_labels):
    labels_mat=np.zeros(indices.shape);

    for r in range(indices.shape[0]):
        for c in range(indices.shape[1]):
            pred_idx=indices[r,c];
            labels_mat[r,c]=gt_labels[pred_idx];

    gt_labels_tile=np.repeat(np.expand_dims(gt_labels,1),5,1);

    bin_correct=np.equal(labels_mat,gt_labels_tile);
    no_correct=[]

    for c in range(bin_correct.shape[1]):
        no_correct.append(sum(np.sum(bin_correct[:,:c+1],1)>0)/float(bin_correct.shape[0]));
    return no_correct

def script_createVisualization():


    path='/disk2/octoberExperiments/nn_trained_vs_notrained';
    # path=os.path.join(path,'notrained');
    file_name='accuracy_performance_binary'
    layers=['pool5','fc6','fc7'];

    layers_performance={};
    
    for layer in layers:
        layers_performance[layer]=[];

    # plt.figure();
    plt.title('Nearest Neighbors on Trained Classes');
    plt.xlabel('Number of Nearest Neighbors K');
    plt.ylabel('Accuracy');
    
    plt.xlim(0,6);
    min_val_seen=float('Inf');
    handles=[];
    # plt.legend(['POOL5','FC6','FC7'],loc='best');
    out_file_im=os.path.join(path,file_name+'.png');
    out_file_data=os.path.join(path,file_name+'.p');
    out_file_txt=os.path.join(path,file_name+'.txt');
    for curr_layer in layers:

        files=os.listdir(path);
        files=[os.path.join(path,f) for f in files if f.endswith('binary_'+curr_layer+'.p')];
        print len(files)

        no_correct_all=[];

        for f in files:

            [img_paths,gt_labels,indices,_]=pickle.load(open(f,'rb'));
            no_correct=getNumberOfCorrectNNMatches(indices,gt_labels);
            no_correct_all.append(no_correct);

        no_correct_all=np.array(no_correct_all);
        means=np.mean(no_correct_all,0);
        stds=np.std(no_correct_all,0);
        layers_performance[curr_layer]=[means,stds];
        min_val_seen=min(min_val_seen,min(means-stds));
        handle=plt.errorbar(range(1,6),means,yerr=stds);
        handles.append(handle);

    plt.ylim(min(min_val_seen,0.45)-0.05,1);
    plt.legend(handles, ['POOL5','FC6','FC7'],loc=2)
    plt.savefig(out_file_im);
    pickle.dump(layers_performance,open(out_file_data,'wb'));

    print 'layer,mean accuracy,std'
    with open(out_file_txt,'wb') as f:
        for k in layers_performance:
            f.write(k+' ');
            [mean_curr,std_curr]=layers_performance[k];
            for idx,mean_curr_curr in enumerate(mean_curr):
                mean_curr_curr=mean_curr_curr*100;
                std_curr_curr=std_curr[idx]*100;
                f.write('%.2f+,-%.2f '%(mean_curr_curr,std_curr_curr))
            f.write('\n');

    for k in layers_performance:
        print k,layers_performance[k][0],layers_performance[k][1]

            # print labels_mat_tile.shape
            # print indices.shape
            # raw_input();

def plotErrorBars(dict_to_plot,x_lim,y_lim,xlabel,y_label,title,out_file,margin=[0.05,0.05],loc=2):
    
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(y_label);
    
    if y_lim is None:
        y_lim=[1*float('Inf'),-1*float('Inf')];
    
    max_val_seen_y=y_lim[1]-margin[1];
    min_val_seen_y=y_lim[0]+margin[1];
    print min_val_seen_y,max_val_seen_y
    max_val_seen_x=x_lim[1]-margin[0];
    min_val_seen_x=x_lim[0]+margin[0];
    handles=[];
    for k in dict_to_plot:
        means,stds,x_vals=dict_to_plot[k];
        
        min_val_seen_y=min(min(np.array(means)-np.array(stds)),min_val_seen_y);
        max_val_seen_y=max(max(np.array(means)+np.array(stds)),max_val_seen_y);
        
        min_val_seen_x=min(min(x_vals),min_val_seen_x);
        max_val_seen_x=max(max(x_vals),max_val_seen_x);
        
        handle=plt.errorbar(x_vals,means,yerr=stds);
        handles.append(handle);
        print max_val_seen_y
    plt.xlim([min_val_seen_x-margin[0],max_val_seen_x+margin[0]]);
    plt.ylim([min_val_seen_y-margin[1],max_val_seen_y+margin[1]]);
    plt.legend(handles, dict_to_plot.keys(),loc=loc)
    plt.savefig(out_file);
    

def main():
    in_dir='/disk2/octoberExperiments/nn_trained_vs_notrained';
    layers=['pool5','fc6','fc7'];
    dict_to_plot=collections.OrderedDict();

    for layer in layers:
        difs=script_getPerformanceDiff(in_dir,layer);
        means=np.mean(difs,0);
        stds=np.std(difs,0);
        dict_to_plot[layer]=[means,stds,range(1,len(means)+1)]
    
    out_file='accuracy_difference_with_binary.png';
    out_file=os.path.join(in_dir,out_file);
    plotErrorBars(dict_to_plot,[0,6],None,'Number of Nearest Neighbors K','Difference in Performance','',out_file);

    return
    in_dir='/disk2/octoberExperiments/nn_trained_vs_notrained/notrained';
    layers=['pool5','fc6','fc7'];
    # file_text_labels='../../data/ilsvrc12/synset_words.txt'
    file_text_labels='/disk2/imagenet/structure_files/id_to_text_desc.txt';
    text_labels= np.loadtxt(file_text_labels, str, delimiter='\t')

    files=os.listdir(in_dir);
    files=[os.path.join(in_dir,f[:-4]) for f in files if f.endswith('npz')];

    for file_name in files:

        test_set,_=pickle.load(open(file_name+'.p','rb'));
        vals=np.load(file_name+'.npz');

        file_name=file_name+'_binary'

        test_set=sorted(test_set,key=lambda x: x[0])
        test_set=zip(*test_set);
        
        img_paths=list(test_set[0]);
        gt_labels=list(test_set[1]);
        
        numberOfN=5;

        for layer in layers:
            file_name_l=file_name+'_'+layer;
            indices,conf_matrix=doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute',binarize=True)
            pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));
            img_paths=[x.replace('/disk2','../../..') for x in img_paths];
            im_paths,captions=createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels)
            writeHTML(file_name_l+'.html',im_paths,captions)

    return


    no_classes=100;
    no_im=50;
    gpu_no=1;
    # val_gt_file='/disk2/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    val_gt_file='../../data/ilsvrc12/val.txt'
    file_text_labels='/disk2/imagenet/structure_files/id_to_text_desc.txt';
    link_to_words='http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=';
    # '../../data/ilsvrc12/synset_words.txt'
    ext='JPEG';
    layers=['pool5','fc6','fc7'];
    central_crop=True;
    
    text_labels= np.loadtxt(file_text_labels, str, delimiter='\t')
    
    path_to_classify='..';
    out_dir='/disk2/octoberExperiments/nn_trained_vs_notrained';
    out_dir=os.path.join(out_dir,'notrained');
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    # path_to_im='/disk2/imagenet/not_trained_matlab/'
    path_to_im='/disk2/imagenet/not_trained_im_tars/'
    ids_meta=pickle.load(open('select_ids_meta.p','rb'));
    
    
    with open(file_text_labels,'rb') as f:
        gt_labels=f.readlines();
    gt_labels=[gt_label.split(' ')[0] for gt_label in gt_labels];

    for iter in range(2,5):
        print 'ITERATION ',iter
        test_set=[]
        ids=ids_meta[iter];
        ids=ids;
        print len(ids)

        # list_of_files=os.listdir(path_to_im);
        for id in ids:
            # print os.path.join(path_to_im,id,'*.JPEG')
            im_list=glob.glob(os.path.join(path_to_im,id,'*.JPEG'));      
            
            # im_list=glob.glob(os.path.join(path_to_im,id+'_*.JPEG'));
            random.shuffle(im_list);
            im_list=im_list[:no_im];

            if id not in gt_labels:
                f = urllib2.urlopen(link_to_words+id)
                words = f.readline()
                f.close();
                words.strip('\r');
                words.strip('\n');
                str_to_append=id+' '+words+'\n';
                with open(file_text_labels, "a") as f:
                    f.write(str_to_append)  
                with open(file_text_labels,'rb') as f:
                    gt_labels=f.readlines();
                gt_labels=[gt_label.split(' ')[0] for gt_label in gt_labels];

            idx=gt_labels.index(id);
            im_list=[(path_curr,idx) for path_curr in im_list];
            test_set.extend(im_list);

        print len(test_set)
        # return
        # print test_set[:10];
        # return


        # return
        # test_set=randomlySelectTestSet(path_to_val,val_gt_file,no_classes=no_classes,no_im=no_im)
        
        # file_name=runClassification(path_to_val,out_dir,val_gt_file,path_to_classify,gpu_no,layers,no_classes=no_classes,no_im=no_im,ext=ext,central_crop=central_crop);
        file_name=runClassificationTestSet(test_set,out_dir,path_to_classify,gpu_no,layers,ext='JPEG',central_crop=central_crop)
        
        # file_name=os.path.join(out_dir,'20151009153119');
        # file_name=os.path.join(out_dir,'20151009172027');
        # file_name=os.path.join(out_dir,'20151009202006');
        # file_name=os.path.join(out_dir,'20151010000109');
        
        test_set,_=pickle.load(open(file_name+'.p','rb'));
        vals=np.load(file_name+'.npz');

        test_set=sorted(test_set,key=lambda x: x[0])
        test_set=zip(*test_set);
        
        img_paths=list(test_set[0]);
        gt_labels=list(test_set[1]);
        
        numberOfN=5;

        for layer in layers:

            file_name_l=file_name+'_'+layer;
            indices,conf_matrix=doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute')
            pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));
            img_paths=[x.replace('/disk2','../../..') for x in img_paths];
            im_paths,captions=createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels)
            writeHTML(file_name_l+'.html',im_paths,captions)

    

if __name__=='__main__':
    main();