
import os;
import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)));
import re
import subprocess
import numpy as np;
import random;
import time
import shutil
import pickle
import sklearn
import sklearn.neighbors as neighbors
import matplotlib.pyplot as plt;
import math;
import glob;

def doNN(img_paths,gt_labels,features_curr,numberOfN=5,distance='cosine',algo='brute'): 
    gt_labels_uni=list(set(gt_labels));
    gt_labels_uni.sort();

    conf_matrix=np.zeros((len(gt_labels_uni),len(gt_labels_uni)));

    nn=neighbors.NearestNeighbors(n_neighbors=numberOfN+1,metric=distance,algorithm=algo);

    feature_len=1;
    for dim_val in list(features_curr.shape)[1:]:
        feature_len *= dim_val
    
    features_curr=np.reshape(features_curr,(features_curr.shape[0],feature_len));
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
    
def main():
    # path_to_val='/disk2/imagenet/val';
    out_dir='/disk2/octoberExperiments/nn_trained_vs_notrained';
    path_to_im='/disk2/imagenet/not_trained_im/'
    ids_meta=pickle.load(open('select_ids_meta.p','rb'));
    path_to_gt='/disk2/imagenet/structure_files/id_to_text_desc.txt';
    
    with open(path_to_gt,'rb') as f:
        gt_labels=f.readlines();
    gt_labels=[gt_label.split(' ')[0] for gt_label in gt_labels];

    iter=0;
    test_set=[]
    ids=ids_meta[iter];
    print len(ids)
    list_of_files=os.listdir(path_to_im);
    for id in ids:
            
        im_list=glob.glob(os.path.join(path_to_im,id+'_*.JPEG'));
        print len(im_list);
        idx=gt_labels.index(id);
        # print gt_labels[:10];
        im_list=[(path_curr,idx) for path_curr in im_list];
        test_set.extend(im_list);

    print len(test_set)
    print test_set[:10];
    return


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
    print len(text_labels)
    return
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

    

if __name__=='__main__':
    main();