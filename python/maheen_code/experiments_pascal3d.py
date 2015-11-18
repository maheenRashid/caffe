import numpy as np;
import cPickle as pickle;
import scipy.io
from pascal3d_db import Pascal3D, Pascal3D_Manipulator
import time;
import os
import caffe_wrapper
import nearest_neighbor
from collections import namedtuple
import collections
import visualize
import util;
import experiments_super;
import imagenet;

def getObjectStruct(file_name,object_idx):
    curr_dict=scipy.io.loadmat(file_name,squeeze_me=True, struct_as_record=False);
    
    objects=curr_dict['record'].objects
    
    if not hasattr(objects, '__iter__'):
        objects=[objects]
    
    return objects[object_idx]

def getBBSizeInformation(file_name,object_idx):
    curr_dict=scipy.io.loadmat(file_name,squeeze_me=True, struct_as_record=False);
    
    objects=curr_dict['record'].objects
    img_size=curr_dict['record'].imgsize
    
    if not hasattr(objects, '__iter__'):
        objects=[objects]
    
    object_bb=objects[object_idx].bbox
    return object_bb,img_size
    
def recreateOriginalPaths(path_to_annotation,img_paths,returnObject_idx=False,postfix='mat'):    

    file_pre=[img_path[img_path.rindex('/')+1:].rsplit('_',2)[0] for img_path in img_paths]
    object_idx=[int(img_path[:-4].rsplit('_',1)[-1]) for img_path in img_paths];
    class_id=[img_path.rsplit('_',2)[1] for img_path in img_paths];
    file_names=[os.path.join(os.path.join(path_to_annotation,class_id[idx]+'_pascal'),file_pre[idx]+'.'+postfix) for idx in range(len(file_pre))];
    if returnObject_idx:
        return file_names,object_idx
    else:
        return file_names

def getBoxOverlapRatio(bbox,img_size):
    bbox=[int(val-1) for val in bbox];
    img_size=[int(val) for val in img_size]
    bbox_h=bbox[2]-bbox[0];

    bbox_w=bbox[3]-bbox[1];
    area=bbox_h*bbox_w;
    img_area=img_size[0]*img_size[1]
    return area/float(img_area);

def setupFilesWithBigObjectForFeatureExtraction(db_path,class_id,threshold,out_file_pre,path_to_annotation,path_to_images):

    if not hasattr(class_id, '__iter__'):
        class_id=[class_id];

    mani=Pascal3D_Manipulator(db_path);
    mani.openSession();
    img_paths=[];
    class_ids_record=[];
    for class_idx_curr,class_id_curr in enumerate(class_id):
        im_list_curr=[img_path[0] for img_path in mani.select((Pascal3D.img_path,),(Pascal3D.class_id==class_id_curr,),distinct=True)];
        class_ids_record.extend([class_id_curr]*len(im_list_curr));
        img_paths.extend(im_list_curr)
    mani.closeSession();

    file_names,object_indices=recreateOriginalPaths(path_to_annotation,img_paths,returnObject_idx=True);
    img_paths_to_keep=[];
    class_ids=[];
    for idx,(file_name,object_idx) in enumerate(zip(file_names,object_indices)):
        bbox,img_size=getBBSizeInformation(file_name,object_idx);

        overlap=getBoxOverlapRatio(bbox,img_size);
        if overlap>=threshold:
            img_paths_to_keep.append(img_paths[idx]);        
            class_ids.append(class_ids_record[idx]);

    img_paths_originals=recreateOriginalPaths(path_to_images,img_paths_to_keep,returnObject_idx=False,postfix='jpg');

    if len(img_paths_originals)!=len(set(img_paths_originals)):
        duplicates=[item for item, count in collections.Counter(img_paths_originals).items() if count > 1];
        img_paths_originals=np.array(img_paths_originals);
        idx_to_del=[];
        for duplicate in duplicates:
            idx_to_del.extend(list(np.where(img_paths_originals==duplicate)[0]));
        img_paths_originals=np.delete(img_paths_originals,idx_to_del);
        img_paths_to_keep=np.delete(img_paths_to_keep,idx_to_del);
        class_ids=np.delete(class_ids,idx_to_del);
    
    check=recreateOriginalPaths(path_to_images,list(img_paths_to_keep),returnObject_idx=False,postfix='jpg');
    assert check==list(img_paths_originals)

    sort_idx=np.argsort(img_paths_originals);
    img_paths_originals=list(img_paths_originals[sort_idx])
    img_paths_to_keep=list(img_paths_to_keep[sort_idx])
    class_ids=list(class_ids[sort_idx])
    class_ids=[(class_id,class_idx) for class_idx,class_id in enumerate(class_ids)]

    assert len(img_paths_originals)==len(set(img_paths_originals));
    assert len(img_paths_originals)==len(class_ids)==len(img_paths_to_keep);

    pickle.dump([img_paths_originals,img_paths_to_keep,class_ids],open(out_file_pre+'.p','wb'));
    with open(out_file_pre+'.txt','wb') as f:
        for img_path in img_paths_originals:
            f.write(img_path+'\n');
    
    return out_file_pre+'.p',out_file_pre+'.txt'

def experiment_nnFullImageMixWithImagenet(params):
    
    if params.out_file_text is None:
        all_files_info=preprocessImagesFromImagenet(params.imagenet_ids_to_test,params.synset_words,params.val_gt_file,path_to_images_imagenet=params.path_to_images_imagenet)
        all_files_info=getImageInfoForMixTestFromPascal3d(params.db_path_in,params.class_id_pascal,all_files_info=all_files_info);
        out_file_pickle=params.out_file_pre+'.p';
        out_file_text=params.out_file_pre+'.txt';
        pickle.dump(all_files_info,open(out_file_pickle,'wb'));
        with open(out_file_text,'wb') as f:
            for dict_curr in all_files_info:
                f.write(dict_curr['img_path']+'\n');
        params=params._replace(out_file_pickle=out_file_pickle)
        params=params._replace(out_file_text=out_file_text)

    out_file_text=params.out_file_text;
    out_file_pickle=params.out_file_pickle;
    
    if params.out_file_layers is None:
        print 'running layers part'
        out_file_layers=caffe_wrapper.saveFeaturesOfLayers(out_file_text,params.path_to_classify,params.gpu_no,params.layers,ext=params.ext,out_file=params.out_file_pre,meanFile=params.caffe_mean,deployFile=params.caffe_deploy,modelFile=params.caffe_model)
        params=params._replace(out_file_layers=out_file_layers)
        
    out_file_layers=params.out_file_layers;
    all_files_info=pickle.load(open(out_file_pickle,'rb'));

    print 'writing to db'
    for layer in params.layers:
        vals=np.load(out_file_layers);
        indices,distances=nearest_neighbor.doCosineDistanceNN(vals[layer],numberOfN=None);
        mani=Pascal3D_Manipulator(params.db_path_out);
        mani.openSession();
        # for idx in range(len(img_paths_originals)):
        for idx,dict_curr in enumerate(all_files_info):
            mani.insert(idx,dict_curr['img_path'],layer,out_file_layers,dict_curr['class_id'],dict_curr['class_idx'],params.caffe_model, azimuth=dict_curr['azimuth'],neighbor_index=indices[idx],neighbor_distance=distances[idx],trainedClass=params.trainFlag)
        mani.closeSession();
    
    return params;



def experiment_nnFullImage(params):
    correctRun=True
    # try:
    if params.out_file_pickle is None:
        out_file_pickle,out_file_text=setupFilesWithBigObjectForFeatureExtraction(params.db_path,params.class_id,params.threshold,params.out_file_pre,params.path_to_annotation,params.path_to_images)
        params=params._replace(out_file_pickle=out_file_pickle)
        params=params._replace(out_file_text=out_file_text)
    
    out_file_pickle=params.out_file_pickle;
    out_file_text=params.out_file_text;
    print out_file_pickle
    [img_paths_originals,img_paths_to_keep,class_id_idx_tuples]=pickle.load(open(out_file_pickle,'rb'));
    assert len(img_paths_originals)==len(class_id_idx_tuples)

    file_names_mat,object_indices=recreateOriginalPaths(params.path_to_annotation,img_paths_to_keep,returnObject_idx=True);
    print 'getting azimuths'
    azimuths=[getObjectStruct(file_name,object_idx).viewpoint.azimuth_coarse  for file_name,object_idx in zip(file_names_mat,object_indices)]
    
    if params.out_file_layers is None:
        print 'running layers part'
        out_file_layers=caffe_wrapper.saveFeaturesOfLayers(out_file_text,params.path_to_classify,params.gpu_no,params.layers,ext='jpg',out_file=params.out_file_pre,meanFile=params.caffe_mean,deployFile=params.caffe_deploy,modelFile=params.caffe_model)
        params=params._replace(out_file_layers=out_file_layers)
        
    out_file_layers=params.out_file_layers;

    print 'writing to db'
    for layer in params.layers:
        vals=np.load(out_file_layers);
        indices,distances=nearest_neighbor.doCosineDistanceNN(vals[layer],numberOfN=None);
        mani=Pascal3D_Manipulator(params.db_path_out);
        mani.openSession();
        for idx in range(len(img_paths_originals)):
            mani.insert(idx,img_paths_originals[idx],layer,out_file_layers,class_id_idx_tuples[idx][0],class_id_idx_tuples[idx][1],params.caffe_model, azimuth=azimuths[idx],neighbor_index=indices[idx],neighbor_distance=distances[idx],trainedClass=params.trainFlag)
        mani.closeSession();
    # except:
    #     correctRun=False;

    return params,correctRun;

def createParams(type_Experiment):
    if type_Experiment=='nnFullImage':
        list_params=['path_to_images',
                    'path_to_annotation',
                    'db_path',
                    'db_path_out',
                    'class_id',
                    'class_idx',
                    'threshold',
                    'path_to_classify',
                    'gpu_no',
                    'layers',
                    'trainFlag',
                    'caffe_model',
                    'caffe_deploy',
                    'caffe_mean',
                    'out_file_pre',
                    'out_file_layers',
                    'out_file_pickle',
                    'out_file_text']
        params=namedtuple('Params_nnFullImage',list_params);
    elif type_Experiment=='visualizeRankDifferenceByAngleImages':
        list_params=['db_path',
                    'class_id',
                    'angle',
                    'diff',
                    'delta',
                    'layer',
                    'out_file',
                    'rel_path']
        params=namedtuple('Params_visualizeRankDifferenceByAngle',list_params);
    elif type_Experiment=='visualizeRankDifferenceByAngleHist':
        list_params=['db_path',
                    'class_id',
                    'angle',
                    'diff',
                    'delta',
                    'layer',
                    'out_file_pre',
                    'bins',
                    'normed',
                    'out_file_html',
                    'rel_path',
                    'height_width']
        params=namedtuple('Params_visualizeRankDifferenceByAngleHist',list_params);
    elif type_Experiment=='nnFullImageMixWithImagenet':
        list_params=['path_to_images_imagenet',
                    'path_to_classify',
                    'imagenet_ids_to_test',
                    'val_gt_file',
                    'synset_words',
                    'caffe_model',
                    'caffe_deploy',
                    'caffe_mean',
                    'class_id_pascal',
                    'db_path_in',
                    'db_path_out',
                    'out_file_pre',
                    'layers',
                    'ext',
                    'gpu_no',
                    'trainFlag',
                    'out_file_text',
                    'out_file_pickle',
                    'out_file_layers']
        params=namedtuple('Params_nnFullImageMixWithImagenet',list_params);
    return params;


def updateAzimuthDifferencesInDB(db_path,class_id,printDebug=True):
    mani=Pascal3D_Manipulator(db_path);
    mani.openSession()
    vals=mani.select((Pascal3D.img_path,Pascal3D.idx,Pascal3D.azimuth),(Pascal3D.class_id==class_id,),distinct=True);
    mani.closeSession();

    img_paths,indices,azimuths=zip(*vals);
    img_paths=[str(img_path) for img_path in img_paths];
    
    print len(img_paths)

    assert len(img_paths),len(set(img_paths))
    assert img_paths==sorted(img_paths)
    assert sorted(indices)==list(indices)

    azimuth_differences=getAzimuthDifferences(azimuths)
    mani.openSession();
    for idx,img_path in enumerate(img_paths):
        if printDebug:
            if idx%100==0:
                print idx,len(img_paths)
        mani.update((Pascal3D.img_path==img_path,),{Pascal3D.azimuth_differences:azimuth_differences[idx]});
    mani.closeSession();

def getAzimuthDifferences(azimuths):
    azimuths=np.array(azimuths).ravel();
    azimuth_subtractor=np.repeat(np.expand_dims(azimuths,axis=1),len(azimuths),axis=1);
    azimuth_differences=azimuth_subtractor.T

    assert list(azimuth_subtractor[:,0])==list(azimuths)
    assert azimuth_subtractor.shape==(len(azimuths),len(azimuths))
    assert list(azimuth_differences[0,:])==list(azimuths)
    assert azimuth_differences.shape==(len(azimuths),len(azimuths))
    
    azimuth_differences=np.absolute(azimuth_subtractor-azimuth_differences);
    azimuth_differences[azimuth_differences>180]=360-azimuth_differences[azimuth_differences>180]
    return azimuth_differences

def getIndexWithAngleDifference(azimuth_differences,diff,delta):
    idx_of_interest=np.where(np.logical_and(azimuth_differences<=diff+delta,azimuth_differences>=diff-delta))
    return idx_of_interest;

def getImgInfoSameClass(db_path,class_id,trainFlag,layer):
    mani=Pascal3D_Manipulator(db_path);
    mani.openSession();
    
    criterion=(Pascal3D.class_id==class_id,Pascal3D.trainedClass==trainFlag,Pascal3D.layer==layer)
    toSelect=(Pascal3D.idx,Pascal3D.img_path);
    rows_same_class=mani.select(toSelect,criterion,distinct=True);
    
    mani.closeSession();
    return rows_same_class

def getNNRankComparisonInfo(params):
    db_path=params.db_path
    class_id=params.class_id
    angle=params.angle
    diff=params.diff
    delta=params.delta
    layer=params.layer
    
    mani=Pascal3D_Manipulator(db_path);
    mani.openSession();
    trainFlag=True

    if angle is not None:
        criterion=(Pascal3D.class_id==class_id,Pascal3D.azimuth==angle,Pascal3D.layer==layer,Pascal3D.layer==layer,Pascal3D.trainedClass==trainFlag)
    else:
        criterion=(Pascal3D.class_id==class_id,Pascal3D.layer==layer,Pascal3D.layer==layer,Pascal3D.trainedClass==trainFlag)
    toSelect=(Pascal3D.img_path,Pascal3D.neighbor_index,Pascal3D.azimuth_differences);
    rows=mani.select(toSelect,criterion,distinct=True)
    rows_trained=sorted(rows,key=lambda x: x[0])
    
    trainFlag=False

    if angle is not None:
        criterion=(Pascal3D.class_id==class_id,Pascal3D.azimuth==angle,Pascal3D.layer==layer,Pascal3D.trainedClass==trainFlag)
    else:
        criterion=(Pascal3D.class_id==class_id,Pascal3D.layer==layer,Pascal3D.trainedClass==trainFlag)

    toSelect=(Pascal3D.img_path,Pascal3D.neighbor_index,Pascal3D.azimuth_differences);
    rows=mani.select(toSelect,criterion,distinct=True)
    rows_no_trained=sorted(rows,key=lambda x: x[0])
    
    img_paths_trained=[row_trained[0] for row_trained in rows_trained];
    img_paths_no_trained=[row_no_trained[0] for row_no_trained in rows_no_trained];
    assert img_paths_trained==img_paths_no_trained
    trainFlag=True
    img_paths_train,nn_rank_train=getImgPathsAndRanksSorted(rows_trained,db_path,class_id,diff,delta,trainFlag,layer);
    trainFlag=False
    img_paths_no_train,nn_rank_no_train=getImgPathsAndRanksSorted(rows_no_trained,db_path,class_id,diff,delta,trainFlag,layer);
    mani.closeSession()

    output={};
    output['img_paths_test']=img_paths_trained
    output['img_paths_nn_train']=img_paths_train
    output['img_paths_nn_no_train']=img_paths_no_train
    output['nn_rank_train']=nn_rank_train
    output['nn_rank_no_train']=nn_rank_no_train
    return output
    

def script_visualizePatchesByAngleDifference(params):
    out_file=params.out_file
    rel_path=params.rel_path
    output=getNNRankComparisonInfo(params)
    
    img_paths_test = output['img_paths_test']
    img_paths_train = output['img_paths_nn_train']
    img_paths_no_train = output['img_paths_nn_no_train']
    nn_rank_train = output['nn_rank_train']
    nn_rank_no_train = output['nn_rank_no_train']
    
    html_img_paths=[];
    captions=[];
    for idx,org_img_path in enumerate(img_paths_test):
        for nn_rank_row,img_path_row in [(nn_rank_train[idx],img_paths_train[idx]),(nn_rank_no_train[idx],img_paths_no_train[idx])]:
            html_row=[];
            caption_row=[];
            html_row.append(org_img_path.replace(rel_path[0],rel_path[1]));
            caption_row.append('Test Image ')
            for idx_im,im in enumerate(img_path_row):
                html_row.append(im.replace(rel_path[0],rel_path[1]));
                caption_row.append(str(nn_rank_row[idx_im])+' '+str(idx))
            html_img_paths.append(html_row);
            captions.append(caption_row);
    visualize.writeHTML(out_file,html_img_paths,captions);


def script_visualizeRankDifferenceAsHist(params):
    out_file_pre = params.out_file_pre;
    out_file_html = params.out_file_html;
    rel_path = params.rel_path;
    class_ids = params.class_id;
    layers = params.layer
    
    if not hasattr(class_ids, '__iter__'):
        class_ids = [class_ids];
    
    if not hasattr(class_ids, '__iter__'):
        layers = [layers];

    img_paths_html=[];
    captions=[];
    for class_id in class_ids:
        img_paths_html_row=[];
        captions_row=[];
        for layer in layers:
            params = params._replace(class_id=class_id)
            params = params._replace(layer=layer)
            output=getNNRankComparisonInfo(params);
            indices_difference=experiments_super.getDifferenceInRank(output['img_paths_nn_train'],
                                                output['img_paths_nn_no_train'],
                                                output['nn_rank_train'],
                                                output['nn_rank_no_train']);
            xlabel='Difference in Rank (Train - Untrained)';
            ylabel='Frequency'
            title=class_id+' '+layer
            indices_difference=[diff_curr for diffs_curr in indices_difference for diff_curr in diffs_curr]
            
            if len(indices_difference)==0:
                continue;
            
            out_file_im=out_file_pre+'_'+str(params.angle)+'_'+str(params.diff)+'_'+str(params.delta)+'_'+class_id+'_'+layer+'.png';
            img_paths_html_row.append(out_file_im.replace(rel_path[0],rel_path[1]));
            total=len(indices_difference)
            sum_less=sum(np.array(indices_difference)<0)/float(total);
            sum_less='%0.2f'%(sum_less,)
            sum_more=sum(np.array(indices_difference)>=0)/float(total);
            sum_more='%0.2f'%(sum_more,)
            captions_row.append('Total '+str(total)+', <0: '+sum_less+', >0: '+sum_more);
            
            visualize.hist(indices_difference,out_file=out_file_im,bins=params.bins,normed=params.normed,xlabel=xlabel,ylabel=ylabel,title=title);
            
        img_paths_html.append(img_paths_html_row);
        captions.append(captions_row);

    visualize.writeHTML(out_file_html,img_paths_html,captions,params.height_width[0],params.height_width[1]);


def getImgPathsAndRanksSorted(rows,db_path,class_id,diff,delta,trainFlag,layer):
    
    rows_same_class=getImgInfoSameClass(db_path,class_id,trainFlag,layer)
    indices,img_paths=zip(*rows_same_class);
    assert list(indices)==sorted(indices)

    img_paths_all=[];
    nn_ranks_all=[];
    for row_curr in rows:

        img_path_curr,neighbor_index,azimuth_differences=row_curr;
        idx_to_del=img_paths.index(img_path_curr);
        azimuth_differences[idx_to_del]=-1*float('Inf');
        idx_of_interest=getIndexWithAngleDifference(azimuth_differences,diff,delta)
        indices=np.array(indices);
        nn_rank,idx_to_order=experiments_super.sortBySmallestDistanceRank(indices[idx_of_interest],neighbor_index);
        img_paths_sorted=np.array(img_paths)[idx_of_interest][idx_to_order];
 
        img_paths_all.append(list(img_paths_sorted));
        nn_ranks_all.append(list(nn_rank));

    return img_paths_all,nn_ranks_all

def getImgPathAndAzimuth(db_path_in,class_id):
    mani=Pascal3D_Manipulator(db_path_in);
    mani.openSession();
    vals=mani.select((Pascal3D.img_path,Pascal3D.azimuth),(Pascal3D.class_id==class_id,),distinct=True);
    mani.closeSession();
    return vals;

def preprocessImagesFromImagenet(imagenet_ids_to_test,synset_words,val_gt_file,path_to_images_imagenet='',all_files_info=None,start_idx=1):
    # class_ids=[];
    ids,labels=zip(*imagenet.readLabelsFile(synset_words));
    list_of_idx=[ids.index(id_curr) for id_curr in imagenet_ids_to_test];
    test_set=imagenet.selectTestSetByID(val_gt_file,list_of_idx,path_to_val=path_to_images_imagenet)
    
    if all_files_info is None:
        all_files_info=[];

    for img_path,imagenet_idx in test_set:
        dict_curr={}
        dict_curr['img_path']=img_path;
        id_curr=ids[imagenet_idx];
        dict_curr['class_idx']=imagenet_ids_to_test.index(id_curr)+start_idx;
        dict_curr['class_id']=id_curr;
        dict_curr['azimuth']=None;
        all_files_info.append(dict_curr);

    return all_files_info

def getImageInfoForMixTestFromPascal3d(db_path_in,class_id_pascal,all_files_info=None,class_idx=0):

    test_set_pascal=getImgPathAndAzimuth(db_path_in,class_id_pascal);
    img_paths_pascal=list(zip(*test_set_pascal)[0]);
    azimuths=list(zip(*test_set_pascal)[1]);
    
    if all_files_info is None:
        all_files_info=[];

    for idx,img_path in enumerate(img_paths_pascal):
        dict_curr={}
        dict_curr['img_path']=img_path;
        dict_curr['class_idx']=class_idx;
        dict_curr['class_id']=class_id_pascal;
        dict_curr['azimuth']=azimuths[idx];
        all_files_info.append(dict_curr);

    return all_files_info;
    


def main():
    db_path='sqlite://///disk2/novemberExperiments/nn_pascal3d_imagenet_mix/cars.db';
    class_id='car';
    angle=0.0;
    diff=90.0;
    delta=5.0;
    layer=['pool5','fc6','fc7'];
    out_file_pre='/disk2/novemberExperiments/nn_pascal3d_imagenet_mix/rank_difference_';
    bins=20;
    normed=True;
    out_file_html='/disk2/novemberExperiments/nn_pascal3d_imagenet_mix/rank_difference_'+class_id+'_'+str(angle)+'_'+str(diff)+'_'+str(delta)+'.html';
    rel_path=['/disk2','../../..'];
    height_width=[400,400];

    params=createParams('visualizeRankDifferenceByAngleHist');
    params=params(db_path=db_path,class_id=class_id,angle=angle,diff=diff,delta=delta,layer=layer,out_file_pre=out_file_pre,bins=bins,normed=normed,out_file_html=out_file_html,rel_path=rel_path,height_width=height_width)
    
    script_visualizeRankDifferenceAsHist(params)

    # db_path_out='sqlite://///disk2/novemberExperiments/nn_pascal3d_imagenet_mix/cars.db';
    # class_id='car';
    # updateAzimuthDifferencesInDB(db_path_out,class_id);

    return
    path_to_images_imagenet='/disk2/imagenet/val';
    path_to_classify='..';
    imagenet_ids_to_test=['n03930630','n03977966','n02974003'];
    val_gt_file='/disk2/octoberExperiments/nn_performance_without_pascal/val.txt'
    synset_words='/disk2/octoberExperiments/nn_performance_without_pascal/synset_words.txt'
    caffe_model='/disk2/octoberExperiments/nn_performance_without_pascal/snapshot_iter_450000.caffemodel'
    caffe_deploy='/disk2/octoberExperiments/nn_performance_without_pascal/deploy.prototxt'
    caffe_mean='/disk2/octoberExperiments/nn_performance_without_pascal/mean.npy'
    class_id_pascal='car';
    db_path_in='sqlite://///disk2/octoberExperiments/nn_pascal3d/full_image_nn_new.db';
    db_path_out='sqlite://///disk2/novemberExperiments/nn_pascal3d_imagenet_mix/cars.db';
    out_file_pre='/disk2/novemberExperiments/nn_pascal3d_imagenet_mix/cars_no_train';
    layers=['pool5','fc6','fc7']
    ext=None
    gpu_no=0;
    trainFlag=False;
    out_file_text=None;
    # out_file_pre+'.txt';
    out_file_pickle=None;
    # out_file_pre+'.p';
    out_file_layers=None;
    # out_file_pre+'.npz';
    
    params=createParams('nnFullImageMixWithImagenet');
    params=params(path_to_images_imagenet=path_to_images_imagenet,path_to_classify=path_to_classify,imagenet_ids_to_test=imagenet_ids_to_test,val_gt_file=val_gt_file,synset_words=synset_words,caffe_model=caffe_model,caffe_deploy=caffe_deploy,caffe_mean=caffe_mean,class_id_pascal=class_id_pascal,db_path_in=db_path_in,db_path_out=db_path_out,out_file_pre=out_file_pre,layers=layers,ext=ext,gpu_no=gpu_no,trainFlag=trainFlag,out_file_text=out_file_text,out_file_pickle=out_file_pickle,out_file_layers=out_file_layers);

    params=experiment_nnFullImageMixWithImagenet(params);
    pickle.dump(params._asdict(),open(params.out_file_pre+'_meta_experiment.p','wb'));


    return

    path_to_images_imagenet='/disk2/imagenet/val';
    path_to_classify='..';
    imagenet_ids_to_test=['n03930630','n03977966','n02974003'];
    val_gt_file='/home/maheenrashid/Downloads/caffe/caffe-rc2/data/ilsvrc12/val.txt'
    synset_words='/home/maheenrashid/Downloads/caffe/caffe-rc2/data/ilsvrc12/synset_words.txt'
    caffe_model='/home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    caffe_deploy='/home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/deploy.prototxt'
    caffe_mean='/home/maheenrashid/Downloads/caffe/caffe-rc2/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    class_id_pascal='car';
    db_path_in='sqlite://///disk2/octoberExperiments/nn_pascal3d/full_image_nn_new.db';
    db_path_out='sqlite://///disk2/novemberExperiments/nn_pascal3d_imagenet_mix/cars.db';
    out_file_pre='/disk2/novemberExperiments/nn_pascal3d_imagenet_mix/cars_train';
    layers=['pool5','fc6','fc7']
    ext=None
    gpu_no=0;
    trainFlag=True;
    out_file_text=None;
    # out_file_pre+'.txt';
    out_file_pickle=None;
    # out_file_pre+'.p';
    out_file_layers=None;
    # out_file_pre+'.npz';
    return
    # ['path_to_images',
    # 'path_to_annotation',
    # 'db_path_out',
    # 'class_id',
    # 'class_idx',
    # 'threshold',
    # 'path_to_classify',
    # 'gpu_no',
    # 'layers',
    # 'trainFlag',
    # 'caffe_model',
    # 'caffe_deploy',
    # 'caffe_mean',
    # 'out_file_pre',
    # 'out_file_layers',
    # 'out_file_text']
    # path_to_images /disk2/pascal_3d/PASCAL3D+_release1.0/Images/
    # path_to_annotation /disk2/pascal_3d/PASCAL3D+_release1.0/Annotations/
    # db_path sqlite://///disk2/octoberExperiments/nn_pascal3d/nn_pascal3d_new.db
    # db_path_out sqlite://///disk2/octoberExperiments/nn_pascal3d/full_image_nn_new.db
    # class_id ['boat', 'train', 'bicycle', 'chair', 'motorbike', 'aeroplane', 'sofa', 'diningtable', 'bottle', 'tvmonitor', 'bus', 'car']
    # class_idx [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # threshold 0.666666666667
    # path_to_classify ..
    # gpu_no 0
    # layers ['pool5', 'fc6', 'fc7']
    # trainFlag True
    # caffe_model /home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
    # caffe_deploy /home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/deploy.prototxt
    # caffe_mean /home/maheenrashid/Downloads/caffe/caffe-rc2/python/caffe/imagenet/ilsvrc_2012_mean.npy
    # out_file_pre /disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new
    # out_file_layers /disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new.npz
    # out_file_pickle /disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new.p
    # out_file_text /disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new.txt

    # meta_file='/disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new_experiment_meta.p'
    # params,correctRun=pickle.load(open(meta_file,'rb'));
    # for key_curr in params:
    #     print key_curr,params[key_curr];

    return      
    db_path_3d='';
    db_path_out='';
    imagnet_ids_to_splay=['n03930630','n03977966','n02974003','n03459775','n04461696','n03345487','n03444034','n04065272','n03445924','n04465501','n03478589','n02860847','n03417042','n02704792','n04252225'];

    params=createParams('nnFullImageWithImagenet');


    experiment_nnFullImageMixWithImagenet(params)

    params_file='/disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new_experiment_meta.p'
    params_dict,_=pickle.load(open(params_file,'rb'));
    print params_dict['class_id']

if __name__=='__main__':
    main();