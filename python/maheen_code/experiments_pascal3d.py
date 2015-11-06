import numpy as np;
import cPickle as pickle;
import scipy.io
from pascal3d_db import Pascal3D, Pascal3D_Manipulator
import time;
import os
import caffe_wrapper
import nearest_neighbor
from collections import namedtuple
import visualize

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
    mani=Pascal3D_Manipulator(db_path);
    mani.openSession();
    img_paths=[img_path[0] for img_path in mani.select((Pascal3D.img_path,),(Pascal3D.class_id==class_id,),distinct=True)]
    mani.closeSession();

    file_names,object_indices=recreateOriginalPaths(path_to_annotation,img_paths,returnObject_idx=True);
    img_paths_to_keep=[];

    for idx,(file_name,object_idx) in enumerate(zip(file_names,object_indices)):
        bbox,img_size=getBBSizeInformation(file_name,object_idx);

        overlap=getBoxOverlapRatio(bbox,img_size);
        if overlap>=threshold:
            img_paths_to_keep.append(img_paths[idx]);        

    img_paths_to_keep.sort();
    img_paths_originals=recreateOriginalPaths(path_to_images,img_paths_to_keep,returnObject_idx=False,postfix='jpg');
    pickle.dump([img_paths_originals,img_paths_to_keep],open(out_file_pre+'.p','wb'));
    with open(out_file_pre+'.txt','wb') as f:
        for img_path in img_paths_originals:
            f.write(img_path+'\n');
    
    return out_file_pre+'.p',out_file_pre+'.txt'

def experiment_nnFullImage(params):
    correctRun=True
    # try:
    if params.out_file_pickle is None:
        out_file_pickle,out_file_text=setupFilesWithBigObjectForFeatureExtraction(params.db_path,params.class_id,params.threshold,params.out_file_pre,params.path_to_annotation,params.path_to_images)
        params=params._replace(out_file_pickle=out_file_pickle)
        params=params._replace(out_file_text=out_file_text)
    
    out_file_pickle=params.out_file_pickle;
    
    [img_paths_originals,img_paths_to_keep]=pickle.load(open(out_file_pickle,'rb'));

    file_names_mat,object_indices=recreateOriginalPaths(params.path_to_annotation,img_paths_to_keep,returnObject_idx=True);
    azimuths=[getObjectStruct(file_name,object_idx).viewpoint.azimuth_coarse  for file_name,object_idx in zip(file_names_mat,object_indices)]

    if params.out_file_layers is None:
        out_file_layers=caffe_wrapper.saveFeaturesOfLayers(out_file_text,params.path_to_classify,params.gpu_no,params.layers,ext='jpg',out_file=params.out_file_pre,meanFile=params.caffe_mean,deployFile=params.caffe_deploy,modelFile=params.caffe_model)
        params=params._replace(out_file_layers=out_file_layers)
        
    out_file_layers=params.out_file_layers;

    for layer in params.layers:
        vals=np.load(out_file_layers);
        indices,distances=nearest_neighbor.doCosineDistanceNN(vals[layer],numberOfN=None);
        mani=Pascal3D_Manipulator(params.db_path_out);
        mani.openSession();
        for idx in range(len(img_paths_originals)):
            mani.insert(idx,img_paths_originals[idx],layer,out_file_layers,params.class_id,params.class_idx,params.caffe_model, azimuth=azimuths[idx],neighbor_index=indices[idx],neighbor_distance=distances[idx],trainedClass=params.trainFlag)
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
    elif type_Experiment=='visualizeRankDifferenceByAngle':
        list_params=['db_path',
                    'class_id',
                    'angle',
                    'diff',
                    'delta',
                    'layer',
                    'out_file',
                    'rel_path']
        params=namedtuple('Params_visualizeRankDifferenceByAngle',list_params);
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


def sortBySmallestDistanceRank(indices_interest,neighbor_index):
    
    n_sort_idx= np.argsort(neighbor_index)
    in_sort_idx= np.searchsorted(neighbor_index[n_sort_idx],indices_interest)
    indices_interest_index = n_sort_idx[in_sort_idx]

    indices_interest_index=np.array(indices_interest_index);
    
    idx_to_order=np.argsort(indices_interest_index);
    
    indices_interest_index=indices_interest_index[idx_to_order];

    indices_ordered=indices_interest[idx_to_order];
    
    nn_rank=neighbor_index[indices_interest_index]
    return indices_interest_index,idx_to_order

def getImgInfoSameClass(db_path,class_id,trainFlag,layer):
    mani=Pascal3D_Manipulator(db_path);
    mani.openSession();
    
    criterion=(Pascal3D.class_id==class_id,Pascal3D.trainedClass==trainFlag,Pascal3D.layer==layer)
    toSelect=(Pascal3D.idx,Pascal3D.img_path);
    rows_same_class=mani.select(toSelect,criterion,distinct=True);
    
    mani.closeSession();
    return rows_same_class

def script_visualizePatchesByAngleDifference(params):
    db_path=params.db_path
    class_id=params.class_id
    angle=params.angle
    diff=params.diff
    delta=params.delta
    layer=params.layer
    out_file=params.out_file
    rel_path=params.rel_path

    mani=Pascal3D_Manipulator(db_path);
    mani.openSession();

    trainFlag=True
    criterion=(Pascal3D.class_id==class_id,Pascal3D.azimuth==angle,Pascal3D.layer==layer,Pascal3D.layer==layer,Pascal3D.trainedClass==trainFlag)
    toSelect=(Pascal3D.img_path,Pascal3D.neighbor_index,Pascal3D.azimuth_differences);
    rows=mani.select(toSelect,criterion,distinct=True)
    rows_trained=sorted(rows,key=lambda x: x[0])

    trainFlag=False
    criterion=(Pascal3D.class_id==class_id,Pascal3D.azimuth==angle,Pascal3D.layer==layer,Pascal3D.layer==layer,Pascal3D.trainedClass==trainFlag)
    toSelect=(Pascal3D.img_path,Pascal3D.neighbor_index,Pascal3D.azimuth_differences);
    rows=mani.select(toSelect,criterion,distinct=True)
    rows_no_trained=sorted(rows,key=lambda x: x[0])

    img_paths_trained=[row_trained[0] for row_trained in rows_trained];
    img_paths_no_trained=[row_no_trained[0] for row_no_trained in rows_no_trained];
    assert img_paths_trained==img_paths_no_trained

    img_paths_train,nn_rank_train=getImgPathsAndRanksSorted(rows_trained,db_path,class_id,diff,delta,trainFlag,layer);
    img_paths_no_train,nn_rank_no_train=getImgPathsAndRanksSorted(rows_no_trained,db_path,class_id,diff,delta,trainFlag,layer);

    mani.closeSession()
    
    html_img_paths=[];
    captions=[];
    for idx in range(len(rows)):
        org_img_path=rows[idx][0];
        for nn_rank_row,img_path_row in [(nn_rank_train[idx],img_paths_train[idx]),(nn_rank_no_train[idx],img_paths_no_train[idx])]:
            html_row=[];
            caption_row=[];
            html_row.append(org_img_path.replace(rel_path[0],rel_path[1]));
            caption_row.append('Test Image ')
            for idx_im,im in enumerate(img_path_row):
                html_row.append(im.replace(rel_path[0],rel_path[1]));
                caption_row.append(str(nn_rank_row[idx_im]))
            html_img_paths.append(html_row);
            captions.append(caption_row);
    visualize.writeHTML(out_file,html_img_paths,captions);


def getImgPathsAndRanksSorted(rows,db_path,class_id,diff,delta,trainFlag,layer):
    
    rows_same_class=getImgInfoSameClass(db_path,class_id,trainFlag,layer)
    indices,img_paths=zip(*rows_same_class);
    assert list(indices)==sorted(indices)

    img_paths_all=[];
    nn_ranks_all=[];
    for row_curr in rows:

        img_path_curr,neighbor_index,azimuth_differences=row_curr;
        idx_of_interest=getIndexWithAngleDifference(azimuth_differences,diff,delta)
        indices=np.array(indices);
        nn_rank,idx_to_order=sortBySmallestDistanceRank(indices[idx_of_interest],neighbor_index);
        img_paths_sorted=np.array(img_paths)[idx_of_interest][idx_to_order];
 
        img_paths_all.append(list(img_paths_sorted));
        nn_ranks_all.append(list(nn_rank));

    return img_paths_all,nn_ranks_all

def main():
    db_path='sqlite://///disk2/octoberExperiments/nn_pascal3d/nn_pascal3d_new.db'
    class_id='car';
    angle=0.0;
    diff=90.0;
    delta=90.0;
    layer='pool5';
    out_dir='/disk2/octoberExperiments/nn_pascal3d/seeingPatches';
    out_file=class_id+'_'+str(angle)+'_'+str(diff)+'_'+str(delta)+'_'+layer+'.html'
    out_file=os.path.join(out_dir,out_file)
    rel_path=['/disk2','../../../..'];

    Params=createParams('visualizeRankDifferenceByAngle')
    params=Params(db_path=db_path,class_id=class_id,angle=angle,diff=diff,delta=delta,layer=layer,out_file=out_file,rel_path=rel_path)
    script_visualizePatchesByAngleDifference(params)
    pickle.dump(params._asdict(),open(out_file+'_meta_experiment.p','wb'))
    
    

    

if __name__=='__main__':
    main();