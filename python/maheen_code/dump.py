FOR nnFullImageMixWithImagenet
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
    out_file_pickle=None;
    out_file_layers=None;
    
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


FOR visualizeRankDiffrenceHist

    params_file='/disk2/octoberExperiments/nn_pascal3d/full_image_nn/all_pascal_new_experiment_meta.p'
    params_dict,_=pickle.load(open(params_file,'rb'));
    # out_dir='/disk2/octoberExperiments/nn_pascal3d/full_image_nn'
    # db_path_out=params_dict['db_path_out'];
    
    out_dir='/disk2/octoberExperiments/nn_pascal3d/patches_nn'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    db_path_out='sqlite://///disk2/octoberExperiments/nn_pascal3d/nn_pascal3d_new.db'
    
    params_dict_new={};
    params_dict_new['db_path']=db_path_out
    params_dict_new['class_id']=params_dict['class_id']
    params_dict_new['angle']=None
    params_dict_new['diff']=90.0
    params_dict_new['delta']=5.0
    params_dict_new['layer']=params_dict['layers']
    params_dict_new['out_file_pre']=os.path.join(out_dir,'rankDifferenceHist');
    params_dict_new['bins']=20;
    params_dict_new['normed']=True
    params_dict_new['out_file_html']=os.path.join(out_dir,'rankDifferenceHist_'+str(params_dict_new['angle'])+'_'+str(params_dict_new['diff'])+'_'+str(params_dict_new['delta'])+'_all.html');
    params_dict_new['rel_path']=['/disk2','../../../..']
    params_dict_new['height_width']=[300,300];
    
    params=createParams('visualizeRankDifferenceByAngleHist');
    params=params(**params_dict_new)
    
    script_visualizeRankDifferenceAsHist(params)



FOR visualizePatches
    db_path='sqlite://///disk2/octoberExperiments/nn_pascal3d/nn_pascal3d_new.db'
    layers=['pool5','fc6','fc7']
    for layer in layers:
        class_id='car';
        angle=0.0;
        diff=90.0;
        delta=5.0;
        # layer='pool5';
        out_dir='/disk2/octoberExperiments/nn_pascal3d/seeingPatches';
        out_file=class_id+'_'+str(angle)+'_'+str(diff)+'_'+str(delta)+'_'+layer+'.html'
        out_file=os.path.join(out_dir,out_file)
        rel_path=['/disk2','../../../..'];

        Params=createParams('visualizeRankDifferenceByAngle')
        params=Params(db_path=db_path,class_id=class_id,angle=angle,diff=diff,delta=delta,layer=layer,out_file=out_file,rel_path=rel_path)
        script_visualizePatchesByAngleDifference(params)
        pickle.dump(params._asdict(),open(out_file+'_meta_experiment.p','wb'))


FOR nnFullImage
    out_dir='/disk2/octoberExperiments/nn_pascal3d/full_image_nn'
    Params=createParams('nnFullImage')
    path_to_images='/disk2/pascal_3d/PASCAL3D+_release1.0/Images/';
    path_to_annotation='/disk2/pascal_3d/PASCAL3D+_release1.0/Annotations/';
    db_path='sqlite://///disk2/octoberExperiments/nn_pascal3d/nn_pascal3d_new.db'
    db_path_out='sqlite://///disk2/octoberExperiments/nn_pascal3d/full_image_nn.db'
    class_id='car';
    class_idx=11;
    threshold=float(2)/3
    path_to_classify='..'
    gpu_no=0
    layers=['pool5','fc6','fc7'];
    trainFlag=False
    caffe_model='/home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    caffe_deploy='/home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/deploy.prototxt';
    caffe_mean='/home/maheenrashid/Downloads/caffe/caffe-rc2/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    
    # caffe_deploy='/disk2/octoberExperiments/nn_performance_without_pascal/deploy.prototxt'
    # caffe_mean='/disk2/octoberExperiments/nn_performance_without_pascal/mean.npy'
    # caffe_model='/disk2/octoberExperiments/nn_performance_without_pascal/snapshot_iter_450000.caffemodel'
    out_file_pre=os.path.join(out_dir,class_id)
    # +'_no_train');
    
    params=Params(path_to_images=path_to_images,
                    path_to_annotation=path_to_annotation,
                    db_path=db_path,
                    db_path_out=db_path_out,
                    class_id=class_id,
                    class_idx=class_idx,
                    threshold=threshold,
                    path_to_classify=path_to_classify,
                    gpu_no=gpu_no,
                    layers=layers,
                    trainFlag=trainFlag,
                    caffe_model=caffe_model,
                    caffe_deploy=caffe_deploy,
                    caffe_mean=caffe_mean,
                    out_file_pre=out_file_pre,
                    out_file_layers=None,
                    out_file_pickle=None,
                    out_file_text=None)
    # params,correctRun=experiment_nnFullImage(params);
    pickle.dump([params._asdict(),correctRun],open(params.out_file_pre+'_experiment_meta.p','wb'));

