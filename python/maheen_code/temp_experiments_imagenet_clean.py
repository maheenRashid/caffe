

# def experiment_nnImagenet(params):
#     in_file_pre
#     out_file_pre
#     out_files
#     in_files
#     caffe_model
#     caffe_mean
#     caffe_deploy
#     per_file
#     path_to_val

#     if in_files is None:
#         list_files=[os.path.join(path_to_val,file_curr) for file_curr in os.listdir(path_to_val) if file_curr.endswith(ext)];

#         in_files,_=writeInputImageFiles(list_files,in_file_pre,per_file);
#         params.in_files=in_files;

#     if out_files is None:
#         out_files=[];
#         for idx in range(len(in_files)):
#             in_file_curr=in_files[idx];
#             out_file_curr=out_file_pre+'_'+str(idx);
#             out_file_curr=caffe_wrapper.saveFeaturesOfLayers(in_file_curr,path_to_classify,gpu_no,layers,ext=ext,out_file=out_file_curr,meanFile=caffe_mean,deployFile=caffe_deploy,modelFile=caffe_model);
#             out_files.append(out_file_curr);

#         params.out_files=out_files;

#     print 'about to combine'
#     t=time.time();
#     val_combined=combineDeepFeaturesFromFiles(out_files,layers);
#     print time.time()-t

#     for layer_curr in layers:
#         print 'about to nn for ',layer_curr
#         t=time.time();
#         indices,distances=nearest_neighbor.doCosineDistanceNN(val_combined[layer_curr],numberOfN=None)
#         print time.time()-t;
#         print indices.shape
#         print distances.shape

#         print 'writing to db';
#         mani=Imagenet_Manipulator(db_path_out);
#         mani.openSession();
#         for idx in range(len(file_list_all)):
#             if idx%100==0:
#                 print idx,len(file_list_all)
#             mani.insert( idx,file_list_all[idx], layer_curr,out_file_layers, trainFlag, imagenet_idx_mapped[idx], imagenet_ids_mapped[idx],caffe_model, class_label_imagenet=imagenet_labels_mapped[idx],  neighbor_index=indices[idx],neighbor_distance=distances[idx])
        
#         mani.closeSession();



def main():
    params=createParams('visualizeRankDifferenceByAngleHist');

    db_path='sqlite://///disk2/novemberExperiments/nn_imagenet/nn_imagenet.db';
    class_id=['boat', 'train', 'bicycle', 'chair', 'motorbike', 'aeroplane', 'sofa', 'diningtable', 'bottle', 'tvmonitor', 'bus', 'car'];
    layer=['pool5','fc6','fc7'];
    out_file_pre='/disk2/novemberExperiments/nn_imagenet/rank_difference_comparison';
    bins=20;
    normed=True;
    out_file_html='/disk2/novemberExperiments/nn_imagenet/rank_difference_comparison_pascal3dClasses.html';
    rel_path=['/disk2','../../..'];
    height_width=[400,400];
    pascalFlag=True;
        
    params=params(db_path=db_path,class_id=class_id,layer=layer,out_file_pre=out_file_pre,bins=bins,normed=normed,out_file_html=out_file_html,rel_path=rel_path,height_width=height_width,pascalFlag=pascalFlag);

    script_visualizeRankDifferenceAsHist(params)
    pickle.dump(params._asdict(),open(out_file_html+'_meta_experiment.p','wb'))

    return
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);

    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);
    pascal_labels=list(zip(*pascal_ids)[1]);
    pascal_labels=[id_curr.strip(' ') for id_curr in pascal_labels];
    pascal_labels[pascal_labels.index('dining_table')]='diningtable';
    pascal_labels[pascal_labels.index('tv/monitor')]='tvmonitor';
    
    pascal3d_ids=['boat', 'train', 'bicycle', 'chair', 'motorbike', 'aeroplane', 'sofa', 'diningtable', 'bottle', 'tvmonitor', 'bus', 'car'];
    idx_mapping=[];
    for id_curr in pascal3d_ids:
        idx_mapping.append(pascal_labels.index(id_curr));

    
    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids,True);

    db_path_out='sqlite://///disk2/novemberExperiments/nn_imagenet/nn_imagenet.db';
    mani=Imagenet_Manipulator(db_path_out);
    mani.openSession();
    
    for idx_idx,idx in enumerate(idx_mapping):
        pascal_id_curr=pascal3d_ids[idx_idx];
        pascal_idx_curr=idx_idx;

        print pascal_id_curr
        print pascal_idx_curr
        for to_exclude_curr in to_exclude[idx]:
            criterion=(Imagenet.class_id_imagenet==to_exclude_curr,);
            updateVals={Imagenet.class_id_pascal:pascal_id_curr,Imagenet.class_idx_pascal:pascal_idx_curr};
            print to_exclude_curr,val_just_labels[val_just_ids.index(to_exclude_curr)];
            mani.update(criterion,updateVals);

    mani.closeSession();
            

    # return


        # print idx_idx,idx,
        # print pascal3d_ids[idx_idx],
        # print pascal_labels[idx],
        # print len(to_exclude[idx])

    # return to_exclude


    return
    # class_id_pascal None
    # class_idx_pascal None

    db_path_out='sqlite://///disk2/novemberExperiments/nn_imagenet/nn_imagenet.db';
    mani=Imagenet_Manipulator(db_path_out);
    mani.openSession();
    vals=mani.filter((Imagenet.idx==0,));

    val=vals[0];
    for key_curr in val.__dict__.keys():
        print key_curr,val.__dict__[key_curr];
    print len(vals);
    mani.closeSession();

    return



    return
    out_dir='/disk2/novemberExperiments/nn_imagenet';
    out_file='equal_mix_ids.p';
    path_to_val='/disk2/imagenet/val'
    out_file=os.path.join(out_dir,out_file);
    all_idx_picked=pickle.load(open(out_file,'rb'));

    val_gt_file='../../data/ilsvrc12/val.txt'
    im_files_gt_classes=imagenet.selectTestSetByID(val_gt_file,all_idx_picked,path_to_val=path_to_val)
    im_files=list(zip(*im_files_gt_classes)[0])
    gt_classes=list(zip(*im_files_gt_classes)[1])

    synset_words='../../data/ilsvrc12/synset_words.txt'
    imagenet_idx_mapped,imagenet_ids_mapped,imagenet_labels_mapped=imagenet.getMappingInfo(im_files,synset_words,val_gt_file)
    print len(imagenet_labels_mapped)
    print 'getting to exclude'
    to_exclude=script_pascalClasses_get();
    print 'to_exclude',len(to_exclude);
    for id_curr in to_exclude:
        assert id_curr in imagenet_ids_mapped;
    
    print len(all_idx_picked),len(set(all_idx_picked));
    assert len(all_idx_picked)==len(set(imagenet_idx_mapped))==len(set(all_idx_picked));
    assert len(imagenet_idx_mapped)== len(imagenet_ids_mapped)== len(imagenet_labels_mapped)

    return
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);

    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);
    pascal_labels=list(zip(*pascal_ids)[1]);
    pascal_labels=[id_curr.strip(' ') for id_curr in pascal_labels];
    pascal_labels[pascal_labels.index('dining_table')]='diningtable';
    pascal_labels[pascal_labels.index('tv/monitor')]='tvmonitor';
    
    pascal3d_ids=['boat', 'train', 'bicycle', 'chair', 'motorbike', 'aeroplane', 'sofa', 'diningtable', 'bottle', 'tvmonitor', 'bus', 'car'];
    idx_mapping=[];
    for id_curr in pascal3d_ids:
        idx_mapping.append(pascal_labels.index(id_curr));

    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids,False);
    

    return
    # print to_exclude

    pascal_ids=[];
    pascal_idx=[];
    imagenet_ids=[];
    imagenet_idx=[];
    imagenet_labels=[];
    prev=0;
    dict_all=[];
    for idx,idx_mapping_curr in enumerate(idx_mapping):
        for idx_exclude_curr,exclude_curr in enumerate(to_exclude[idx_mapping_curr]):
            dict_curr={};
            dict_curr['pascal_id']=pascal3d_ids[idx];
            dict_curr['pascal_idx']=idx;
            dict_curr['imagenet_idx']=prev+idx_exclude_curr;
            dict_curr['imagenet_id']=exclude_curr;
            dict_curr['imagenet_label']=val_just_labels[val_just_ids.index(exclude_curr)];
            dict_all.append(dict_curr);
        prev=prev+len(to_exclude[idx_mapping_curr]);

    test_data=[];
    val_gt_file='../../data/ilsvrc12/val.txt'
    # mapping_file='../../data/ilsvrc12/synset_words.txt';

    # with open(mapping_file,'rb') as f:
    #     ids=f.readlines();
    #     ids=[id.strip('\n') for id in ids];
    # print ids[:10];
    # print val_just_ids[:10];
    # assert ids==val_just_ids

    out_file_html='/disk2/temp/verify.html';
    replace=['/disk2','..'];
    path_to_val='/disk2/imagenet/val';

    val_data=imagenet.readLabelsFile(val_gt_file)
    im_paths,im_idx=zip(*val_data);
    im_paths=list(im_paths);
    im_idx=[int(idx) for idx in im_idx];

    img_paths=[];
    captions=[];
    for idx,im_path in enumerate(im_paths):
        im_path=os.path.join(path_to_val,im_path);
        im_path_rel=im_path.replace(replace[0],replace[1]);
        img_paths.append([im_path_rel]);
        class_idx=im_idx[idx];
        captions.append([val_just_ids[class_idx]+' '+val_just_labels[class_idx]]);


    visualize.writeHTML(out_file_html,img_paths,captions);

    img_paths=[];
    for dict_curr in dict_all:
        dict_curr['img_paths']=[];
        imagenet_id=dict_curr['imagenet_id'];




    return
    selectTestSetByID(val_gt_file,list_of_ids,path_to_val=None,random=False,max_num=None)


    # f_info='/disk2/octoberExperiments/nn_performance_without_pascal/excluded_classes_info.txt'
    # with open(f_info,'wb') as f:
    #     for idx,to_exclude_curr in enumerate(to_exclude):
    #         f.write(str(pascal_ids[idx])+'\n');
    #         for to_exclude_id in to_exclude_curr:
    #             f.write(to_exclude_id+' '+val_just_labels[val_just_ids.index(to_exclude_id)]+'\n');
    #         f.write('___'+'\n');


if __name__=='__main__':
    main();