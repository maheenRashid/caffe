FOR script_scoreRandomFrames experiments_hashScoring
    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    class_idx_all=[tuple_curr[1] for tuple_curr in class_labels_map];
    # total_counts=getTotalCountsPerClass(path_to_db,class_idx_all)
    total_counts={0: 622034, 1: 245763, 2: 664689, 3: 125286, 4: 311316, 5: 500093, 6: 889816, 7: 839481, 8: 358913, 9: 1813897}
    params_path='/disk2/decemberExperiments/analyzing_scores/scores_of_samples_11.html_meta_experiment.p';
    params_dict=pickle.load(open(params_path,'rb'));
    out_file_pre=params_dict['out_file_frames'];
    out_file_pre=out_file_pre[:out_file_pre.rindex('.')]+'_frameCountNormalized';
    params_dict['out_file_frames'] = out_file_pre+'.p';
    params_dict['out_file_html'] = out_file_pre+'.html';
    params_dict['n_jobs']=12;
    params_dict['frameCountNorm']=True

    params=createParams('scoreRandomFrames');
    params=params(**params_dict);

    script_scoreRandomFrames(params)
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));

FOR script_saveNpzScorePerShot experiments_hashScoring
    out_dir='/disk2/decemberExperiments/shot_scores';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);


    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    mani=TubeHash_Manipulator(path_to_db);
    mani.openSession();
    vals = mani.select((Tube.class_idx_pascal,Tube.video_id,Tube.shot_id),distinct=True);
    # deep_features_path = mani.select((Tube.deep_features_path,),distinct= True);
    mani.closeSession();

    params_dict = {};
    params_dict['total_class_counts'] = {0: 622034, 1: 245763, 2: 664689, 3: 125286, 4: 311316, 5: 500093, 6: 889816, 7: 839481, 8: 358913, 9: 1813897};
    params_dict['path_to_db'] = 'sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db'
    params_dict['path_to_hash'] = '/disk2/decemberExperiments/hash_tables'
    params_dict['num_hash_tables'] = 32

    # params_obj=createParams('saveNpzScorePerShot');
    params_all=[];
    for idx_val,val in enumerate(vals):
        params_dict['class_idx'] =  val[0];
        params_dict['video_id'] =  val[1];
        params_dict['shot_id'] =  val[2]
        params_dict['out_file_scores'] =  os.path.join(out_dir,'_'.join(map(str,val))+'.p')
        params_dict['idx']=idx_val
        # params=params_obj(**params_dict);
        # params_all.append(params);
        params_all.append(copy.deepcopy(params_dict))

    print len(params_all);
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.map(script_saveNpzScorePerShot,params_all)



FOR script_fixHorseCountError experiments_hashScoring
    score_dir='/disk2/decemberExperiments/shot_scores';
    out_dir='/disk2/decemberExperiments/shot_scores_analysis';
    path_to_patches='/disk2/res11/tubePatches';
    class_labels_map = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    [class_labels,class_idx_all]=zip(*class_labels_map);
    


    selected_class=7
    class_label=class_labels[class_idx_all.index(selected_class)]
    out_file=os.path.join(out_dir,'all_scores_patches_'+class_label+'.p')
    print out_file
    score_files=[os.path.join(score_dir,file_curr) for file_curr in os.listdir(score_dir) if file_curr.endswith('.p') and file_curr.startswith(str(selected_class)+'_')];
    out_file='/disk2/temp/horse_debug.p'
    saveRecordOfCountErrorFiles(score_files,class_label,path_to_patches,out_file)
    script_fixHorseCountError(out_file)

FOR getListScoresAndPatches experiments_hashScoring

    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    score_dir='/disk2/decemberExperiments/shot_scores';
    out_dir='/disk2/decemberExperiments/shot_scores_analysis';
    path_to_patches='/disk2/res11/tubePatches';
    class_labels_map = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    [class_labels,class_idx_all]=zip(*class_labels_map);
    for selected_class in [3]:
        class_label=class_labels[class_idx_all.index(selected_class)]
        out_file=os.path.join(out_dir,'all_scores_patches_'+class_label+'.p')
        print out_file
        score_files=[os.path.join(score_dir,file_curr) for file_curr in os.listdir(score_dir) if file_curr.endswith('.p') and file_curr.startswith(str(selected_class)+'_')];
        list_scores,list_files=getListScoresAndPatches(score_files,class_label,path_to_patches)
        pickle.dump([list_scores,list_files],open(out_file,'wb'));

FOR visualizeRankedPatchesPerClass experiments_hashScoring

    out_dir='/disk2/decemberExperiments/shot_scores_analysis';
    class_labels_map = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    [class_labels,class_idx_all]=zip(*class_labels_map);
    
    num_to_display=100;
    out_file_html=os.path.join(out_dir,'all_scores_patches_sorted_'+str(num_to_display))+'.html';
    rel_path=['/disk2','../../..'];
    height_width=[400,400];
    
    class_score_info=[];
    for class_label,class_idx in class_labels_map:
        out_file=os.path.join(out_dir,'all_scores_patches_'+class_label+'.p')
        tuple_curr=(class_idx,class_label,out_file)
        print tuple_curr
        class_score_info.append(tuple_curr);
    
    visualizeRankedPatchesPerClass(class_score_info,num_to_display,out_file_html,rel_path,height_width)

FOR visualizeBestTubeRank experiments_hashScoring
    params_dict={};
    params_dict['class_labels_map'] = [('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    params_dict['rel_path'] = ['/disk2','../../..']
    params_dict['out_file_html'] = '/disk2/decemberExperiments/shot_scores_analysis/best_tube_hists.html';
    params_dict['out_dir'] = '/disk2/decemberExperiments/shot_scores_analysis';    
    params_dict['score_info_file'] = os.path.join(params_dict['out_dir'],'shot_scores_info.p');
    params=createParams('visualizeBestTubeRank');
    params=params(**params_dict);
    visualizeBestTubeRank(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));

FOR script_testNpzScoreAccuracy experiments_hashScoring 
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    path_to_hash='/disk2/decemberExperiments/hash_tables';
    total_class_counts={0: 622034, 1: 245763, 2: 664689, 3: 125286, 4: 311316, 5: 500093, 6: 889816, 7: 839481, 8: 358913, 9: 1813897}
    class_idx = 0;
    video_id = 1;
    shot_id = 1;
    num_hash_tables=32;
    num_hash_vals=256;
    tube_id=0;
    deep_features_idx=0;
    tube_file='/disk2/temp/temp.p';

    params_dict={};
    params_dict['path_to_db'] = path_to_db
    params_dict['path_to_hash'] = path_to_hash
    params_dict['total_class_counts'] = total_class_counts
    params_dict['class_idx'] = class_idx 
    params_dict['video_id'] = video_id 
    params_dict['shot_id'] = shot_id 
    params_dict['num_hash_tables'] = num_hash_tables
    params_dict['num_hash_vals'] = num_hash_vals
    params_dict['tube_id'] = tube_id
    params_dict['deep_features_idx'] = deep_features_idx
    params_dict['tube_file']=tube_file;
    params=createParams('testNpzScoreAccuracy');
    params=params(**params_dict);
    script_testNpzScoreAccuracy(params)

FOR script_saveCounts experiments_hashScoring
    path_to_hash='/disk2/decemberExperiments/hash_tables';
    num_hash_tables=32;
    num_hash_vals=256;
    n_jobs=12;
    script_saveCounts(path_to_hash,num_hash_tables,num_hash_vals,n_jobs)

FOR script_scoreRandomFrames experiments_hashScoring    
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    npz_path='/disk2/decemberExperiments/hash_tables';
    numberOfFrames=10;
    max_idx=6371288;
    [class_labels,class_idx]=zip(*class_labels_map)
    n_jobs=32;
    table_idx_all=random.sample(xrange(max_idx), numberOfFrames)
    
    out_dir='/disk2/decemberExperiments/analyzing_scores';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);
    out_file_html=os.path.join(out_dir,'scores_of_samples_'+str(numberOfFrames)+'.html');
    rel_path=['/disk2','../../..']
    width_height=[300,300];

    out_file_frames=os.path.join(out_dir,'scores_of_samples_'+str(numberOfFrames)+'.p');
    params_dict={};
    params_dict['path_to_db'] = path_to_db;
    params_dict['class_labels_map'] = class_labels_map;
    params_dict['npz_path'] = npz_path;
    params_dict['numberOfFrames'] = numberOfFrames;
    params_dict['max_idx'] = max_idx;
    params_dict['n_jobs'] = n_jobs;
    params_dict['table_idx_all'] = table_idx_all;
    params_dict['out_file_html'] = out_file_html;
    params_dict['rel_path'] = rel_path;
    params_dict['width_height'] = width_height;
    params_dict['out_file_frames'] = out_file_frames;
    params=createParams('scoreRandomFrames');
    params=params(**params_dict);
    script_scoreRandomFrames(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));


FOR script_visualizeNNComparisonWithHash experiments_hashing    
    params_dict={};
    params_dict['in_file']='/disk2/decemberExperiments/toyExample/tubes_nn_32.p'
    params_dict['in_file_hash']='/disk2/decemberExperiments/toyExample/tubes_nn_32_8_32.p'
    params_dict['out_file_html']=params_dict['in_file_hash'][:-2]+'_qualitative_comparsion.html';
    params_dict['rel_path']=['/disk2','../../..'];
    params_dict['topn']=10;
    params_dict['img_size']=[200,200];
    params=createParams('visualizeNNComparisonWithHash');
    params=params(**params_dict);
    script_visualizeNNComparisonWithHash(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));
    


FOR script_saveBigFeatureMats experiments_hashing

    out_dir='/disk2/decemberExperiments/gettingNN';
    out_dir_featureMats=os.path.join(out_dir,'big_feature_mats');
    
    if not os.path.exists(out_dir_featureMats):
        os.mkdir(out_dir_featureMats);

    out_file_featureMats_pre=os.path.join(out_dir_featureMats,'feature_mats');
    out_file_meta_pre=os.path.join(out_dir_featureMats,'feature_mats_meta');

    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    out_file_paths=os.path.join(out_dir,'paths_to_features_shuffled.p');
    num_batches=50;

    params_dict={};
    params_dict['out_file_featureMats_pre'] = out_file_featureMats_pre
    params_dict['out_file_meta_pre'] = out_file_meta_pre
    params_dict['path_to_db'] = path_to_db
    params_dict['out_file_paths'] = out_file_paths
    params_dict['num_batches'] = num_batches

    params=createParams('saveBigFeatureMats');
    params=params(**params_dict);

    script_saveBigFeatureMats(params);

    pickle.dump(params._asdict(),open(out_file_featureMats_pre+'_meta_experiment.p','wb'));



FOR saveHashAnalysisImages HTML experiments_hashing
    out_dir='/disk2/decemberExperiments/analysis_8_32/detailed'
    out_file_html='/disk2/decemberExperiments/analysis_8_32/detailed.html'
    img_size=[450,600];
    rel_path=['/disk2','../../../..'];
    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]

    img_paths=[];
    captions=[];

    # out_dir='temp'
    for hashtable in range(32):
        out_file_pre='hashtable_'+str(hashtable);
        img_row=[];
        caption_row=[];
        img_row.append(os.path.join(out_dir,out_file_pre+'_simple.png'))
        img_row.append(os.path.join(out_dir,out_file_pre+'_byClass.png'));
        caption_row.append('Simple sorted bin composition');
        caption_row.append('Sorted bin composition colored by class');
        for class_id,class_idx in class_labels_map:
            file_curr=os.path.join(out_dir,out_file_pre+'_'+class_id+'.png');
            img_row.append(file_curr);
            caption_row.append(class_id);

        img_row=[img_curr.replace(rel_path[0],rel_path[1]) for img_curr in img_row];
        img_paths.append(img_row);
        captions.append(caption_row);

    visualize.writeHTML(out_file_html,img_paths,captions,img_size[0],img_size[1]);

    
FOR script_saveHashAnalysisImages experiments_hashing
    params_dict={};
    out_dir='/disk2/decemberExperiments/analysis_8_32/detailed'
    # out_dir='temp'
    for hashtable in range(1,32):
        print hashtable
        t=time.time();
        out_file_pre='hashtable_'+str(hashtable);
        params_dict['path_to_db']='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
        params_dict['class_labels_map']=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
        params_dict['percents']=[0.25,0.5,0.75]
        params_dict['out_file_class_pre'] =os.path.join(out_dir,out_file_pre);
        params_dict['out_file_hash_simple'] = os.path.join(out_dir,out_file_pre+'_simple.png');
        params_dict['out_file_hash_byClass'] = os.path.join(out_dir,out_file_pre+'_byClass.png');
        params_dict['hashtable'] = hashtable
        params_dict['inc'] = 5
        params_dict['dtype']=np.uint8
        # params_dict['in_file']='temp/hash_1_breakdown.npz'
        params=createParams('saveHashAnalysisImages');
        params=params(**params_dict);
        script_saveHashAnalysisImages(params);
        pickle.dump(params._asdict(),open(os.path.join(out_dir,out_file_pre+'_meta_experiment.p'),'wb'))
        print time.time()-t


FOR script_visualizeHashBinDensity youtube
    out_dir='/disk2/decemberExperiments/analysis_8_32';
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);

    params_dict={};
    params_dict['hash_tables'] = range(32);
    in_file_pre = '/disk2/novemberExperiments/experiments_youtube/patches_nn_hash_densities'
    params_dict['in_files'] = [in_file_pre+'_'+str(hash_table)+'.p' for hash_table in params_dict['hash_tables']];
    params_dict['out_files'] = [os.path.join(out_dir,'densities_'+str(hash_table)+'.png') for hash_table in params_dict['hash_tables']];
    params_dict['out_file_html'] = os.path.join(out_dir,'densities_all.html');
    params_dict['rel_path'] = ['/disk2','../../..'];
    params_dict['bins'] = 20;
    params_dict['height_width'] = [450,600];

    params=createParams('visualizeHashBinDensity');
    params=params(**params_dict);
    script_visualizeHashBinDensity(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));


FOR script_saveHashTableDensities youtube
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    out_file_pre='/disk2/novemberExperiments/experiments_youtube/patches_nn_hash_densities'

    for hash_table in range(32):
        out_file=out_file_pre+'_'+str(hash_table)+'.p';
        print out_file
        t=time.time();
        script_saveHashTableDensities(hash_table,path_to_db,out_file);
        print time.time()-t;


FOR script_toyNNExperiment youtube
    out_file_html='/disk2/temp/tubes_nn_16.html';
    params_dict=pickle.load(open(out_file_html+'_meta_experiment.p','rb'));
    params_dict['dtype']='float32';
    params_dict['out_file_html']=os.path.join(out_dir,'tubes_nn_32.html');
    params_dict['out_file_hist']=os.path.join(out_dir,'tubes_nn_32.png');
    params_dict['out_file_pickle']=os.path.join(out_dir,'tubes_nn_32.p');
    
    params=createParams('toyNNExperiment');
    params=params(**params_dict);
    script_toyNNExperiment(params)


FOR script_compareHashWithToyExperiment youtube
    params_dict={};
    params_dict['in_file']='/disk2/decemberExperiments/toyExample/tubes_nn_big_32.p'
    params_dict['num_hash_tables_all']=[32,64,128,512,1024]
    params_dict['key_type']=np.uint16;
    type_str=str(16)
    params_dict['out_file_pres']=[params_dict['in_file'][:-2]+'_'+type_str+'_'+str(x) for x in params_dict['num_hash_tables_all']];
    params_dict['out_file_indices']=params_dict['in_file'][:-2]+'_indices.png'
    params_dict['out_file_html']=params_dict['in_file'][:-2]+'_'+type_str+'.html';
    params_dict['rel_path']=['/disk2','../../..'];

    params=createParams('compareHashWithToyExperiment');
    params=params(**params_dict);
    script_compareHashWithToyExperiment(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));


FOR saveHash youtube
    hash_file='/disk2/novemberExperiments/experiments_youtube/hasher.npy'
    # feature_dim=4096;
    # num_hash_tables=32;
    # hp_hash=lsh.HyperplaneHash((feature_dim,num_hash_tables),key_type=np.uint8);
    # np.save(hash_file,hp_hash.hasher);

    # return

    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_test.db';
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    deep_features_path=mani.select((Tube.deep_features_path,),distinct=True);
    deep_features_path=[x[0] for x in deep_features_path]
    mani.closeSession();    
    print len(deep_features_path),deep_features_path[:1];

    deep_features_path_all=deep_features_path;
    args=[];
    for idx,deep_features_path in enumerate(deep_features_path_all):
        out_file=deep_features_path[:-4]+'_hash';
        key_type=np.uint8
        arg_curr=(hash_file,deep_features_path,out_file,key_type,idx)
        args.append(arg_curr);

    p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    p.map(saveHash,args)


FOR script_toyNNExperiment youtube
    path_to_db='sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_test.db';
    class_id_pascal='aeroplane';
    video_id=10;
    shot_id=1;
    tube_id=0;
    numberofVideos=2;
    numberOfFrames=120;
    out_file_html='/disk2/temp/tubes_nn_16.html';
    rel_path=('/disk2','..');
    out_file_hist='/disk2/temp/tubes_nn_16.png';
    gpuFlag=True;
    dtype='float16';
    mani=Tube_Manipulator(path_to_db);
    mani.openSession();
    pascal_ids=mani.select((Tube.class_id_pascal,),distinct=True);
    pascal_ids=[pascal_id[0] for pascal_id in pascal_ids];
    mani.closeSession();

    video_info={};
    for pascal_id in pascal_ids:
        video_info[pascal_id]=[1,2]

    params_dict={};
    params_dict['path_to_db'] = path_to_db
    params_dict['class_id_pascal'] = class_id_pascal
    params_dict['video_id'] = video_id
    params_dict['shot_id'] = shot_id
    params_dict['tube_id'] = tube_id
    params_dict['numberofVideos'] = numberofVideos
    params_dict['numberOfFrames'] = numberOfFrames
    params_dict['out_file_html'] = out_file_html
    params_dict['rel_path'] = rel_path
    params_dict['out_file_hist'] = out_file_hist
    params_dict['gpuFlag'] = gpuFlag
    params_dict['dtype'] = dtype
    params_dict['pascal_ids'] = pascal_ids
    params_dict['video_info'] = video_info
    params=createParams('toyNNExperiment');
    params=params(**params_dict);
    
    script_toyNNExperiment(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));



FOR script_saveTubePatches youtube
    path_to_data='/disk2/youtube/categories'
    out_dir_patches='/disk2/res11/tubePatches';
    path_to_mat='/disk2/res11';

    if not os.path.exists(out_dir_patches):
        os.mkdir(out_dir_patches);

    mat_files=[os.path.join(path_to_mat,file_curr) for file_curr in os.listdir(path_to_mat) if file_curr.endswith('.mat')]
    # idx=18;

    # mat_files=mat_files[:idx]
    script_saveTubePatches(mat_files,path_to_data,out_dir_patches,numThreads=8)

FOR script_breakUpInFilesListForFeatureExtraction youtube
    path_to_patches='/disk2/res11/tubePatches'
    file_index='/disk2/res11/featureExtractionInputOutputFiles/list_of_files.p';
    in_file_meta_pre='/disk2/res11/featureExtractionInputOutputFiles/in_files';
    out_file_meta_pre='/disk2/res11/featureExtractionInputOutputFiles/out_files'
    
    script_breakUpInFilesListForFeatureExtraction(file_index,in_file_meta_pre,out_file_meta_pre)

FOR fixing val.txt
    new_val='/disk2/octoberExperiments/nn_performance_without_pascal/new_val.txt'
    parent_val='../../data/ilsvrc12/val.txt';
    synsets='/disk2/octoberExperiments/nn_performance_without_pascal/synsets.txt';
    class_ids=imagenet.readSynsetsFile(synsets);
    print len(class_ids);
    print class_ids[:10];

    synsets_parent='../../data/ilsvrc12/synsets.txt';
    class_ids_parent=imagenet.readSynsetsFile(synsets_parent);
    print len(class_ids_parent);
    print class_ids_parent[:10];

    idx_to_keep=np.where(np.in1d(class_ids_parent,class_ids))[0];

    val_gt_data=imagenet.readLabelsFile(parent_val);
    img_paths,class_idx=zip(*val_gt_data);
    class_idx=np.array([int(class_idx_curr) for class_idx_curr in class_idx]);
    img_paths=np.array(list(img_paths));

    img_paths_to_keep=[];
    class_idx_to_keep=[];
    for idx_to_keep_curr in idx_to_keep:
        # print idx_to_keep_curr
        img_paths_rel=img_paths[class_idx==idx_to_keep_curr];
        class_idx_rel=class_ids.index(class_ids_parent[idx_to_keep_curr]);
        class_idx_to_keep.extend([class_idx_rel]*len(img_paths_rel));
        img_paths_to_keep.extend(img_paths_rel);

    print len(img_paths_to_keep),len(set(class_idx_to_keep)),min(class_idx_to_keep),max(class_idx_to_keep)
    assert sorted(list(set(class_idx_to_keep)))==range(len(idx_to_keep));
    with open(new_val,'wb') as f:
        for idx in range(len(img_paths_to_keep)):
            f.write(img_paths_to_keep[idx]+' '+str(class_idx_to_keep[idx])+'\n');


FOR visualizePascalNeighborsFromOtherClass
    db_path_out='sqlite://///disk2/novemberExperiments/nn_imagenet/nn_imagenet.db';
    class_id_pascal='car';
    limit=None;
    layer='fc7';
    trainFlag=False;
    rel_path=['/disk2','../../..'];
    out_file_html='/disk2/novemberExperiments/nn_imagenet/car_nn_non_car_new.html';
    top_n=5;
    height_width=[300,300];
    
    params=createParams('visualizePascalNeighborsFromOtherClass');
    params=params(db_path_out=db_path_out,class_id_pascal=class_id_pascal,limit=limit,layer=layer,trainFlag=trainFlag,rel_path=rel_path,out_file_html=out_file_html,top_n=top_n,height_width=height_width);
    script_visualizePascalNeighborsFromOtherClass(params);
    pickle.dump(params._asdict(),open(params.out_file_html+'_meta_experiment.p','wb'));


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

