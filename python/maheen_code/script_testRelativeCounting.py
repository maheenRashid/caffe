from experiments_hashScoring import *;


def main():
    path_to_db = 'sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    class_labels_map=[('boat', 2), ('train', 9), ('dog', 6), ('cow', 5), ('aeroplane', 0), ('motorbike', 8), ('horse', 7), ('bird', 1), ('car', 3), ('cat', 4)]
    out_dir='/disk2/decemberExperiments/alternative_scoring/comparison_scoring';
    [class_labels,class_idx_all]=zip(*class_labels_map)
    
    # img_path='/disk2/res11/tubePatches/dog_21_19/6/63.jpg';
    # deep_features_idx=157
    # out_file_pre=os.path.join(out_dir,'best_dog_shot_');
    # original_scores_file='/disk2/decemberExperiments/shot_scores/6_21_19.p'
    
    # img_path='/disk2/res11/tubePatches/dog_35_1/0/33.jpg';
    # deep_features_idx=None
    # out_file_pre=os.path.join(out_dir,'7_dog_shot');
    
    # img_path='/disk2/res11/tubePatches/dog_35_1/0/33.jpg';
    # deep_features_idx=None
    # out_file_pre=os.path.join(out_dir,'trial_dog_shot');

    # img_path='/disk2/res11/tubePatches/cat_10_4/1/283.jpg';
    # deep_features_idx=None;
    # out_file_pre=os.path.join(out_dir,'best_cat_shot');    
    
    img_path='/disk2/res11/tubePatches/cat_10_6/5/91.jpg';
    deep_features_idx=None;
    out_file_pre=os.path.join(out_dir,'7_cat_shot');    

    class_label,video_id,shot_id,tube_id,_=getPatchInfoFromPath(img_path);
    class_idx=class_idx_all[class_labels.index(class_label)];
    
    if deep_features_idx is None:
        mani=Tube_Manipulator(path_to_db);
        mani.openSession();
        deep_features_idx=mani.select((Tube.deep_features_idx,),(Tube.img_path==img_path,));
        deep_features_idx=deep_features_idx[0][0];
        mani.closeSession();


    score_files=[];
    params={};
    params['path_to_db'] = 'sqlite://///disk2/novemberExperiments/experiments_youtube/patches_nn_hash.db';
    params['total_class_counts']= {0: 622034, 1: 245763, 2: 664689, 3: 125286, 4: 311316, 5: 500093, 6: 889816, 7: 839481, 8: 358913, 9: 1813897}
    params['class_idx'] = class_idx;
    params['video_id'] = video_id;
    params['shot_id'] = shot_id;
    params['path_to_hash'] ='/disk2/decemberExperiments/hash_tables'
    params['num_hash_tables']=32
    params['idx']=0;
    
    for class_label,class_idx_assume in class_labels_map:
        params['class_idx_assume']=class_idx_assume;
        params['out_file_scores']=out_file_pre+'_'+class_label+'.p';
        if not os.path.exists(params['out_file_scores']):
            script_saveNpzScorePerShot(params);
            print params['out_file_scores']
        score_files.append(params['out_file_scores']);


    scores_best_patch=[];

    for file_curr in score_files:
        scores_curr=pickle.load(open(file_curr,'rb'));
        scores_best_patch_curr=scores_curr[tube_id][deep_features_idx,:]
        # print scores_best_patch_curr.shape
        scores_best_patch.append(np.mean(scores_best_patch_curr));

    print out_file_pre
    max_idx=np.argmax(scores_best_patch);
    print 'max',scores_best_patch[max_idx],max(scores_best_patch),class_labels[max_idx];
    sort_idx=np.argsort(scores_best_patch)[::-1];
    # print sort_idx
    for idx_curr in sort_idx:
        print class_labels[idx_curr],scores_best_patch[idx_curr];


    return

    class_labels=['dog']*3;
    class_idx_all=[6]*3;
    score_files=[original_scores_file,out_file_pre+'_dog.p',out_file_pre+'_original.p'];

    scores_all=[pickle.load(open(score_file_curr,'rb')) for score_file_curr in score_files];
    for tube_id_curr in scores_all[0].keys():
        print tube_id_curr
        scores_rel=[scores_curr[tube_id_curr] for scores_curr in scores_all];
        assert len(scores_rel)==3;
        print np.allclose(scores_rel[0],scores_rel[1]),np.allclose(scores_rel[0],scores_rel[2]),np.allclose(scores_rel[1],scores_rel[2])
        assert np.allclose(scores_rel[0],scores_rel[1])
        assert np.allclose(scores_rel[0],scores_rel[2])
        assert np.allclose(scores_rel[1],scores_rel[2]);
    

if __name__=='__main__':
    main();