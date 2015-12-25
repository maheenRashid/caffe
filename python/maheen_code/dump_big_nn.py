FOR getNNIndicesForBigFeatureMats experiments_hashing
    
    out_dir='/disk2/decemberExperiments/gettingNN';
    out_dir_featureMats=os.path.join(out_dir,'big_feature_mats');
    meta_replace=['/feature_mats','/feature_mats_meta']
    
    big_feature_files=[os.path.join(out_dir_featureMats,file_curr) for file_curr in os.listdir(out_dir_featureMats) if file_curr.endswith('.npz')];
    # big_feature_files=big_feature_files[:3];

    # meta_feature_files=[];
    # for file_curr in big_feature_files:
    #     file_curr=file_curr.replace(meta_replace[0],meta_replace[1]);
    #     file_curr=file_curr[:file_curr.rfind('.')]+'.p';
    #     meta_feature_files.append(file_curr);
    #     assert os.path.exists(file_curr);

    # meta_info_all=[];
    # for meta_file_curr in meta_feature_files:
    #     [paths,sizes]=pickle.load(open(meta_file_curr,'rb'));
    #     meta_info_all.append([paths,sizes]);

    test_path= '/disk2/res11/tubePatches/aeroplane_10_1/0/0.npz';
    # meta_info_all[0][0][0]

    print test_path
    out_path=test_path[:test_path.rfind('.')]+'_indices.npz';
    print out_path
    script_getNNIndicesForTestMat([test_path],big_feature_files,[out_path])

    return
    t=time.time();
    for test_mat_no in range(len(meta_info_all)):
        [paths,sizes]=meta_info_all[test_mat_no];
        for test_no in range(len(sizes)):

            start_idx=sum(sizes[:test_no]);
            end_idx=start_idx+sizes[test_no];

            test=mats[test_mat_no][start_idx:end_idx,:];

            indices = getNNIndicesForBigFeatureMats(test,mats);
            print indices[:10,:10]
            print indices.shape,type(indices[0,0])
            path_out=path_curr[:path_curr.rfind('.')]+'_indices.npz';
            print path_out
            np.savez(path_out,indices);
            break;
        break;
    print time.time()-t,'Time'

    