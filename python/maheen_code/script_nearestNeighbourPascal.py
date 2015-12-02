import imagenet;
import script_nearestNeigbourExperiment;
import pickle
import numpy as np;
import os;
import matplotlib.pyplot as plt;

def main():
    # script_runNNOnPascalExcludedInTraining()
    script_visualizePerformanceDifference();

def script_setUpPascalExcludedTextFiles():
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);
    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);

    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids);

    im_list_file='../../data/ilsvrc12/val.txt';
    mapping_file='../../data/ilsvrc12/synsets.txt';

    ims_to_keep,class_ids_to_keep,classes_to_keep=imagenet.removeImagesFromListByClass(im_list_file,mapping_file,to_exclude);
    
    new_file_val='/disk2/octoberExperiments/nn_performance_without_pascal/val.txt';
    classes_uni_val=writeNewDataClassFile(new_file_val,zip(ims_to_keep,classes_to_keep));
    
    im_list_file='../../data/ilsvrc12/train.txt';
    ims_to_keep,class_ids_to_keep,classes_to_keep=imagenet.removeImagesFromListByClass(im_list_file,mapping_file,to_exclude);

    new_file_val='/disk2/octoberExperiments/nn_performance_without_pascal/train.txt';
    classes_uni_train=imagenet.writeNewDataClassFile(new_file_val,zip(ims_to_keep,classes_to_keep));

    assert(str(classes_uni_val)==str(classes_uni_train))
    
    class_file='/disk2/octoberExperiments/nn_performance_without_pascal/synsets.txt';
    
    with open(class_file,'wb') as f:
        for class_id in classes_uni_train:
            f.write(class_id+'\n');

    with open(new_file_val,'rb') as f:
        content=f.read();

    #sanity check
    for id_to_exclude in to_exclude:
        if id_to_exclude in content:
            print 'FOUND ERROR',id_to_exclude

def script_printExcludedInfoFile():
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);

    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);

    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids,keepMapping=True);

    f_info='/disk2/octoberExperiments/nn_performance_without_pascal/excluded_classes_info.txt'
    with open(f_info,'wb') as f:
        for idx,to_exclude_curr in enumerate(to_exclude):
            f.write(str(pascal_ids[idx])+'\n');
            for to_exclude_id in to_exclude_curr:
                f.write(to_exclude_id+' '+val_just_labels[val_just_ids.index(to_exclude_id)]+'\n');
            f.write('___'+'\n');


def script_runNNOnPascalExcludedInTraining():
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);
    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);

    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids,keepMapping=True);

    val_gt_file='../../data/ilsvrc12/val.txt'
    list_of_ids_im=[id for id_list in to_exclude for id in id_list];
    mapping_file='../../data/ilsvrc12/synsets.txt';
    print len(list_of_ids_im)

    list_of_ids,_=imagenet.getImagenetIdToTrainingIdMapping(mapping_file,list_of_ids_im)
    print len(list_of_ids)
    # print list_of_ids[0]
    list_of_ids_pascal=[];

    for id_no in range(len(to_exclude)):
        list_of_ids_pascal=list_of_ids_pascal+[id_no]*len(to_exclude[id_no])
    
    path_to_val='/disk2/imagenet/val'
    test_set=imagenet.selectTestSetByID(val_gt_file,list_of_ids,path_to_val);
    
    # out_dir='/disk2/octoberExperiments/nn_performance_without_pascal/notrained'
    out_dir='/disk2/novemberExperiments/nn_imagenet_top5/trained'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir);

    layers=['pool5','fc6','fc7'];
    gpu_no=1
    path_to_classify='..';
    numberOfN=5
    relativePaths=['/disk2','../../../..'];

    # deployFile='/disk2/octoberExperiments/nn_performance_without_pascal/deploy.prototxt'
    # meanFile='/disk2/octoberExperiments/nn_performance_without_pascal/mean.npy'
    # modelFile='/disk2/octoberExperiments/nn_performance_without_pascal/snapshot_iter_450000.caffemodel'
    
    modelFile='/home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    deployFile='/home/maheenrashid/Downloads/caffe/caffe-rc2/models/bvlc_reference_caffenet/deploy.prototxt';
    meanFile='/home/maheenrashid/Downloads/caffe/caffe-rc2/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    

    # modelFile='/disk2/novemberExperiments/network_no_pascal/snapshots/snapshot_iter_450000.caffemodel';
    # deployFile='/disk2/novemberExperiments/network_no_pascal/deploy.prototxt';
    # meanFile='/disk2/novemberExperiments/network_no_pascal/mean.npy';
    
    # out_file=script_nearestNeigbourExperiment.runClassificationTestSet(test_set,out_dir,path_to_classify,gpu_no,layers,deployFile=deployFile,meanFile=meanFile,modelFile=modelFile)

    # return
    # file_name='/disk2/octoberExperiments/nn_performance_without_pascal/notrained/20151026132705'
    # file_name='/disk2/novemberExperiments/nn_imagenet_top5/notrained/20151130193757';
    file_name='/disk2/novemberExperiments/nn_imagenet_top5/trained/20151130230243'
    file_text_labels='../../data/ilsvrc12/synset_words.txt'

    text_labels= np.loadtxt(file_text_labels, str, delimiter='\t')

    vals=np.load(file_name+'.npz');
    
    test_set=sorted(test_set,key=lambda x: x[0])
    test_set=zip(*test_set);
    img_paths=list(test_set[0]);
    gt_labels=list(test_set[1]);
    gt_labels_pascal=[list_of_ids_pascal[list_of_ids.index(gt_label)] for gt_label in gt_labels];

    for layer in layers:
        print layer
        file_name_l=file_name+'_'+layer;
        indices=script_nearestNeigbourExperiment.doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute')
        conf_matrix=0;
        pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));


    file_text_labels_pascal='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    text_labels_pascal= np.loadtxt(file_text_labels_pascal, str, delimiter='\t')

    for layer in layers:
        print layer
        file_name_l=file_name+'_'+layer;
        [img_paths,gt_labels,indices,_]=pickle.load(open(file_name_l+'.p','rb'));
        img_paths_curr=[x.replace(relativePaths[0],relativePaths[1]) for x in img_paths];
        im_paths,captions=script_nearestNeigbourExperiment.createImageAndCaptionGrid(img_paths_curr,gt_labels,indices,text_labels)
        script_nearestNeigbourExperiment.writeHTML(file_name_l+'.html',im_paths,captions)
        no_correct,_=script_nearestNeigbourExperiment.getNumberOfCorrectNNMatches(indices,gt_labels);
        print no_correct
        with open(file_name_l+'.txt','wb') as f:
            for no_correct_curr in no_correct:
                f.write(str(no_correct_curr)+' ');



        file_name_l=file_name+'_'+layer+'_pascal';
        im_paths,captions=script_nearestNeigbourExperiment.createImageAndCaptionGrid(img_paths_curr,gt_labels_pascal,indices,text_labels_pascal)
        script_nearestNeigbourExperiment.writeHTML(file_name_l+'.html',im_paths,captions)
        no_correct,_=script_nearestNeigbourExperiment.getNumberOfCorrectNNMatches(indices,gt_labels_pascal);
        with open(file_name_l+'.txt','wb') as f:
            for no_correct_curr in no_correct:
                f.write(str(no_correct_curr)+' ');

        print no_correct


def script_runNNOnPascalIncludedInTraining():
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);
    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);

    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids,keepMapping=True);

    val_gt_file='../../data/ilsvrc12/val.txt'
    list_of_ids_im=[id for id_list in to_exclude for id in id_list];
    mapping_file='../../data/ilsvrc12/synsets.txt';

    list_of_ids=imagenet.getImagenetIdToTrainingIdMapping(mapping_file,list_of_ids_im)
    print len(list_of_ids)

    list_of_ids_pascal=[];

    for id_no in range(len(to_exclude)):
        list_of_ids_pascal=list_of_ids_pascal+[id_no]*len(to_exclude[id_no])
    
    path_to_val='/disk2/imagenet/val'
    test_set=imagenet.selectTestSetByID(val_gt_file,list_of_ids,path_to_val);
    
    out_dir='/disk2/octoberExperiments/nn_performance_without_pascal/trained'
    layers=['pool5','fc6','fc7'];
    gpu_no=0
    path_to_classify='..';
    numberOfN=5
    relativePaths=['/disk2','../../../..'];
    # out_file=script_nearestNeigbourExperiment.runClassificationTestSet(test_set,out_dir,path_to_classify,gpu_no,layers)

    file_name='/disk2/octoberExperiments/nn_performance_without_pascal/trained/20151023153522'
    
    file_text_labels='../../data/ilsvrc12/synset_words.txt'

    text_labels= np.loadtxt(file_text_labels, str, delimiter='\t')

    vals=np.load(file_name+'.npz');
    
    test_set=sorted(test_set,key=lambda x: x[0])
    test_set=zip(*test_set);
    img_paths=list(test_set[0]);
    gt_labels=list(test_set[1]);
    gt_labels_pascal=[list_of_ids_pascal[list_of_ids.index(gt_label)] for gt_label in gt_labels];

    # for layer in layers:
    #     file_name_l=file_name+'_'+layer;
    #     indices,conf_matrix=doNN(img_paths,gt_labels,vals[layer],numberOfN=numberOfN,distance='cosine',algo='brute')
    #     pickle.dump([img_paths,gt_labels,indices,conf_matrix],open(file_name_l+'.p','wb'));


    file_text_labels_pascal='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    text_labels_pascal= np.loadtxt(file_text_labels_pascal, str, delimiter='\t')

    for layer in layers:
        print layer
        file_name_l=file_name+'_'+layer;
        [img_paths,gt_labels,indices,_]=pickle.load(open(file_name_l+'.p','rb'));
        img_paths_curr=[x.replace(relativePaths[0],relativePaths[1]) for x in img_paths];
        im_paths,captions=script_nearestNeigbourExperiment.createImageAndCaptionGrid(img_paths_curr,gt_labels,indices,text_labels)
        script_nearestNeigbourExperiment.writeHTML(file_name_l+'.html',im_paths,captions)
        no_correct,_=script_nearestNeigbourExperiment.getNumberOfCorrectNNMatches(indices,gt_labels);
        print no_correct
        with open(file_name_l+'.txt','wb') as f:
            for no_correct_curr in no_correct:
                f.write(str(no_correct_curr)+' ');



        file_name_l=file_name+'_'+layer+'_pascal';
        im_paths,captions=script_nearestNeigbourExperiment.createImageAndCaptionGrid(img_paths_curr,gt_labels_pascal,indices,text_labels_pascal)
        script_nearestNeigbourExperiment.writeHTML(file_name_l+'.html',im_paths,captions)
        no_correct,_=script_nearestNeigbourExperiment.getNumberOfCorrectNNMatches(indices,gt_labels_pascal);
        with open(file_name_l+'.txt','wb') as f:
            for no_correct_curr in no_correct:
                f.write(str(no_correct_curr)+' ');

        print no_correct


def script_visualizePerformanceDifference():
    # trained_file='/disk2/octoberExperiments/nn_performance_without_pascal/trained/20151023153522';
    # notrained_file='/disk2/octoberExperiments/nn_performance_without_pascal/notrained/20151026132705';
    # out_file='/disk2/octoberExperiments/nn_performance_without_pascal/nn_accuracy_pascal_labels.png';
    
    notrained_file='/disk2/novemberExperiments/nn_imagenet_top5/notrained/20151130193757';
    trained_file='/disk2/novemberExperiments/nn_imagenet_top5/trained/20151130230243'
    out_file='/disk2/novemberExperiments/nn_imagenet_top5/nn_accuracy_imagenet_labels.png';

    file_pres=[trained_file,notrained_file]
    layers=['pool5','fc6','fc7'];
    legend_pres=['Trained','Unfamiliar'];
    legend_entries=[];
    for leg in legend_pres:
        legend_entries.extend([leg+' '+layer for layer in layers]);

    vecs_to_plot=[];
    file_names=[file_pre+'_'+layer for file_pre in file_pres for layer in layers];
    for file_name in file_names:
        with open(file_name+'.txt','rb') as f:
            no_correct=f.readline();
        no_correct=[float(no_correct_curr) for no_correct_curr in no_correct.strip('\n').split()];
        vecs_to_plot.append(no_correct);

    print legend_entries;
    print vecs_to_plot;
    print len(vecs_to_plot);
    print len(legend_entries);

    plt.figure();
    plt.xlabel('Number of Nearest Neighbours K');
    plt.ylabel('Accuracy');
    plt.title('NN Accuracy DNN Features for Imagenet Classes'); 
    plt.xlim(0,6);
    plt.ylim(min([min(vec) for vec in vecs_to_plot])-0.05,max([max(vec) for vec in vecs_to_plot])+0.05);
    handles=[];
    for vec in vecs_to_plot:
        handle,=plt.plot(range(1,len(vec)+1),vec);
        handles.append(handle);
        
    plt.legend(handles, legend_entries,loc=2,prop={'size':10});
    
    plt.savefig(out_file);





def script_writeHTMLS():
    path_to_file='../../data/ilsvrc12/synset_words.txt'
    val_ids=imagenet.readLabelsFile(path_to_file);
    val_just_ids=list(zip(*val_ids)[0]);
    val_just_labels=list(zip(*val_ids)[1]);

    pascal_ids_file='/disk2/octoberExperiments/nn_performance_without_pascal/pascal_classes.txt'
    pascal_ids=imagenet.readLabelsFile(pascal_ids_file);
    pascal_just_ids=list(zip(*pascal_ids)[0]);

    val_gt_file='../../data/ilsvrc12/val.txt'
    list_of_ids_im=[id for id_list in to_exclude for id in id_list];
    mapping_file='../../data/ilsvrc12/synsets.txt';

    list_of_ids=imagenet.getImagenetIdToTrainingIdMapping(mapping_file,list_of_ids_im)
    print len(list_of_ids)

    to_exclude=imagenet.removeClassesWithOverlap(val_just_ids,pascal_just_ids,keepMapping=True);
    list_of_ids_pascal=[];

    for id_no in range(len(to_exclude)):
        list_of_ids_pascal=list_of_ids_pascal+[id_no]*len(to_exclude[id_no])
    
    file_text_labels='../../data/ilsvrc12/synset_words.txt'
    text_labels= np.loadtxt(file_text_labels, str, delimiter='\t')

    trained_file='/disk2/octoberExperiments/nn_performance_without_pascal/trained/20151023153522';
    notrained_file='/disk2/octoberExperiments/nn_performance_without_pascal/notrained/20151026132705';

    file_pres=[trained_file,notrained_file]
    layers=['pool5','fc6','fc7'];
    file_names=[file_pre+'_'+layer for file_pre in file_pres for layer in layers];

    for file_name in file_names:
        [img_paths,gt_labels,indices,_]=pickle.load(open(file_name+'.p','rb'));
        gt_labels_pascal=[list_of_ids_pascal[list_of_ids.index(gt_label)] for gt_label in gt_labels];

        file_name_l=file_name+'_'+layer+'_pascal';
        # img_paths_curr=[x.replace(relativePaths[0],relativePaths[1]) for x in img_paths];
        # im_paths,captions=script_nearestNeigbourExperiment.createImageAndCaptionGrid(img_paths_curr,gt_labels,indices,text_labels)
        # script_nearestNeigbourExperiment.writeHTML(file_name_l+'.html',im_paths,captions)
        
        # gt_labels_pascal=[];
        
        # file_name_l=file_name+'_'+layer+'_pascal';
        # im_paths,captions=script_nearestNeigbourExperiment.createImageAndCaptionGrid(img_paths_curr,gt_labels_pascal,indices,text_labels_pascal)
        # script_nearestNeigbourExperiment.writeHTML(file_name_l+'.html',im_paths,captions)
        # no_correct,_=script_nearestNeigbourExperiment.getNumberOfCorrectNNMatches(indices,gt_labels_pascal);
        # with open(file_name_l+'.txt','wb') as f:
        #     for no_correct_curr in no_correct:
        #         f.write(str(no_correct_curr)+' ');

    



if __name__=='__main__':
    main();