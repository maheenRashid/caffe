
import os;
import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)));
import caffe
import imagenet
import numpy as np;
import cPickle as pickle;
import util
from imagenet_db import Imagenet, Imagenet_Manipulator

def getTopNError(net,transformer,im_files,imagenet_idx_mapped,batch_size,top_n,printDebug=True,pred_classes=False):
    
    im_files=im_files;
    num_files=len(im_files);
    idx_range=util.getIdxRange(num_files,batch_size);
    
    error_bin=np.zeros((num_files,),dtype='int');
    
    if pred_classes:
        pred_classes_mat=-1*np.ones((num_files,top_n),dtype='int');

    for idx_idx,idx_begin in enumerate(idx_range[:-1]):
        idx_end=idx_range[idx_idx+1];

        im_files_curr=im_files[idx_begin:idx_end];
        gt_class_curr=imagenet_idx_mapped[idx_begin:idx_end];
        batch_size_curr=len(im_files_curr);

        net.blobs['data'].reshape(batch_size_curr,3,227,227)

        for idx_im in range(batch_size_curr):
            net.blobs['data'].data[idx_im,:,:,:] = transformer.preprocess('data', caffe.io.load_image(im_files_curr[idx_im]))
        out = net.forward()

        for idx_im in range(net.blobs['prob'].data.shape[0]):
            top_k=net.blobs['prob'].data[idx_im].flatten().argsort()[-1:-(top_n+1):-1]
            gt_class=gt_class_curr[idx_im];
            
            if pred_classes:
                pred_classes_mat[idx_im+idx_begin,:]=top_k;

            if sum(top_k==gt_class)>0:
                error_bin[idx_im+idx_begin]=1;

        if printDebug:      
            print idx_begin,idx_end,batch_size_curr

    if printDebug:
        print sum(error_bin),len(error_bin);
    
    if pred_classes:
        return error_bin,pred_classes_mat
    else:
        return error_bin

def setUpNet(model_file,deploy_file,mean_file,gpu_no):
    caffe.set_device(gpu_no)
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_file,model_file,caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return net,transformer

def main():
    
    # in_file='/disk2/temp/small_set.txt';
    path_to_val='/disk2/imagenet/val'
    
    # val_gt_file='../../data/ilsvrc12/val.txt'
    # synset_words='../../data/ilsvrc12/synset_words.txt'
    # deploy_file='../../models/bvlc_reference_caffenet/deploy.prototxt';
    # model_file='../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
    # mean_file='../../python/caffe/imagenet/ilsvrc_2012_mean.npy';
    # out_file_topk='/disk2/octoberExperiments/nn_performance_without_pascal/val_performance_top5_trained.p';

    # val_gt_file='/disk2/octoberExperiments/nn_performance_without_pascal/val.txt'
    synset_words='/disk2/octoberExperiments/nn_performance_without_pascal/synset_words.txt'
    deploy_file='/disk2/octoberExperiments/nn_performance_without_pascal/deploy.prototxt';
    model_file='/disk2/octoberExperiments/nn_performance_without_pascal/snapshot_iter_450000.caffemodel';
    mean_file='/disk2/octoberExperiments/nn_performance_without_pascal/mean.npy';
    # out_file_topk='/disk2/octoberExperiments/nn_performance_without_pascal/val_performance_top5_no_trained.p';

    db_path='sqlite://///disk2/novemberExperiments/nn_imagenet/nn_imagenet.db';
    
    all_pascal_id=['boat', 'train', 'bicycle', 'chair', 'motorbike', 'aeroplane', 'sofa', 'diningtable', 'bottle', 'tvmonitor', 'bus', 'car'];


    batch_size=1000;
    top_n=1;
    gpu_no=1;
    getClasses=True;

    ids_labels=imagenet.readLabelsFile(synset_words);
    labels=list(zip(*ids_labels)[1]);
    imagenet_ids=list(zip(*ids_labels)[0]);

    net,transformer=setUpNet(model_file,deploy_file,mean_file,gpu_no)

    for pascal_id in ['car']:
    # all_pascal_id:
        print pascal_id
        out_file_topk='/disk2/novemberExperiments/nn_imagenet/'+pascal_id+'_pred_no_train.p';
        out_file_topk_txt='/disk2/novemberExperiments/nn_imagenet/'+pascal_id+'_pred_no_train.txt';

        mani=Imagenet_Manipulator(db_path);
        mani.openSession();
        criterion=(Imagenet.class_id_pascal==pascal_id,);
        vals=mani.select((Imagenet.img_path,),criterion,distinct=True);

        mani.closeSession();
        im_files=[val[0] for val in vals];
        imagenet_idx_mapped=[772]*len(im_files);
        print len(im_files);

        # return
        # error_bin=pickle.load(open(out_file_topk,'rb'));
        # print sum(error_bin),len(error_bin),sum(error_bin)/float(len(error_bin))

        # out_file_topk='/disk2/octoberExperiments/nn_performance_without_pascal/val_performance_top5_trained.p';
        # error_bin=pickle.load(open(out_file_topk,'rb'));
        # print sum(error_bin),len(error_bin),sum(error_bin)/float(len(error_bin))

        # return
        
        # val_gt=imagenet.readLabelsFile(val_gt_file);
        # im_files=list(zip(*val_gt)[0]);
        # im_files=[os.path.join(path_to_val,im_file) for im_file in im_files]
        # imagenet_idx_mapped=list(zip(*val_gt)[1])
        # imagenet_idx_mapped=[int(x) for x in imagenet_idx_mapped];
        

        # print idx_mapped_simple[:10],len(idx_mapped_simple);
        # print 'getting mapping'
        
        # imagenet_idx_mapped,imagenet_ids_mapped,imagenet_labels_mapped=imagenet.getMappingInfo(im_files,synset_words,val_gt_file)
        # print imagenet_idx_mapped[:10],len(imagenet_idx_mapped),type(imagenet_idx_mapped),type(imagenet_idx_mapped[0]),type(idx_mapped_simple),type(idx_mapped_simple[0])
        
        # for idx_idx,idx in enumerate(idx_mapped_simple):
        #   print idx,imagenet_idx_mapped[idx_idx]
        #   assert idx==imagenet_idx_mapped[idx_idx]
        # return

        
        error_bin,pred_classes=getTopNError(net,transformer,im_files,imagenet_idx_mapped,batch_size,top_n,printDebug=True,pred_classes=True)
        print sum(error_bin),len(error_bin)
        print pred_classes[:10]

        pickle.dump([error_bin,pred_classes],open(out_file_topk,'wb'));

        pred_classes=pred_classes.ravel()
        tuples=[];
        for class_curr in np.unique(pred_classes):
            # print imagenet_ids[class_curr],labels[class_curr];
            tuples.append((class_curr,labels[class_curr],sum(pred_classes==class_curr)/float(len(pred_classes))));
            
        tuples=sorted(tuples,key=lambda x: x[2])[::-1];
        with open(out_file_topk_txt,'wb') as f:
            for tuple_curr in tuples:
                print imagenet_ids[tuple_curr[0]],tuple_curr
                f.write(str(tuple_curr[0])+'\t'+str(tuple_curr[1])+'\t'+str(round(tuple_curr[2]*100,2)));
                f.write('\n');
        # for tuple_curr in tuples:
        #     print tuple_curr



    


if __name__=='__main__':
    main();