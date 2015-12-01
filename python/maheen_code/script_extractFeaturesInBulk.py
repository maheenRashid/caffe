import sys;
import os;
import cPickle as pickle;
import script_top5error
import time;

def main(argv):
    print argv
    in_file_meta=argv[1];
    out_file_meta=argv[2];
    print in_file_meta,out_file_meta
    path_to_classify='..';
    ext='jpg'
    layers=['fc7'];
    gpu_no=0;
    out_files_record=[];
    
    deploy_file = '../../models/bvlc_reference_caffenet/deploy.prototxt';
    model_file = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
    mean_file = '../../python/caffe/imagenet/ilsvrc_2012_mean.npy';
    
    # t=time.time();
    in_files=pickle.load(open(in_file_meta,'rb'));
    net,transformer=script_top5error.setUpNet(model_file,deploy_file,mean_file,gpu_no)
    # in_files=in_files[:10];
    print len(in_files)
    args=[];
    for idx_in_file,in_file in enumerate(in_files):
        print idx_in_file+1,len(in_files);
        out_file=in_file[:in_file.rindex('.')];
        t=time.time();
        out_file_curr=script_top5error.saveLayers(net,transformer,in_file,layers,out_file,rewrite=False);
        print time.time()-t
        out_files_record.append(out_file_curr);
    # print out_files_record

    pickle.dump(out_files_record,open(out_file_meta,'wb'));

	# pass;

if __name__=='__main__':
	main(sys.argv)
