
import os;
import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)));
import caffe
import imagenet
import numpy as np;
import cPickle as pickle;
def main():
	
	in_file='/disk2/temp/small_set.txt';
	path_to_val='/disk2/imagenet/val'
	
	val_gt_file='../../data/ilsvrc12/val.txt'
	synset_words='../../data/ilsvrc12/synset_words.txt'
	deploy_file='../../models/bvlc_reference_caffenet/deploy.prototxt';
	model_file='../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
	mean_file='../../python/caffe/imagenet/ilsvrc_2012_mean.npy';
	out_file_topk='/disk2/octoberExperiments/nn_performance_without_pascal/val_performance_top5_trained.p';

	# val_gt_file='/disk2/octoberExperiments/nn_performance_without_pascal/val.txt'
	# synset_words='/disk2/octoberExperiments/nn_performance_without_pascal/synset_words.txt'
	# deploy_file='/disk2/octoberExperiments/nn_performance_without_pascal/deploy.prototxt';
	# model_file='/disk2/octoberExperiments/nn_performance_without_pascal/snapshot_iter_450000.caffemodel';
	# mean_file='/disk2/octoberExperiments/nn_performance_without_pascal/mean.npy';
	# out_file_topk='/disk2/octoberExperiments/nn_performance_without_pascal/val_performance_top5.p';


	batch_size=1000;
	top_n=5;
	gpu_no=1;
	
	val_gt=imagenet.readLabelsFile(val_gt_file);
	im_files=list(zip(*val_gt)[0]);
	im_files=[os.path.join(path_to_val,im_file) for im_file in im_files]
	imagenet_idx_mapped=list(zip(*val_gt)[1])
	imagenet_idx_mapped=[int(x) for x in imagenet_idx_mapped];

	# print idx_mapped_simple[:10],len(idx_mapped_simple);
	# print 'getting mapping'
	
	# imagenet_idx_mapped,imagenet_ids_mapped,imagenet_labels_mapped=imagenet.getMappingInfo(im_files,synset_words,val_gt_file)
	# print imagenet_idx_mapped[:10],len(imagenet_idx_mapped),type(imagenet_idx_mapped),type(imagenet_idx_mapped[0]),type(idx_mapped_simple),type(idx_mapped_simple[0])
	
	# for idx_idx,idx in enumerate(idx_mapped_simple):
	# 	print idx,imagenet_idx_mapped[idx_idx]
	# 	assert idx==imagenet_idx_mapped[idx_idx]
	# return


	caffe.set_device(gpu_no)
	caffe.set_mode_gpu()
	net = caffe.Net(deploy_file,model_file,caffe.TEST)

	# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

	

	im_files=im_files;
	num_files=len(im_files);
	idx_range=range(0,num_files+1,batch_size);
	print num_files,idx_range[-1]
	if idx_range[-1]!=num_files:
		idx_range.append(num_files);
	# print idx_range

	error_bin=np.zeros((num_files,),dtype='int');
	control_case=[];

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
			
			if sum(top_k==gt_class)>0:
				error_bin[idx_im+idx_begin]=1;
			control_case.append(sum(top_k==gt_class))
			
		print idx_begin,idx_end,batch_size_curr

	for idx in range(len(error_bin)):
		assert error_bin[idx]==control_case[idx];

	print sum(error_bin),len(error_bin),sum(control_case),len(control_case);

	pickle.dump(error_bin,open(out_file_topk,'wb'));


	


if __name__=='__main__':
	main();