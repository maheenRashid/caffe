# from treelib import Node, Tree
import re
import numpy as np
import random
# def pickNoOverlapClasses(numClasses,parentChild):
import urllib2
import pickle
import os
import urllib

def getCounts(total_ids_str,per_class_urls):
	total_ids_count=[];
	for id in total_ids_str:
		url_curr=per_class_urls+id;
		f=urllib2.urlopen(url_curr);
		number_images=len(f.readlines());
		total_ids_count.append(number_images);
	return np.array(total_ids_count)



def main():

	val_data_ids='/home/maheenrashid/Downloads/caffe/caffe-rc2/data/ilsvrc12/synsets.txt';
	is_file='/disk2/imagenet/structure_files/is_a.txt';
	out_dir='/disk2/imagenet/not_trained_im_tars'

	list_of_nets='http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list';
	per_class_urls='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=';
	# f = urllib2.urlopen(list_of_nets)
	# total_ids = f.readlines()
	# total_ids=[id.strip('\n') for id in total_ids]

	# counts=[];
	# for idx,id in enumerate(total_ids):
	# 	url_curr=per_class_urls+id;
	# 	f=urllib2.urlopen(url_curr);
	# 	number_images=len(f.readlines());
	# 	counts.append(number_images);
	# 	if idx==10:
	# 		break;

	# print total_ids[:10];
	# print counts[:10];
	# # with open(list_of_nets,'rb') as f:
	# # 	nets_list=f.readlines();

	# # print nets_list[:10];

	# return


	with open(is_file,'rb') as f:
		content=f.readlines();

	parents=[];
	children=[];
	# total_ids_str=[];
	for c in content:
		c_split=c.split(' ');
		
		# total_ids_str.append(c_split[0]);
		# total_ids_str.append(c_split[1].strip('\n'));

		parents.append(int(c_split[0].strip('n')));
		children.append(int(c_split[1].strip('\n').strip('n')));

	couples=np.array([parents,children]);

	[total_ids_count,total_ids_str,total_ids]=pickle.load(open('number_images.p','rb'));
	
	total_ids_count=np.array(total_ids_count)
	idx_to_keep=np.where(total_ids_count>=50)[0];
	
	total_ids=total_ids[idx_to_keep];
	total_ids_str=total_ids_str[idx_to_keep];
	total_ids_count=total_ids_count[idx_to_keep];
	print total_ids.shape,total_ids_str.shape,total_ids_count.shape
	print total_ids[:10]
	print total_ids_str[:10]
	print total_ids_count[:10];

	# return
	# total_ids=np.unique(couples);
	# total_ids_str=np.unique(total_ids_str);

	# # total_ids_str=total_ids_str[:100]
	# print len(total_ids_str)
	# total_ids_count=[];
	# for idx,id in enumerate(total_ids_str):
	# 	if idx%100==0:
	# 		print idx,len(total_ids_str);
	# 	url_curr=per_class_urls+id;
	# 	f=urllib2.urlopen(url_curr);
	# 	number_images=len(f.readlines());
	# 	total_ids_count.append(number_images);

	# pickle.dump([total_ids_count,total_ids_str,total_ids],open('number_images.p','wb'));

	# return

	# print len(total_ids_str)
	# total_ids_str=total_ids_str[np.array(total_ids_count)>=50];
	# print len(total_ids_str);

	
	# return
	# with open(val_data_ids,'rb') as f:
	# 	val_ids=f.readlines()

	# val_ids=[int(val.strip('\n').strip('n')) for val in val_ids];
	# val_ids=np.array(val_ids);

	# no_val=np.setdiff1d(total_ids,val_ids);
	
	# noClasses=100;
	# select_ids_meta=[];
	# for iter in range(5):
	# 	while (True):
	# 		random_idx=random.sample(range(no_val.shape[0]),noClasses)

	# 		select_ids=no_val[random_idx];
	# 		par_int=np.in1d(couples[0],select_ids);
	# 		child_int=np.in1d(couples[1],select_ids);
			
	# 		both_int=par_int+child_int==2;
			
	# 		select_ids_str=total_ids_str[[np.where(total_ids==select_id)[0] for select_id in select_ids]]
	# 		select_ids_str=select_ids_str.ravel();
	# 		print select_ids_str.shape,select_ids_str[:10]
	# 		counts=total_ids_count[[np.where(total_ids==select_id)[0] for select_id in select_ids]]
	# 		counts=np.ravel(counts);
	# 		# counts=getCounts(select_ids_str,per_class_urls);
	# 		print sum(counts<50);


	# 		if both_int.sum()==0 and sum(counts<50)==0:
	# 			break;

	# 	select_ids_meta.append(select_ids_str)

	# pickle.dump(select_ids_meta,open('select_ids_meta.p','wb'));
	# return
	select_ids_meta=pickle.load(open('select_ids_meta.p','rb'));
	if not os.path.exists(out_dir):
		os.mkdir(out_dir);

	pre_path='http://www.image-net.org/download/synset?wnid='
	post_path='&username=maheenr&accesskey=a4e22b31277877bf1089983e0b79ce4464fe1c91&release=latest&src=stanford'
	for idx_bunch,select_ids_str in enumerate(select_ids_meta):
		print idx_bunch
		for select_id in select_ids_str:
			print select_id,
			out_file_curr=os.path.join(out_dir,select_id+'.tar');
			print out_file_curr
			image_link=pre_path+select_id+post_path;
			urllib.urlretrieve(image_link,out_file_curr);



			# curr_url=per_class_urls+select_id;
			# f = urllib2.urlopen(curr_url)
			# image_links = f.readlines()
			# image_links = [id.strip('\n').strip('\r') for id in image_links]
			# f.close();
			# print len(image_links);				
			# random.shuffle(image_links);
			# out_file_pre=select_id+'_';
			# saved=0;
			# for idx,image_link in enumerate(image_links):
			# 	out_file_curr=os.path.join(out_dir,out_file_pre+str(idx)+'.JPEG');
			# 	try:
			# 		urllib.urlretrieve(image_link,out_file_curr);
			# 	except:
			# 		continue;
			# 	saved+=1;
			# 	if saved==50:
			# 		break;

			


	# for select_id in select_ids:
	# 	pass;
if __name__=='__main__':
	main();