import re
import numpy as np
import random
import urllib2
import pickle
import os
import urllib


IMAGE_URLS='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=';
IS_A='http://www.image-net.org/archive/wordnet.is_a.txt';
SUBSYNSET=['http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=','&full=1'];

def selectTestIdsNotTrained(ids,image_counts,noClasses,no_im=0,val_ids=None,parent_child=None):
	if parent_child is None:
		parent_child=getIsARelationships();		

	image_counts=np.array(image_counts)
	idx_to_keep=np.where(image_counts>=no_im)[0];
	
	ids=ids[idx_to_keep];
	image_counts=image_counts[idx_to_keep];

	if val_ids is not None:
		no_val=np.setdiff1d(total_ids,val_ids);
	else:
		no_val=ids;
	
	max_iter=1000;
	iter_count=0;
	while (True):
		random_idx=random.sample(range(no_val.shape[0]),noClasses)
		select_ids=no_val[random_idx];
		par_int=np.in1d(couples[0],select_ids);
		child_int=np.in1d(couples[1],select_ids);
		both_int=par_int+child_int==2;
		counts=total_ids_count[[np.where(total_ids==select_id)[0] for select_id in select_ids]]
		counts=np.ravel(counts);
		if both_int.sum()==0 and sum(counts<50)==0:
			break;
		itercount+=1;
		if itercount>=max_iter:
			exception_str='Could not find '+str(noClasses)+' non-overlapping non-trained classes';
			raise Exception(exception_str);

	return select_ids
	
def selectTestIdsTrained():
	pass;

def addToLabelsFile():
	pass;

def getIsARelationships():
	#is_a relationships from imagenet
	#2xn np array with parent ids as ints on top row, 
	#and their children at the bottom

	with urllib2.urlopen(IS_A) as f:
		content=f.readlines();

	parents=[];
	children=[];
	for c in content:
		c_split=c.split(' ');
		parents.append(int(c_split[0].strip('n')));
		children.append(int(c_split[1].strip('\n').strip('n')));
	couples=np.array([parents,children]);
	return couples

def getNumberOfImages(total_ids):
	#get the number of images in the synset

	counts=[];
	for id in enumerate(total_ids):
		url_curr=IMAGE_URLS+id;
		with urllib2.urlopen(url_curr) as f:
			number_images=len(f.readlines());
		counts.append(number_images);
	return counts;

def readLabelsFile(path_to_file):
	text_labels= np.loadtxt(path_to_file, str, delimiter='\t')
	ids=[];
	labels=[];
	for r,text_label in enumerate(text_labels):
		label_idx=text_label.index(' ');
		ids.append(text_label[:label_idx]);
		labels.append(text_label[label_idx+1:].strip('\n'));
	text_labels=zip(ids,labels);
	return text_labels

def getSubSynset(id):
	url_subtree=SUBSYNSET[0]+id+SUBSYNSET[1];
	f = urllib2.urlopen(url_subtree)
	total_ids = f.readlines()
	total_ids=[id.strip('-').strip('\n').strip('\r') for id in total_ids];
	return total_ids

def makeIdIntoInt(ids):
	ids_int=[];
	for id in ids:
		ids_int.append(int(id.strip('n')));
	return ids_int

def removeClassesWithOverlap(ids,ids_to_remove,keepMapping=False):
	val_just_ids_int=np.array(makeIdIntoInt(ids));

	to_exclude=[];
	for id in ids_to_remove:
		to_exclude_curr=getSubSynset(id);
		to_exclude_curr=np.array(makeIdIntoInt(to_exclude_curr));
		indices=np.where(np.in1d(val_just_ids_int,to_exclude_curr))[0];
		ids_to_exclude=[ids[idx] for idx in indices];
		if keepMapping:
			to_exclude.append(ids_to_exclude);
		else:
			to_exclude.extend(ids_to_exclude);

	return to_exclude;

def getImagenetIdToTrainingIdMapping(mapping_file,list_of_ids_imagenet):
	with open(mapping_file,'rb') as f:
		mappings=f.readlines();
	mappings=[mapping.strip('\n').strip('\r') for mapping in mappings];
	
	class_ids=[mappings.index(class_id) for class_id in list_of_ids_imagenet];

	return class_ids
	
def removeImagesFromListByClass(im_list_file,mapping_file,classes_to_remove):
	ims,class_ids=zip(*readLabelsFile(im_list_file));
	class_ids=np.array([int(class_id) for class_id in class_ids]);
	class_id_to_remove=getImagenetIdToTrainingIdMapping(mapping_file,classes_to_remove);
	idx_to_keep=np.where(np.in1d(class_ids,class_id_to_remove)==0)[0];
	ims_to_keep=[ims[idx] for idx in idx_to_keep];
	classes_to_keep=[mappings[class_ids[idx]] for idx in idx_to_keep];
	class_ids_to_keep=class_ids[idx_to_keep]

	return ims_to_keep,class_ids_to_keep,classes_to_keep

def writeNewDataClassFile(new_file_val,im_and_class,shuffle=True):
	classes_to_keep=zip(*im_and_class)[1];
	classes_uni=list(set(classes_to_keep));
	classes_uni.sort();
	strs_to_write=[];
	for im_curr,class_curr in im_and_class:
		str_to_write=im_curr+' '+str(classes_uni.index(class_curr))+'\n';
		strs_to_write.append(str_to_write);

	if shuffle:
		random.shuffle(strs_to_write);

	with open(new_file_val,'wb') as f:
		for str_to_write in strs_to_write:
			f.write(str_to_write);
	return classes_uni

def selectTestSetByID(val_gt_file,list_of_ids,path_to_val=None,random=False,max_num=None):
    im_list,gt_classes=zip(*readLabelsFile(val_gt_file));
    gt_classes=makeIdIntoInt(list(gt_classes));
    
    gt_classes=np.array(gt_classes);
    uni_classes=np.unique(gt_classes);
    idx_chosen_images=[];

    for select_class in list_of_ids:
        idx_rand_class=np.where(gt_classes==select_class)[0];
        
        if max_num is None:
        	no_im=len(idx_rand_class);
        else:
        	no_im=min(max_num,len(idx_rand_class));

        if random:
			idx_chosen=random.sample(idx_rand_class,no_im);        	
        else:
        	idx_chosen=idx_rand_class[:no_im];

        idx_chosen_images.extend(idx_chosen);
        
    im_list_chosen=[ os.path.join(path_to_val,im_list[idx_curr]) if path_to_val is not None else im_list[idx_curr] for idx_curr in idx_chosen_images ]

    gt_class_chosen=gt_classes[idx_chosen_images];

    return zip(im_list_chosen,gt_class_chosen);

def main():
	print 'hello. running imagenet'

if __name__=='__main__':
	main();