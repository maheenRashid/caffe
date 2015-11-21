import imagenet;
import Tkinter;
import numpy as np;
import scipy;
from scipy import misc;
import matplotlib.pyplot as plt
from Tkinter import *
import functools
import cPickle as pickle;
from PIL import Image, ImageTk
import os;
class Window(object):
	def __init__(self,pascal_classes,imagenet_labels,sample_images,mappingDict=None,out_file='temp.p',snapshot_iter=10):
		self.root=Tkinter.Tk();
		self.imagenet_labels=imagenet_labels;
		self.sample_images=sample_images;
		self.imagenet_idx=0;
		self.snapshot_iter=snapshot_iter;
		self.photo=None;
		self.out_file=out_file;
		if mappingDict is None:
			self.mappingDict={};
		else:
			self.mappingDict=mappingDict;

		self.cols=5;
		self.label=Label(self.root,text='label',name='label');
		self.image_panel=Label(self.root,name='image_panel');
		##self.label.pack(side=LEFT);
		self.image_panel.grid(row=0,column=2);
		self.label.grid(row=1,column=2);

		
		self.buttons=[];
		self.initializeButtons(pascal_classes);
		
		# print 'done init'
		self.root.after(0,self.updateLabels);
		self.root.mainloop();

	def initializeButtons(self,pascal_classes):
		self.buttons=[];
		for idx,pascal_class in enumerate(pascal_classes):
			button_curr=Button(self.root,text=pascal_class,name=pascal_class);
			button_curr.bind("<ButtonPress-1>",self.recordPascal)
			self.buttons.append(button_curr);
			mod=idx%self.cols;
			div=idx/self.cols;
			# print mod,div
			button_curr.grid(row=div+2,column=mod);

	def recordPascal(self,event):
		if self.imagenet_idx<len(self.imagenet_labels):
			print self.imagenet_idx
			pascal_class=event.widget._name;
			imagenet_label=self.imagenet_labels[self.imagenet_idx];
			check=event.widget.master.children['label'].cget('text');
			assert imagenet_label==check;
			self.mappingDict[imagenet_label]=pascal_class;
			# print self.mappingDict
			self.imagenet_idx+=1;
			self.updateLabels();

	def updateLabels(self):
		if self.imagenet_idx<len(self.imagenet_labels):
			imagenet_image=self.sample_images[self.imagenet_idx];
			im=Image.open(imagenet_image);

			print im.size
			base_height=227;
			h_percent=base_height/float(im.size[1])
			width=int(float(im.size[0]*h_percent));
			im=im.resize((width,base_height),Image.ANTIALIAS);

			print im.size
			photo=ImageTk.PhotoImage(im);
			self.image_panel['image']=photo;
			self.photo=photo;

			# im=PhotoImage(file=imagenet_image);
			# im.
			imagenet_label=self.imagenet_labels[self.imagenet_idx];
			self.label['text']=imagenet_label
			##self.label.pack(side=LEFT);
			self.image_panel.grid(row=0,column=2);
			self.label.grid(row=1,column=2);
			if self.imagenet_idx % self.snapshot_iter==0:
				pickle.dump(self.mappingDict,open(self.out_file+str(self.imagenet_idx),'wb'))
		else:
			imagenet_label='Finished';
			self.label['text']=imagenet_label
			##self.label.pack(side=LEFT);
			self.label.grid(row=0,column=0);
			pickle.dump(self.mappingDict,open(self.out_file,'wb'));

def writeExcludedClassInfoFile(mappingDict,out_file_text):
	vals=mappingDict.values();
	keys=np.array(mappingDict.keys());

	vals=np.array(vals);
	vals_unique=np.unique(vals);
	vals_unique=np.delete(vals_unique,np.where(vals_unique=='none'));
	with open(out_file_text,'wb') as f:
		for val_unique in vals_unique:
			idx=np.where(vals==val_unique);
			keys_rel=keys[idx];
			f.write(val_unique+'\n');
			for key_rel in keys_rel:
				key_rel=key_rel.replace('\n',' ');
				f.write(key_rel+'\n');
			f.write('___'+'\n')

def writeExcludedIdsFile(mappingDict,out_file):
	vals=mappingDict.values();
	keys=mappingDict.keys();
	to_exclude=[];
	for idx,val in enumerate(vals): 
		if val!='none':
			key_rel=keys[idx];
			key_rel=key_rel[:key_rel.index('\n')]
			to_exclude.append(key_rel);
	print len(to_exclude);
	with open(out_file,'wb') as f:
		for to_exclude_curr in to_exclude:
			f.write(to_exclude_curr+'\n');


def main():

	synset_words='/disk2/octoberExperiments/nn_performance_without_pascal/synset_words.txt'
	pascal_classes=['person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','dining_table','potted_plant','sofa','tv/monitor','none']
	out_dir='/disk2/novemberExperiments/network_no_pascal'
	out_file_to_exclude_text=os.path.join(out_dir,'to_exclude.txt');

	print len(pascal_classes);
	ids_labels=imagenet.readLabelsFile(synset_words);
	print len(ids_labels);
	out_file=os.path.join(out_dir,'mappingImagenetToPascal.p');
	mappingDict=pickle.load(open(out_file,'rb'));
	writeExcludedIdsFile(mappingDict,out_file_to_exclude_text)

	
	# print sum(vals=='none'),len(vals)

	return
	out_file=os.path.join(out_dir,'mappingImagenetToPascal_temp.p');
	out_file_text=os.path.join(out_dir,'excluded_classes_info_temp.txt');

	imagenet_labels=[x[0]+'\n'+x[1] for x in ids_labels];
	sample_images=[os.path.join(out_dir,'sample_images/'+x[0]+'.JPEG') for x in ids_labels]

	imagenet_labels=imagenet_labels[:10];
	sample_images=sample_images[:10];

	window=Window(pascal_classes,imagenet_labels,sample_images,out_file=out_file);

	mappingDict=pickle.load(open(out_file,'rb'));
	writeExcludedClassInfoFile(mappingDict,out_file_text)
	print out_file_text
	

if __name__=='__main__':
	main();