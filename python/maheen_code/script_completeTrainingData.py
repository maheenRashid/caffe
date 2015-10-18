import os

path_to_train='/disk2/imagenet/train';
path_to_tar='/run/user/1001/gvfs/archive:host=file%253A%252F%252F%252Fdisk2%252Fimagenet%252FILSVRC2012_img_train.tar';

files_tar=[f for f in os.listdir(path_to_tar) if f.endswith('.tar')]
x=0;

for f in files_tar:
	just_id=f[:-4];
	out_file=os.path.join(path_to_train,just_id)
	if not os.path.exists(out_file):
		command='tar -xf '+os.path.join(path_to_tar,f)+' -C '+out_file
		# print command
		x+=1;
		# break;
		os.mkdir(os.path.join(path_to_train,just_id))
		os.system(command);

print x