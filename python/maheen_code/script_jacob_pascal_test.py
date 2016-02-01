import os;
import random;
import visualize;

def main():
	out_dir='/disk2/januaryExperiments/jacob_pascal_2007';
	img_folder=os.path.join(out_dir,'im_results');
	res_folder=os.path.join(out_dir,'results');
	out_file_html=os.path.join(out_dir,'jacob_comparison.html');
	width_heigh=[400,400];
	rel_path=['/disk2','../../../..']
	rel_path_res=['/disk2','../../..']
	path_to_im='/disk2/pascal_voc_2007/VOCdevkit/VOC2007/JPEGImages/';
	im_list=[im_curr for im_curr in os.listdir(img_folder) if im_curr.endswith('.jpg')];
	# im_just_name=[im_curr[:im_curr.index('.')] for im_curr in im_list];
	# text_files=[os.path.join(res_folder,im_curr+'.txt') for im_curr in im_just_name]
	img_corrs=[];
	img_arrows=[];
	for im_curr in im_list:
		im_jn=im_curr[:im_curr.index('.')];
		im_txt=os.path.join(res_folder,im_jn+'.txt');
		with open(im_txt,'rb') as f:
			img_corr=f.readline();
			img_corr=img_corr[:img_corr.index(' ')];
		img_corrs.append(img_corr);
		img_arrows.append(os.path.join(img_folder,im_curr));
		# print img_corrs,img_arrow
		# raw_input();

	img_paths=[];
	captions=[];
	for img_org,img_arrow in zip(img_corrs,img_arrows):
		img_paths.append([img_org.replace(rel_path[0],rel_path[1]),img_arrow.replace(rel_path_res[0],rel_path_res[1])])
		captions.append([img_org,img_arrow]);

	visualize.writeHTML(out_file_html,img_paths,captions,400,400)
	return
	path_to_im='/disk2/pascal_voc_2007/VOCdevkit/VOC2007/JPEGImages/';
	num_im=1000;
	im_list=[os.path.join(path_to_im,im_curr) for im_curr in os.listdir(path_to_im) if im_curr.endswith('.jpg')];
	
	idx=range(len(im_list));
	random.shuffle(idx);
	idx=idx[:num_im];
	im_list=[im_list[idx_curr] for idx_curr in idx];
	
	out_dir='/disk2/januaryExperiments/jacob_pascal_2007';
	if not os.path.exists(out_dir):
		os.mkdir(out_dir);

	jacob_dir='/home/maheenrashid/Downloads/jacob/try_2/optical_flow_prediction/examples/opticalflow';
	txt_file=os.path.join(jacob_dir,'test.txt');
	res_folder=os.path.join(out_dir,'results');
	if not os.path.exists(res_folder):
		os.mkdir(res_folder);

	with open(txt_file,'wb') as f:
		for im in im_list:
			f.write(im+' 1\n');





if __name__=='__main__':
	main();