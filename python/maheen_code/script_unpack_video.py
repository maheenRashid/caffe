import os;
import util;

def script_saveCommands(in_dir_meta,out_dir_meta,out_file_text):
	cats=[dir_curr for dir_curr in os.listdir(in_dir_meta) if os.path.isdir(os.path.join(in_dir_meta,dir_curr))];
	# print len(cats);
	commands=[];
	command_text='ffmpeg -i '
	for cat in cats:
		dir_curr=os.path.join(in_dir_meta,cat);
		dir_out_curr=os.path.join(out_dir_meta,cat);
		print dir_out_curr;
		
		if not os.path.exists(dir_out_curr):
			os.mkdir(dir_out_curr);
		videos=[file_curr for file_curr in os.listdir(dir_curr) if file_curr.endswith('.avi')];
		for video in videos:
			out_dir_video_org=os.path.join(dir_out_curr,video[:-4])
			out_dir_video = util.escapeString(out_dir_video_org)
			command_curr='mkdir '+out_dir_video+';'+command_text+util.escapeString(os.path.join(dir_curr,video))+' '+out_dir_video+'/image_%d.ppm'
			commands.append(command_curr);
	with open(out_file_text,'wb') as f:
		for command in commands:
			f.write(command+'\n');



def main():
	video_list_file='/disk2/video_data/video_list.txt'
	path_to_im_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data';
	path_to_video_meta='/disk2/video_data';

	video_list=util.readLinesFromFile(video_list_file);
	print len(video_list);
	image_dirs=[video_curr.replace(path_to_video_meta,path_to_im_meta)[:-4] for video_curr in video_list];
	print len(image_dirs),image_dirs[0];
	# for image_dir in image_dirs:
		# print image_dir,os.path.exists(image_dir);
		# assert os.path.exists(image_dir);

	# cats=[dir_curr for dir_curr in os.listdir(out_dir_meta) if os.path.isdir(os.path.join(out_dir_meta,dir_curr))];

	# im_counts=[];
	# video_dirs=[];

	# for cat in cats:
	# 	dir_cat=os.path.join(out_dir_meta,cat);
	# 	videos=[dir_curr for dir_curr in os.listdir(dir_cat) if os.path.isdir(os.path.join(dir_cat,dir_curr))];
	# 	im_counts_curr=[len(os.listdir(os.path.join(dir_cat,video_curr))) for video_curr in videos]
	# 	video_dirs_curr=[os.path.join(dir_cat,video_curr) for video_curr in videos];

	# 	im_counts.extend(im_counts_curr);
	# 	video_dirs.extend(video_dirs_curr);

	# print len(im_counts);
	# print len(video_dirs);
	# print im_counts[:10];

	# return

if __name__=='__main__':
	main();