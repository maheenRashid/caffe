FOR script_saveCommands  script_unpack_video

	out_dir_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data/UCF-101';
	in_dir_meta='/disk2/video_data/UCF-101';
	out_file_text='/disk2/video_data/unpack_UCF-101_commands.txt';
	script_saveCommands(in_dir_meta,out_dir_meta,out_file_text)
	
	out_dir_meta='/media/maheenrashid/e5507fe3-2bff-4cbe-bc63-400de6deba92/maheen_data/image_data/hmdb';
	in_dir_meta='/disk2/video_data/hmdb';
	out_file_text='/disk2/video_data/unpack_hmdb_commands.txt';
	script_saveCommands(in_dir_meta,out_dir_meta,out_file_text)