import os;
import util;

def main():
	path_meta='/disk2/res11/tubePatches';
	out_commands='/disk2/res11/commands_deleteAllImages.txt';
	dirs=[os.path.join(path_meta,dir_curr) for dir_curr in os.listdir(path_meta) if os.path.isdir(os.path.join(path_meta,dir_curr))];
	print len(dirs);
	commands=[];
	for dir_curr in dirs:
		dirs_in=[os.path.join(dir_curr,dir_in) for dir_in in os.listdir(dir_curr) if os.path.isdir(os.path.join(dir_curr,dir_in))];
		commands.extend(['rm -v '+dir_in+'/*.jpg' for dir_in in dirs_in]);
	print len(commands);
	print commands[:10];
	util.writeFile(out_commands,commands);



if __name__=='__main__':
	main();