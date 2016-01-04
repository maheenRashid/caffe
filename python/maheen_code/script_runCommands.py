import os;
import subprocess
import time;

def main():
	path_to_commands='/disk2/decemberExperiments/hash_tables/commands_list.txt'
	commands=[];
	with open(path_to_commands,'rb') as f:
		commands=f.readlines();



	for idx in range(len(commands)):
		command=commands[idx].strip('\n');
		out_file=command.split(' ');
		out_file=os.path.join(out_file[3],out_file[4]+'_'+out_file[5]+'.npz');
		
		if os.path.exists(out_file):
			continue;
		
		command='nohup '+commands[idx].strip('\n')+' 2>/dev/null'
		print command
		subprocess.Popen(command, shell=True)

		if idx%8==0:
			time.sleep(20);


	# print len(commands);
	# print commands[0];

if __name__=='__main__':
	main();