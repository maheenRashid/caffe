import os;
import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)));
import subprocess


def saveFeaturesOfLayers(in_file,path_to_classify,gpu_no,layers,ext='JPEG',central_crop=True,meanFile=None,deployFile=None,modelFile=None,out_file=None, images_dim=None):
    
    if out_file is None:
        out_file= time.strftime("%Y%m%d%H%M%S");
    
    if ext is None:
        ext=str(None);

    command=[os.path.join(path_to_classify,'classify.py'),in_file,out_file,'--ext',ext,'--gpu',str(gpu_no),'--layer']+layers;

    if central_crop:
        command=command+["--center_only"]

    if meanFile is not None:
        command=command+["--mean_file",meanFile];
        
    if deployFile is not None:
        command=command+["--model_def",deployFile];
    
    if modelFile is not None:
        command=command+["--pretrained_model",modelFile];

    if images_dim is not None:
       command=command+["--images_dim",str(images_dim[0])+','+str(images_dim[1])];


    command_formatted=' '.join(command);
    success=subprocess.call(command_formatted, shell=True)
    print command_formatted
    
    return out_file+'.npz';
