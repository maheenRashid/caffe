import sys
import os;
import time;
import numpy as np;
from tube_db import Tube, Tube_Manipulator,TubeHash_Manipulator,TubeHash


def main(argv):
    path_to_db=argv[1]
    out_dir=argv[2];
    hash_table=int(argv[3]);
    hash_val=int(argv[4]);

    mani_hash=TubeHash_Manipulator(path_to_db);
    mani_hash.openSession();

    out_file=os.path.join(out_dir,str(hash_table)+'_'+str(hash_val)+'.npz');
    print out_file
    
    if not os.path.exists(out_file):
        
	    t=time.time();
	    toSelect=(TubeHash.idx,Tube.class_idx_pascal,Tube.video_id,Tube.shot_id,Tube.frame_id);
	    criterion=(TubeHash.hash_table==hash_table,TubeHash.hash_val==hash_val);
	    vals=mani_hash.selectMix(toSelect,criterion);

	    vals=np.array(vals);
	    print vals.shape;
	    
	    np.savez(out_file,vals);
	    print time.time()-t

    mani_hash.closeSession();

if __name__=='__main__':
	main(sys.argv);
    