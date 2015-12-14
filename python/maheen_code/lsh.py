import numpy as np;
from bitarray import bitarray;

class HyperplaneHash(object):
    def __init__(self,dimensions=None,hasher_file=None,key_type=np.uint8):
        
        self.key_type = key_type
        self.key_size = 8*np.dtype(self.key_type).itemsize;
        
        if hasher_file is not None:
            self.hasher=np.load(hasher_file);
            if self.hasher.shape[1]%self.key_size!=0:
                dims=str(self.shape[1]);
                message=dims+' columns of hasher invalid for '+str(self.key_type)+' hash keys. Must be divisible by '+str(self.key_size);
                raise Exception( message);

            self.feature_dim=self.hasher.shape[0];
            self.num_hash_tables=self.hasher.shape[1]/self.key_size;

        else:
            self.feature_dim=dimensions[0];
            self.num_hash_tables=dimensions[1];
            self.hasher=self.generateHasher();

    def generateHasher(self):
        hasher = np.random.randn(self.feature_dim,self.num_hash_tables*self.key_size);
        return hasher
    
    def hash(self,features_to_hash):
        if features_to_hash.shape[1] != self.feature_dim or len(features_to_hash.shape)!=2:
            dims=str(features_to_hash.shape);
            message=dims+' input shape is not valid';
            raise Exception( message);
        
        hashed_vals = np.dot(features_to_hash,self.hasher)
        hashed_vals=hashed_vals>0

        num_frames=features_to_hash.shape[0];
        hash_keys=np.zeros((num_frames,self.num_hash_tables),dtype=self.key_type);
        for idx in range(num_frames):
            test=hashed_vals[idx].tolist()
            test_bit=bitarray(test);
            hash_keys[idx,:]=np.fromstring(test_bit.tobytes(), dtype=self.key_type)

        return hash_keys



