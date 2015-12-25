import numpy as np
import util
# import cudarray as ca
import time
import cProfile
import StringIO
import pstats
# def normalize(matrix,gpuFlag=False):
#     if gpuFlag==True:
#         norm=ca.sqrt(ca.sum(ca.power(matrix,2),1,keepdims=True));
#         matrix_n=matrix/norm
#     else:
#         norm=np.sqrt(np.sum(np.square(matrix),1,keepdims=True));
#         matrix_n=matrix/norm
        
#     return matrix_n

def nn(test,train):
    test=util.normalize(test,gpuFlag=True);
    train=util.normalize(train,gpuFlag=True);
    distances=ca.dot(test,ca.transpose(train));
    # print distances.shape
    distances_np=np.array(distances);
    indices=np.argsort(distances,axis=1)[:,::-1]
    distances=(1-np.sort(distances,axis=1))[:,::-1];        
    return indices,distances

def main():

    shape=(100000,100000);
    file_npz='/disk2/decemberExperiments/gettingNN/npz_'+str(shape[0])+'.npz';
    # file_byte='temp/blob_'+str(shape[0])+'.b';

    arr=np.random.randn(*shape).astype('float32');
    print arr.dtype
    print arr.shape
    np.savez(file_npz,arr);
    # blob_str=bytearray(np.array(arr,dtype='float32').tostring());
    
    # with open(file_byte,'wb') as f:
    #     f.write(blob_str)

    t=time.time();
    arr=np.load(file_npz)['arr_0'];
    print time.time()-t;

    # t=time.time();
    # with open(file_byte,'rb') as f:
    #     blob_str=f.read();
    
    # arr_byte=np.fromstring(blob_str,dtype='float32')
    # arr_byte=arr_byte.reshape(shape)
    # print time.time()-t

    

    # print arr_byte.shape,np.allclose(arr,arr_byte);









    return
    # train_size=(,4096);
    feat_size=4096
    test_size=(200,4096);
    total=6371288
    batch_size=1000

    # test=ca.random.uniform(size=test_size)

    idx=util.getIdxRange(total,batch_size);
    print len(idx)
    sizes=[];
    for idx_idx,start_idx in enumerate(idx[:-1]):

        end_idx=idx[idx_idx+1]
        print idx_idx,start_idx,end_idx
        curr_size=end_idx-start_idx;
        if idx_idx==0:
            train=ca.random.uniform(size=(curr_size,feat_size))
        else:
            # curr_array=ca.random.uniform(size=(curr_size,feat_size))
            curr_array=np.random.randn(curr_size,feat_size)
            curr_array=ca.array(curr_array);
            print curr_array.shape,type(curr_array[0,0])
            train=ca.extra.concatenate(train,curr_array,axis=0)
            print train.shape
        # sizes.append(curr_size);

    # print len(sizes);
    # print sum(sizes);
    print train.shape

    return

    # train=ca.zeros(train_size);
    
    pr = cProfile.Profile()
    pr.enable()
    indices,distances = nn(test,train);
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()    
    print indices.shape,distances.shape

    return
    shape=(10000,4096);
    a = np.random.randn(*shape);

    ga = ca.array(a);
    print a.shape,ga.shape
    
    # ga=ca.concatenate(a,a);

    ga_b=ca.extra.concatenate(ga,ga,axis=0);
    ga_test=ca.random.uniform(size=(100,4096))

    pr = cProfile.Profile()
    pr.enable()
    indices,distances = nn(ga_b,ga_b);
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

    print indices.shape,distances.shape
    # print ga_b.shape
    return
    t=time.time();
    a_n=util.normalize(a);
    print time.time()-t;
    print a_n.shape




    t=time.time();
    ga_n=util.normalize(ga,True);
    print time.time()-t;
    print ga_n.shape

    ga_n_conv = np.array(ga_n)
    print np.allclose(a_n,ga_n_conv)




if __name__=='__main__':
    main();