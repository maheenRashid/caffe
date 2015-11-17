import os
import sys
from sqlalchemy import Column, Float, Integer, String,Boolean
from sqlalchemy.dialects.mysql import BLOB as Blob
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine,and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
import sqlalchemy

import numpy as np;

Base = declarative_base()

class Imagenet(Base):
    __tablename__ = 'imagenet'
    # id = Column(Integer, primary_key=True,autoincrement=True)
    idx = Column(Integer,nullable=False,primary_key=True);
    img_path = Column(String,nullable=False,primary_key=True);
    layer = Column(String,nullable=False,primary_key=True);
    nn_experiment_timestamp = Column(String,nullable=False,primary_key=True);
    trainedClass=Column(Boolean,nullable=False,primary_key=True);

    class_id_imagenet = Column(String,nullable=False);
    class_label_imagenet = Column(String);
    class_id_pascal = Column(String);
    class_idx_imagenet =  Column(Integer,nullable=False);
    class_idx_pascal =  Column(Integer);
    caffe_model=Column(String,nullable=False);

    neighbor_index=Column(Blob);
    neighbor_distance=Column(Blob);
    
class Imagenet_Manipulator(object):

    def __init__(self,path_to_db):
        self.db_path=path_to_db;
        self.engine=create_engine(self.db_path);
        if not os.path.exists(path_to_db):
            Base.metadata.create_all(self.engine);    
        # print self.engine
        Base.metadata.bind=self.engine;
        self.session=None;

    def openSession(self):
        DBSession=sessionmaker(bind=self.engine);
        self.session=DBSession();

    def closeSession(self):
        self.session.commit();
        self.session.close();
        self.session=None;

    def _getBlobsFromArrays(self,neighbor_distance,neighbor_index):
        if neighbor_distance is not None:
            neighbor_distance=np.array(neighbor_distance,dtype='float64').tostring()
        if neighbor_index is not None:
            neighbor_index=np.array(neighbor_index,dtype='int64').tostring()
        return neighbor_distance,neighbor_index


    def insert(self, idx, img_path, layer, nn_experiment_timestamp, trainedClass,  class_idx_imagenet ,class_id_imagenet,caffe_model, class_label_imagenet=None,  class_id_pascal =None, class_idx_pascal =None,  neighbor_index=None,neighbor_distance=None):
        if self.session is None:
            raise Exception('Open a Session First');

        neighbor_distance,neighbor_index=self._getBlobsFromArrays(neighbor_distance,neighbor_index);
        

        new_val = Imagenet(idx=idx, img_path=img_path, layer=layer, nn_experiment_timestamp=nn_experiment_timestamp, trainedClass=trainedClass, class_id_imagenet=class_id_imagenet, class_idx_imagenet=class_idx_imagenet ,caffe_model=caffe_model, class_label_imagenet=class_label_imagenet,  class_id_pascal=class_id_pascal , class_idx_pascal=class_idx_pascal ,  neighbor_index=neighbor_index, neighbor_distance=neighbor_distance)

        self.session.add(new_val)
        self.session.commit();

        return True;
            
    def filter(self,criterion):
        if self.session is None:
            raise Exception('Open a Session First');
        vals=self.session.query(Imagenet).filter(*criterion).all();
        vals=[self._getArraysFromBlobs(val) for val in vals];
        return vals
    
    def _getArraysFromBlobs(self,val):
        if val.neighbor_index is not None:
            val.neighbor_index = np.fromstring(val.neighbor_index,dtype='int64')

        if val.neighbor_distance is not None:
            val.neighbor_distance = np.fromstring(val.neighbor_distance,dtype='float64')

        return val;

    def update(self,criterion,updateVals):

        if self.session is None:
            raise Exception('Open a Session First');

        keys_blobs=[Imagenet.neighbor_distance,Imagenet.neighbor_index]
        
        x=self._getBlobsFromArrays(updateVals.get(keys_blobs[0],None),updateVals.get(keys_blobs[1],None));

        for idx,key_blob in enumerate(keys_blobs):
            if key_blob in updateVals:
                updateVals[key_blob]=x[idx];

        self.session.query(Imagenet).filter(*criterion).update(updateVals);
        return True

    def _getArrayFromBlob(self,column,blob):
        if column.key=='neighbor_index':
            arr = np.fromstring(blob,dtype='int64')
        elif column.key=='neighbor_distance':
            arr = np.fromstring(blob,dtype='float64')
        else:
            arr=None
        return arr

    def select(self,toSelect,criterion=None,distinct=False):
        if self.session is None:
            raise Exception('Open a Session First');
        
        if criterion is None:
            vals=self.session.execute(select(toSelect,distinct=distinct))
        else:
            if len(criterion)==1:
                vals=self.session.execute(select(toSelect,distinct=distinct).where(*criterion))
            else:

                vals=self.session.execute(select(toSelect,distinct=distinct).where(and_(*criterion)))

        val_list=[];
        for val in vals:
            val=list(val);
            for idx,field in enumerate(val):
                if str(toSelect[idx].type)=='VARCHAR':
                    val[idx]=str(field);
                if str(toSelect[idx].type)=='BLOB':
                    val[idx]=self._getArrayFromBlob(toSelect[idx],field)
            val=tuple(val);
            val_list.append(val);

        return val_list

def main():
    
    
    # Session.rollback();
    mani=Imagenet_Manipulator('sqlite:///temp/Imagenet.db');
    mani.openSession();
    # for idx in range(10):
    #     mani.insert( idx=idx, img_path='img_path', layer='layer', nn_experiment_timestamp='nn_experiment_timestamp', trainedClass=True, class_id_imagenet='class_id_imagenet', class_idx_imagenet =0,caffe_model='caffe_model', class_label_imagenet='class_label_imagenet',  class_id_pascal='class_id_pascal', class_idx_pascal=0,  neighbor_index=range(10),neighbor_distance=np.random.rand(10))
    # for idx in range(10):
    #     mani.insert( idx=idx, img_path='img_path', layer='layer', nn_experiment_timestamp='nn_experiment_timestamp', trainedClass=False, class_id_imagenet='class_id_imagenet', class_idx_imagenet =0,caffe_model='caffe_model', class_label_imagenet='class_label_imagenet',  class_id_pascal='class_id_pascal', class_idx_pascal=0,  neighbor_index=range(10),neighbor_distance=np.random.rand(10))

    # mani.closeSession();

    # return
    vals=mani.filter((Imagenet.img_path=='img_path',))
    
    for val in vals:
        # print val
        # for field in val:
        #     print field, type(field)
    #     print '___'
        for val_key in val.__dict__.keys():
            print val_key,val.__dict__[val_key]
    # 
    mani.closeSession();

if __name__=='__main__':
    main();


