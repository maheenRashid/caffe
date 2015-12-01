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

class Pascal3D(Base):
    __tablename__ = 'pascal3d'
    # id = Column(Integer, primary_key=True,autoincrement=True)
    idx = Column(Integer,nullable=False,primary_key=True);
    img_path = Column(String,nullable=False,primary_key=True);
    layer = Column(String,nullable=False,primary_key=True);
    nn_experiment_timestamp = Column(String,nullable=False,primary_key=True);
    class_id = Column(String,nullable=False);
    class_idx= Column(Integer,nullable=False);
    caffe_model=Column(String,nullable=False);

    azimuth= Column(Float);
    neighbor_index=Column(Blob);
    neighbor_distance=Column(Blob);
    trainedClass=Column(Boolean);
    azimuth_differences=Column(Blob);

class Pascal3D_Manipulator(object):

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

    def _getBlobsFromArrays(self,azimuth_differences,neighbor_distance,neighbor_index):
        if azimuth_differences is not None:
            azimuth_differences=np.array(azimuth_differences,dtype='float64').tostring()
        if neighbor_distance is not None:
            neighbor_distance=np.array(neighbor_distance,dtype='float64').tostring()
        if neighbor_index is not None:
            neighbor_index=np.array(neighbor_index,dtype='int64').tostring()
        return azimuth_differences,neighbor_distance,neighbor_index

    def insert(self,idx,img_path,layer,nn_experiment_timestamp,class_id,class_idx,caffe_model, azimuth=None,neighbor_index=None,neighbor_distance=None,trainedClass=None,azimuth_differences=None,commitFlag=True):
        
        if self.session is None:
            raise Exception('Open a Session First');

        azimuth_differences,neighbor_distance,neighbor_index=self._getBlobsFromArrays(azimuth_differences,neighbor_distance,neighbor_index);
        

        new_val = Pascal3D(idx=idx,img_path=img_path,nn_experiment_timestamp=nn_experiment_timestamp,layer=layer,class_id=class_id,class_idx=class_idx,caffe_model=caffe_model, azimuth=azimuth, neighbor_index=neighbor_index, neighbor_distance=neighbor_distance,trainedClass=trainedClass,azimuth_differences=azimuth_differences)

        self.session.add(new_val)
        if commitFlag:
            self.session.commit();

        return True;
            
    def filter(self,criterion):
        if self.session is None:
            raise Exception('Open a Session First');
        vals=self.session.query(Pascal3D).filter(*criterion).all();
        vals=[self._getArraysFromBlobs(val) for val in vals];
        return vals
    
    def _getArraysFromBlobs(self,val):
        if val.neighbor_index is not None:
            val.neighbor_index = np.fromstring(val.neighbor_index,dtype='int64')

        if val.neighbor_distance is not None:
            val.neighbor_distance = np.fromstring(val.neighbor_distance,dtype='float64')

        if val.azimuth_differences is not None:
            val.azimuth_differences = np.fromstring(val.azimuth_differences,dtype='float64')

        return val;

    def update(self,criterion,updateVals):

        if self.session is None:
            raise Exception('Open a Session First');

        keys_blobs=[Pascal3D.azimuth_differences,Pascal3D.neighbor_distance,Pascal3D.neighbor_index]
        
        x=self._getBlobsFromArrays(updateVals.get(keys_blobs[0],None),updateVals.get(keys_blobs[1],None),updateVals.get(keys_blobs[2],None));

        for idx,key_blob in enumerate(keys_blobs):
            if key_blob in updateVals:
                updateVals[key_blob]=x[idx];

        self.session.query(Pascal3D).filter(*criterion).update(updateVals);
        return True

    def _getArrayFromBlob(self,column,blob):
        if column.key=='neighbor_index':
            arr = np.fromstring(blob,dtype='int64')
        elif column.key=='azimuth_differences' or  column.key=='neighbor_distance':
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
    mani=Pascal3D_Manipulator('sqlite:///temp/pascal3d.db');
    mani.openSession();
    # for idx in range(10):
    #     try:
    #         mani.insert(idx=idx,img_path='img_path',layer='layer',nn_experiment_timestamp='nn_experiment_timestamp',class_id='class_id',class_idx=idx,caffe_model='caffe_model', azimuth=0.0,neighbor_index=range(10),neighbor_distance=np.random.rand(10),trainedClass=True,azimuth_differences=np.random.rand(10))
    #     except Exception:
    #         mani.session.rollback();
    #         print 'Hello';


    vals=mani.select((Pascal3D.img_path,Pascal3D.class_id,Pascal3D.azimuth_differences,Pascal3D.neighbor_distance,Pascal3D.neighbor_index),criterion=(Pascal3D.img_path=='img_path',),distinct=False)
    
    for val in vals:
        print val
    #     for field in val:
    #         print field, type(field)
    #     print '___'
    #     # for val_key in val.__dict__.keys():
        #     print val_key,val.__dict__[val_key]
    # 
    mani.closeSession();

if __name__=='__main__':
    main();
