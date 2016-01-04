import os
import sys
from sqlalchemy import Column, Float, Integer, String,Boolean,func
from sqlalchemy.dialects.mysql import BLOB as Blob
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine,and_
import sqlalchemy
# distinct
from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.sql import select
import sqlalchemy
from sqlalchemy.schema import ForeignKey
import numpy as np;
import time
import random
# from tubehash_db import TubeHash

Base = declarative_base()


class Tube(Base):
    __tablename__ = 'Tube'
    
    idx = Column(Integer,nullable=False,primary_key=True);
    img_path = Column(String,nullable=False,primary_key=True);
    frame_id=Column(Integer,nullable=False,primary_key=True);
    video_id=Column(Integer,nullable=False,primary_key=True);
    tube_id=Column(Integer,nullable=False,primary_key=True);
    shot_id=Column(Integer,nullable=False,primary_key=True);

    frame_path=Column(String);
    layer = Column(String);
    deep_features_path=Column(String);
    deep_features_idx=Column(Integer);

    nn_file_path = Column(String);
    
    class_id_pascal = Column(String);
    class_idx_pascal =  Column(Integer);
    
    neighbor_index=Column(Blob);
    neighbor_distance=Column(Blob);
    neighbor_index_in_db=Column(Blob);
    children = relationship("TubeHash");

class TubeHash(Base):
    __tablename__ = 'TubeHash'
    # 'user_id', Integer, ForeignKey("user.user_id")
    idx = Column(Integer,ForeignKey('Tube.idx'),nullable=False,primary_key=True);
    hash_table = Column(Integer,nullable=False,primary_key=True);
    hash_val = Column(Integer);


class Tube_Manipulator(object):

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

    def _getBlobsFromArrays(self,neighbor_distance,neighbor_index,neighbor_index_in_db):
        if neighbor_distance is not None:
            neighbor_distance=np.array(neighbor_distance,dtype='float64').tostring()

        if neighbor_index is not None:
            neighbor_index=np.array(neighbor_index,dtype='int64').tostring()

        if neighbor_index_in_db is not None:
            neighbor_index_in_db=np.array(neighbor_index_in_db,dtype='int64').tostring()
        
        return neighbor_distance,neighbor_index,neighbor_index_in_db


    def insert(self, idx, img_path, frame_id, video_id, tube_id, shot_id, frame_path=None, layer=None, deep_features_path=None, deep_features_idx=None, nn_file_path=None, class_id_pascal=None, class_idx_pascal=None, neighbor_index=None, neighbor_distance=None, neighbor_index_in_db=None,commit=True):

        
        if self.session is None:
            raise Exception('Open a Session First');

        neighbor_distance,neighbor_index,neighbor_index_in_db=self._getBlobsFromArrays(neighbor_distance,neighbor_index,neighbor_index_in_db);
        

        new_val= Tube(idx=idx,
                    img_path=img_path,
                    frame_id=frame_id,
                    video_id=video_id,
                    tube_id=tube_id,
                    shot_id=shot_id,
                    frame_path=frame_path,
                    layer=layer,
                    deep_features_path=deep_features_path,
                    deep_features_idx=deep_features_idx,
                    nn_file_path=nn_file_path,
                    class_id_pascal=class_id_pascal,
                    class_idx_pascal=class_idx_pascal,
                    neighbor_index=neighbor_index,
                    neighbor_distance=neighbor_distance,
                    neighbor_index_in_db=neighbor_index_in_db)

        self.session.add(new_val)
        if commit:
            self.session.commit();

        return True;
            
    def filter(self,criterion):
        if self.session is None:
            raise Exception('Open a Session First');
        vals=self.session.query(Tube).filter(*criterion).all();
        vals=[self._getArraysFromBlobs(val) for val in vals];
        return vals
    
    def _getArraysFromBlobs(self,val):
        if val.neighbor_index is not None:
            val.neighbor_index = np.fromstring(val.neighbor_index,dtype='int64')

        if val.neighbor_distance is not None:
            val.neighbor_distance = np.fromstring(val.neighbor_distance,dtype='float64')

        if val.neighbor_index_in_db is not None:
            val.neighbor_index_in_db = np.fromstring(val.neighbor_index_in_db,dtype='int64')

        return val;

    def update(self,criterion,updateVals):

        if self.session is None:
            raise Exception('Open a Session First');

        keys_blobs=[Tube.neighbor_distance,Tube.neighbor_index,Tube.neighbor_index_in_db]
        
        x=self._getBlobsFromArrays(updateVals.get(keys_blobs[0],None),updateVals.get(keys_blobs[1],None),updateVals.get(keys_blobs[2],None));

        for idx,key_blob in enumerate(keys_blobs):
            if key_blob in updateVals:
                updateVals[key_blob]=x[idx];
        print updateVals
        self.session.query(Tube).filter(*criterion).update(updateVals);
        return True

    def _getArrayFromBlob(self,column,blob):
        if column.key=='neighbor_index':
            arr = np.fromstring(blob,dtype='int64')
        elif column.key=='neighbor_distance':
            arr = np.fromstring(blob,dtype='float64')
        elif column.key=='neighbor_index_in_db':
            arr = np.fromstring(blob,dtype='int64')
        else:
            arr=None
        return arr

    def select(self,toSelect,criterion=None,distinct=False,limit=None):
        if self.session is None:
            raise Exception('Open a Session First');
        
        query=select(toSelect,distinct=distinct);
        if criterion is not None:
            if len(criterion)==1:
                query=query.where(*criterion);
            else:
                query=query.where(and_(*criterion));
        
        if limit is not None:
            query=query.limit(limit);

        vals=self.session.execute(query);
        
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

    def selectMix(self,toSelect,criterion,limit=None):
        if self.session is None:
            raise Exception('Open a Session First');

        query=self.session.query(*toSelect).join(TubeHash,Tube.idx==TubeHash.idx).filter(*criterion)
        if limit is not None:
            query=query.limit(limit);
        vals=self.session.execute(query)
        vals=[val for val in vals];

        return vals

    def count(self,toSelect=None,criterion=None,distinct=False,mix=False):
        if self.session is None:
            raise Exception('Open a Session First');

        if distinct:
            query=self.session.query(sqlalchemy.distinct(*toSelect))
        else:
            query=self.session.query(*toSelect)

        if mix:
            query=query.join(TubeHash,Tube.idx==TubeHash.idx)
        
        if criterion is not None:
            query=query.filter(*criterion)
        # print query
        count=query.count();
       
        return count        


class TubeHash_Manipulator(object):

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

    def insert(self, idx, hash_table,hash_val,commit=True):

        
        if self.session is None:
            raise Exception('Open a Session First');

        new_val= TubeHash(idx=idx,
                    hash_table=hash_table,
                    hash_val=hash_val)

        self.session.add(new_val)
        if commit:
            self.session.commit();

        return True;
            
    def filter(self,criterion):
        if self.session is None:
            raise Exception('Open a Session First');
        vals=self.session.query(TubeHash).filter(*criterion).all();
        return vals
    
    def update(self,criterion,updateVals):

        if self.session is None:
            raise Exception('Open a Session First');

        self.session.query(TubeHash).filter(*criterion).update(updateVals);
        return True

    def select(self,toSelect,criterion=None,distinct=False,limit=None):
        if self.session is None:
            raise Exception('Open a Session First');
        
        query=select(toSelect,distinct=distinct);
        if criterion is not None:
            if len(criterion)==1:
                query=query.where(*criterion);
            else:
                query=query.where(and_(*criterion));
        
        if limit is not None:
            query=query.limit(limit);
        
        vals=self.session.execute(query);
        vals=[val for val in vals];
        return vals

    def selectMix(self,toSelect,criterion,limit=None,distinct=False):
        if self.session is None:
            raise Exception('Open a Session First');
        
        if distinct:
            query=self.session.query(sqlalchemy.distinct(*toSelect))
        else:
            query=self.session.query(*toSelect)

        query=query.join(Tube,Tube.idx==TubeHash.idx).filter(*criterion)

        if limit is not None:
            query=query.limit(limit);
        # print query
        vals=self.session.execute(query)
        
        vals=[val for val in vals];

        return vals
        
    def count(self,toSelect=None,criterion=None,distinct=False,mix=False):
        if self.session is None:
            raise Exception('Open a Session First');

        if distinct:
            query=self.session.query(sqlalchemy.distinct(*toSelect))
        else:
            query=self.session.query(*toSelect)

        if mix:
            query=query.join(Tube,Tube.idx==TubeHash.idx)
        
        if criterion is not None:
            query=query.filter(*criterion)
        # print query
        count=query.count();
       
        return count        



def main():


    mani=TubeHash_Manipulator('sqlite:///temp/Tube.db');
    mani.openSession();
    criterion=(TubeHash.hash_table>1,TubeHash.hash_val>1)
    # ,Tube.video_id==0)
    toSelect=(TubeHash.idx,)
    distinct=True
    count_new=mani.count(toSelect,criterion,mix=False,distinct=distinct);
    print count_new
    vals_new=mani.selectMix(toSelect,criterion,distinct=distinct);
    print vals_new,len(vals_new)
    vals_new=mani.select(toSelect,criterion,distinct=distinct);
    print vals_new,len(vals_new)
    mani.closeSession();
    
    # return


    return
    mani=TubeHash_Manipulator('sqlite:///temp/Tube.db');
    mani.openSession();
    mani.insert(idx=0,hash_table=3,hash_val=3);
    mani.closeSession();
    return
    idx=0;
    img_path='img_path';
    frame_id=0;
    video_id=0;
    tube_id=0;
    shot_id=0;
    frame_path='frame_path';
    layer='layer';
    deep_features_path='deep_features_path';
    deep_features_idx=0;
    nn_file_path='nn_file_path';
    class_id_pascal='class_id_pascal';
    class_idx_pascal=0;
    neighbor_index=np.array(range(10));
    neighbor_distance=np.random.rand(10);
    neighbor_index_in_db=np.array(range(10));
    
    for idx in range(10):
        vals=mani.filter((Tube.idx==idx,));
        print len(vals);
        for val in vals:
            print val;
            for key_curr in val.__dict__:
                print key_curr,val.__dict__[key_curr];

        vals=mani.select((Tube.idx,Tube.img_path),(Tube.idx==idx,));
        print len(vals);
        print vals;

        mani.update((Tube.idx==idx,),{Tube.img_path:'new_img_path',Tube.neighbor_distance:[1.0,2.0,3.0,4.0,5.0],Tube.neighbor_index:range(5),Tube.neighbor_index_in_db:range(5)});
        print mani.select((Tube.neighbor_distance,),(Tube.idx==idx,));
        print mani.select((Tube.neighbor_index,),(Tube.idx==idx,));
        print mani.select((Tube.neighbor_index_in_db,),(Tube.idx==idx,));
        print '____';

    mani.closeSession();

if __name__=='__main__':
    main();


