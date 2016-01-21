import os;
import numpy as np;
import scipy.stats
import scipy.io
import cPickle as pickle
import copy
from scipy import misc;
import visualize;
import math;
import random;
import time;
import util;
from tube_db import Tube, Tube_Manipulator,TubeHash_Manipulator,TubeHash
from collections import namedtuple
import lsh;
import matplotlib.pyplot as plt;
import cudarray as ca
import nearest_neighbor
import multiprocessing
from collections import Counter
import itertools

def getBestTubeAvgScore(score_file):
	scores=pickle.load(open(score_file));
	tube_keys=scores.keys();
	tube_scores=[np.mean(scores[tube_id]) for tube_id in tube_keys];
	sort_idx=np.argsort(tube_scores)[::-1];
	tubes_ranked=np.array(tube_keys)[sort_idx];
	scores_ranked=np.array(tube_scores)[sort_idx];
	return tubes_ranked[0],tubes_ranked,scores_ranked

def main():
	shot_dir='/disk2/januaryExperiments/shot_score_normalized';
	class_idx=0
	video_id=1
	shot_id=1
	score_file=os.path.join(shot_dir,str(class_idx)+'_'+str(video_id)+'_'+str(shot_id)+'.p');
	best_tube_rank,tubes_ranked,scores_ranked = getBestTubeAvgScore(os.path.join(shot_dir,score_file))
	


if __name__=='__main__':
	main();