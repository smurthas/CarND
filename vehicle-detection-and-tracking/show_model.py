""" display a model """
import sys
import pickle

with open(sys.argv[1], 'rb') as fid:
    clf = pickle.load(fid)
    print(clf)
