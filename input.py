import random
import numpy as np
import pickle
from pickle import Pickler as pkl

def get_train_data():
    emb = pickle.load(open('data/ned/train_wvec','rb'))
    features = pickle.load(open('data/ned/train_features','rb'))
    return emb,features

def get_test_data():
    emb = pickle.load(open('data/ned/testa_wvec','rb'))
    features = pickle.load(open('data/ned/testa_features','rb'))
    return emb,features

def get_final_data():
    emb = pickle.load(open('data/ned/testb_wvec','rb'))
    features = pickle.load(open('data/ned/testb_features','rb'))
    return emb,features
