import pandas as pd
import numpy as np 
import json
import csv

import matplotlib.pyplot as plt
# import seaborn as sns

from tqdm import tqdm


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
import nltk
# nltk.download('averaged_perceptron_tagger')

import spacy
import math

import string
import sys
import random
import pickle


from collections import Counter
from itertools import chain

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from sklearn.metrics.pairwise import cosine_similarity

from numba import jit, cuda

gwords=pd.read_csv("path_to_dev_good_words/dev_good_words.csv")

nlp = spacy.load("en_trf_bertbaseuncased_lg")

@jit
def parameter4(sentence,size,WSIML):
    lists=[]
    vals=[]
    for x in tqdm(range(len(sentence))):
        arr=[]
        s=sentence[x]
        word = s.split()
        length=len(word)
        sl=0
        for i in (range(length)):
            sm=0
            token_l=nlp(word[i])
            for j in (range(length)):
                if(i!=j):
                    token_m=nlp(word[j])
                    sm=sm+token_l.similarity(token_m)
            sm=sm/length
            arr.append(sm)
            sm=abs(sm-WSIML)
            sl=sl+sm
        lists.append(arr)
        vals.append(size/sl)
    df=pd.DataFrame(lists)
    df.to_csv("path_to_wordsimgsabs/wordsimgabs.csv")
    df=pd.DataFrame(vals)
    df.to_csv("path_to_dqic4gabs/dqic4gabs.csv")  

size=len(gwords)
WSIML=0.5
sentence=gwords['fullnopunc']
parameter4(sentence,size,WSIML)
