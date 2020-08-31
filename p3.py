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

words=pd.read_csv("./words.csv")
snlitrain_l_list = []
pandas_json_attempt = []
with open('./snli_1.0_test.jsonl') as snlitrain_file_pointer:
    for item in snlitrain_file_pointer:
        snlitrain_l_list.append(item)
data = []
for item in snlitrain_l_list:
    data.append(json.loads(item))
df_snlitrain = pd.DataFrame.from_dict(data)
nlp = spacy.load("en_trf_bertbaseuncased_lg")

@jit
def func():
    x=[]
    for l in tqdm(range(len(data))):
        f=0
        token_l=nlp(data[l])
        for m in tqdm(range(len(data))):
            if(m!=l):
                token_m=nlp(data[m])
                slm=token_l.similarity(token_m)                        
                diff=SIML-slm
                if(diff<=0):
                    f=f+1
        x.append(f)
    t = pd.Series(x)
    tf1=t.var()
    return tf1

@jit
def amaxelements(list1, N): 
    sa=0
    for i in range(0, N):  
        max1 = 0  
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
        list1.remove(max1); 
        sa=sa+max1
    return sa

@jit
def func2():
    a=3
    sl=0
    for l in tqdm(range(len(data))):
        x=[]
        token_l=nlp(data[l])
        for m in tqdm(range(len(data))):
            if(m!=l):
                token_m=nlp(data[m])
                slm=token_l.similarity(token_m)                        
                diff1=SIML-slm
                diff2=abs(diff1)
                diff=abs(diff1-diff2)
                x.append(diff)
        sl=sl+amaxelements(x,a)
    tf2=sl/5

data=words['fullnopunc']
SIML=0.8

tf1 = func()
tf2 = func2()
print(tf1)
print(tf2)
dqic3=tf1+tf2
print(dqic3)
