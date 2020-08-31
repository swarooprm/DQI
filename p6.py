import pandas as pd
import numpy as np 
import json
import csv


import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm_notebook as tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download('averaged_perceptron_tagger')

import spacy
import math

import string
import sys
import random

from collections import Counter
from itertools import chain

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

ge=pd.read_csv('./testgoodwordse.csv')
gn=pd.read_csv('./testgoodwordsn.csv')
gc=pd.read_csv('./testgoodwordsc.csv')

be=pd.read_csv('./testbadwordse.csv')
bn=pd.read_csv('./testbadwordsn.csv')
bc=pd.read_csv('./testbadwordsc.csv')

glabel=pd.read_csv('./testgoodwords.csv')
blabel=pd.read_csv('./testbadwords.csv')


nlp = spacy.load("en_trf_bertbaseuncased_lg")

def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))

data = blabel['fullnopunc'].to_list()
df = pd.DataFrame(data,columns=['sentence'])
df = df.dropna(thresh=1)
df['trigrams'] = df['sentence'].map(lambda x: find_ngrams(x.split(" "), 3))
trigrams = df['trigrams'].tolist()
trigrams = list(chain(*trigrams))
trigrams = [(x.lower(), y.lower(),z.lower()) for x,y,z in trigrams]
trigram_counts = Counter(trigrams)
tri=pd.Series(trigram_counts)

datae = be['fullnopunc'].to_list()
dfe = pd.DataFrame(datae,columns=['sentence'])
dfe = dfe.dropna(thresh=1)
dfe['trigrams'] = dfe['sentence'].map(lambda x: find_ngrams(x.split(" "), 3))

datan = bn['fullnopunc'].to_list()
dfn = pd.DataFrame(datan,columns=['sentence'])
dfn = dfn.dropna(thresh=1)
dfn['trigrams'] = dfn['sentence'].map(lambda x: find_ngrams(x.split(" "), 3))

datac = bc['fullnopunc'].to_list()
dfc = pd.DataFrame(datac,columns=['sentence'])
dfc = dfc.dropna(thresh=1)
dfc['trigrams'] = dfc['sentence'].map(lambda x: find_ngrams(x.split(" "), 3))

dfb=pd.DataFrame(tri)
dfb['sentence'] = [' '.join(map(str,i)) for i in dfb.index.tolist()]
arrv=[]
arrsd=[]
for i in tqdm(range(len(dfb))):
    arr=[]
    c1=0
    c2=0
    c3=0
    s=dfb.iloc[i]['sentence']
    bg=s.split()
    for j in range(len(dfe)):
        x=dfe.iloc[j]['trigrams']
        for k in range(len(x)):
            if(bg==list(x[k])):
                c1=c1+1
    for j in range(len(dfn)):
        x=dfn.iloc[j]['trigrams']
        for k in range(len(x)):
            if(bg==list(x[k])):
                c2=c2+1
    for j in range(len(dfc)):
        x=dfc.iloc[j]['trigrams']
        for k in range(len(x)):
            if(bg==list(x[k])):
                c3=c3+1    
    if(c1+c2+c3>1):
        arr.append(c1)
        arr.append(c2)
        arr.append(c3)
        v=pd.Series(arr).var()
        sd=pd.Series(arr).std()
        arrv.append(v)
        arrsd.append(sd)
    else:
        arrv.append(0)
        arrsd.append(0)
btgv=arrv
btgstd=arrsd
print(pd.Series(arrv).sum()/len(df))
print(pd.Series(arrsd).sum()/len(df))

badvar=pd.DataFrame(data=btgv,columns=['var'])
badvar.to_csv("./p6t3bv.csv")
badvar=pd.DataFrame(data=btgstd,columns=['std'])
badvar.to_csv("./p6t3bs.csv")
