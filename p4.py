import pandas as pd
import numpy as np 
import json
import csv
import argparse

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

nlp = spacy.load("en_trf_bertbaseuncased_lg")

@jit
def parameter4(sentence,size,WSIML,wordpath,dqipath):
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
            sm=sm-WSIML
            sl=sl+sm
        lists.append(arr)
        vals.append(size/sl)
    df=pd.DataFrame(lists)
    df.to_csv(wordpath)
    df=pd.DataFrame(vals)
    df.to_csv(dqipath)  

def main():                                                                     
    parser = argparse.ArgumentParser()                                          
                                                                                
    ## Required parameters                                                      
    parser.add_argument("--dataset_path",                                       
                        type=str,                                               
                        required=True,                                          
                        help="Which dataset to attack.")                        
    parser.add_argument("--output_path",                                        
                        type=str,                                               
                        required=True,                                          
                        help="Where to save")                                   
    args = parser.parse_args()                                                  
                                                                                
                                                                                   
    gwords=pd.read_csv(args.dataset_path+".csv")  

    size=len(gwords)
    WSIML=0.5
    sentence=gwords['fullnopunc']
    wordpath=args.output_path+"/word.csv"
    dqipath=args.output_path+"/dqi.csv"
    parameter4(sentence,size,WSIML,wordpath,dqipath)

if __name__ == "__main__":
    main()
