import pickle
import torch                                                                    
from transformers import RobertaModel, RobertaTokenizer                         
from sklearn.metrics.pairwise import cosine_similarity                          

import pandas as pd
#with open("path_to_file/my_matrix.dat",'rb') as f:
#    cos_sim_matrix = pickle.load(f)

#print(cos_sim_matrix.shape)
import numpy as np
import sys
from numba import jit, cuda
from tqdm import tqdm

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')                  

                       
@jit
def cosine_similarity_n_space(m1, m2, batch_size=100):
    print(m1.shape)
    print(m2.shape)
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
    for row_i in tqdm(range(0, int(m1.shape[0] / batch_size) + 1)):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        ret[start: end] = sim
    return ret

#product = cosine_similarity_n_space(embeddings,embeddings,batch_size=128)
#np.save(('path_to_file/cos_sim_counter_fitting_1.npy'), product)
# Read the files
#train_df = pd.read_csv('path_to_file/snli_1.0_train.txt', sep='\t')
#test_df = pd.read_csv('path_to_file/snli_1.0_test.txt', sep='\t')
train_df = pd.read_csv('path_to_file/testbad.csv')
print(train_df['sentence1'])
print(train_df['sentence2'])
#train_df = pd.read_csv('path_to_file/snli_data_adv/snli_1.0/snli_1.0_train.txt', sep='\t')
sentence_list = []
sentence_list.extend(train_df['sentence1'])
sentence_list.extend(train_df['sentence2']) 
print(sentence_list[:5])



model = RobertaModel.from_pretrained('roberta-large', output_hidden_states=True)
                                                                                
# Set the device to GPU (cuda) if available, otherwise stick with CPU           
device = 'cuda' if torch.cuda.is_available() else 'cpu'                         

model = model.to(device)                                                        
"""
sentence_ids = []
for sentence in test_df['sentence1']:                                                  
        sentence_enc = tokenizer.encode(sentence,add_special_tokens=True)               
        sentence_ids.append(sentence_enc)   

test_id_tensors = []                                                            
for sentence_id in sentence_ids:                                                        
    test_id_tensors.append(torch.LongTensor(sentence_id))
     
for i in range(0,len(test_id_tensors)):                                         
    test_id_tensors[i] = test_id_tensors[i].to(device)   

for i in range(0,len(test_id_tensors)):                                         
    test_id_tensors[i] = test_id_tensors[i].unsqueeze(0)
"""
sentence_ids = []                                                               
for sentence in sentence_list:
    try:
        sentence_enc = tokenizer.encode(sentence,add_special_tokens=True)       
        sentence_ids.append(sentence_enc)                                       
    except:
        print(sentence)
        continue    
train_id_tensors = []                                                            
for sentence_id in sentence_ids:                                                
    train_id_tensors.append(torch.LongTensor(sentence_id))                       

for i in range(0,len(train_id_tensors)):                                         
    train_id_tensors[i] = train_id_tensors[i].to(device)                          
                                                                                
for i in range(0,len(train_id_tensors)):                                         
    train_id_tensors[i] = train_id_tensors[i].unsqueeze(0)      

def get_embeddings_from_id_tensors(word_id_tensors):
    sentence_embedding_list = []
    for word_id_tensor in tqdm(word_id_tensors):
        with torch.no_grad():
            out = model(input_ids=word_id_tensor)                               
            hidden_states = out[2]                                              
            sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze() 
            sentence_embedding_list.append(sentence_embedding.cpu().numpy())
    return sentence_embedding_list                

#test_embedding_list = np.array(get_embeddings_from_id_tensors(test_id_tensors))
train_embedding_list = np.array(get_embeddings_from_id_tensors(train_id_tensors))
print(train_embedding_list[:10])
#product = cosine_similarity_n_space(test_embedding_list,test_embedding_list,batch_size=128)
product = cosine_similarity_n_space(train_embedding_list,train_embedding_list,batch_size=128)
np.save(('path_to_file/test_similarity_bad_a.npy'), product) 
