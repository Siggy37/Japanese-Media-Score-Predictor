#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:30:06 2019

@author: siggy
"""

import fasttext
import numpy as np

def create_model(text, save=True):
    assert save in (True, False)
    model = fasttext.train_unsupervised(text, model='skipgram')
    if save == True:
        model.save_model("anime_fasttext.bin")
        return None
    elif save == False:      
        return model

def get_summary_tensor(text, load_model=True, save=False):
    assert load_model in (True, False)
    assert save in (True, False)
    
    tensor = list()
    text = text.split('\n')
    if load_model == True:
        model = fasttext.load_model("anime_fasttext.bin")
    elif load_model == False:
        model = fasttext.train_unsupervised(text)
    for sent in text[1:]:
        tensor.append(model[sent])
    
    if save == True:
        np.save("summary_tensor.npy", np.array(tensor))
    return np.array(tensor)
       
     
    
    
    
    

if __name__ == '__main__':
    textfile = 'summary_output.txt'
    create_model(textfile)

    