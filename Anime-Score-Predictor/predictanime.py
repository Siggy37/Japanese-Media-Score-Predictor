#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:27:10 2019

@author: siggy
"""
import torch
import fasttext
from model1 import predictor
import wordencoding as we
import numpy as np
import json
import animedata as ad


class AnimePredictor:
    def __init__(self, model=None):
        self.model = None
        self.optimizer = None
        

    def load_inference_model(self, modelpath):
        model = predictor(1, 11)
        optimizer = torch.optim.Adam(model.parameters(), lr = .0001)
        info_dict = torch.load(modelpath)
        model.load_state_dict(info_dict['model_state_dict'])
        optimizer.load_state_dict(info_dict['optimizer_state_dict'])
        self.model = model
        self.optimizer = optimizer

    def format_inference_data(self, title, genres, platform, summary):
        valid_platforms = ["OAV", "TV", "special", "movie", "ONA"]
        embedding_model = fasttext.load_model("anime_fasttext.bin")
        BoW = np.load("BoW.npy")
        BoW = BoW.tolist()
        genre_map = ad.clean_genres()
        title = we.get_title_vector(title, BoW)
        genre = we.get_genre_vector(genres, genre_map)
        platform_vec = np.zeros(len(valid_platforms))
        platform_vec[valid_platforms.index(platform)] += 1  
        pv = np.array(platform_vec)
        summary = embedding_model[summary]        
        title = torch.Tensor(title).float()
        title = title.view(1, -1)
        genre = torch.Tensor(genre).float()
        genre = genre.view(1, -1)
        pv = torch.Tensor(pv).float()
        pv = pv.view(1, -1)
        summary = torch.Tensor(summary).float()
        summary = summary.view(-1, 1, 100)
        
        
        return title, genre, pv, summary
    
    def get_score(self, title, genres, platform, summary):
        tiers = ['Masterpiece', 'Excellent', 'Very Good', 'Good', 'Decent',
             'So-so', 'Not Very Good', 'Weak', 'Bad', 'Awful', 'Worst Ever'] 
        #ap = AnimePredictor()
        self.load_inference_model("ModelState3.pth")
        t, g, p, s = self.format_inference_data(title, genres, platform, summary)
       # print(t.shape)
        #print(g.shape)
        #print(p.shape)
        #print(s.shape)
        self.model.eval()
        outcome = self.model(s, g, t, p)
        print(outcome)
        outcome = outcome.tolist()[0]        
        idx = outcome.index(max(outcome))
        return tiers[idx]        
    
    
if __name__ == '__main__':
    """
    ap = AnimePredictor()
    ap.load_inference_model("model_info.pth")
    t, g, p, s = ap.format_inference_data("Naruto", ["action"], "TV", "Kid kicks ass")
    print(t.shape)
    print(g.shape)
    print(p.shape)
    print(s.shape)
    model = ap.model.eval()
    outcome = model(s, g, t, p)
    print(max(outcome.tolist()[0]))
    """
    ap = AnimePredictor()
    print(ap.get_score("Full Metal Alchemist", ["horror"], "special", ""))
    
    
    
    
    
    