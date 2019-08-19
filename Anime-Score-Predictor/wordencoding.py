# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 01:40:09 2019

@author: brand
"""
import animedata as ad
import numpy as np

def tokenize_titles():
    items = ad.get_titles()
    BoW = list()
    for ID in items:
        info = items[ID]
        title = info[0]
        title = title.split(' ')
        for word in title:
            i = len(word)
            while i > 0:
                BoW.append(word[:i])
                i -= 1
    
    return sorted(set(BoW))

def get_title_vector(title, BoW):    
    vec = list(np.zeros(len(BoW)))
    for word in title.split():
        i = len(word)
        while i > 0:                
            if word[:i] in BoW:
                vec[BoW.index(word[:i])] += 1
            i -= 1    
    return np.array(vec)

def get_genre_vector(cur_genres, genre_map):
    vec = list(np.zeros(len(genre_map)))
    for g in cur_genres:
        if len(g) > 0:
            if g.lower() in genre_map:
                vec[genre_map.index(g.lower())] = 1        
    return np.array(vec)

def get_all_genre_vecs():
    vecs = list()
    genre_dict = ad.get_genres()
    genre_map = ad.clean_genres()
    for ID in ad.get_ids():
        genres = genre_dict[ID]
        vec = get_genre_vector(genres, genre_map)
        vecs.append(vec)        
    
    
    return np.array(vecs)

def get_all_title_vecs(platform_len):
    title_vecs = list()
    platform_vecs = list()
    title_map = ad.get_titles()
    BoW = tokenize_titles()
    for ID in ad.get_ids():
        title = title_map[ID][0]
        platform_idx = title_map[ID][2]
        title_vec = get_title_vector(title, BoW)
        title_vecs.append(title_vec)
        platform_vec = list(np.zeros(platform_len))
        platform_vec[platform_idx] += 1
        platform_vecs.append(platform_vec)
    return np.array(title_vecs), np.array(platform_vecs)
        

def create_summary_file(output_file):
    summaries = ad.get_summaries()
    of = open(output_file, 'w', encoding='utf-8')
    for ID in summaries:
        summ = summaries[ID]
        if summ != 'Synopsis':
            of.write(summaries[ID] + '\n')
    of.close()

def get_all_tags():
    tag_list = list()
    tag_dict = ad.normalize_scores()
    for ID in tag_dict:
        info = tag_dict[ID]
        loc = info[0]
        tag_list.append(loc)
    return tag_list

def get_all_data():
    summary       = np.load("summary_tensor.npy")
    title_info    = get_all_title_vecs(5)
    title_vecs    = title_info[0]
    platform_vecs = title_info[1]
    genre_vecs    = get_all_genre_vecs()
    tags          = get_all_tags()
    
    return summary, title_vecs, platform_vecs, genre_vecs, tags
    

    
    
if __name__ == "__main__":

   
   gs = ["Super robot", "police"]
   genre_map = ad.clean_genres()
   print(len(genre_map))
   
    
    
    
    